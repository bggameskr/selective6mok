(() => {
  'use strict';

  const CORE = window.SIXMOK_MODEL_CORE;
  const MODEL_URL = 'sixmok.onnx?v=6.2-policy20-opening35-t1-inspector-greedy';
  const MODEL_BOARD_SIZE = 19;
  const DEFAULT_POLICY_CUTOFF_REL = 0.20;
  const OPENING_POLICY_CUTOFF_REL = 0.35;

  if (!CORE || typeof ort === 'undefined') {
    window.MODEL_STATUS = 'failed';
    window.MODEL_STATUS_DETAIL = '모델 분석 코어를 초기화하지 못했습니다.';
    window.dispatchEvent(new CustomEvent('model-status'));
    return;
  }

  let sessionPromise = null;
  let cachedInputPlanes = null;
  let lastKey = null;
  let lastPromise = null;

  function announce(status, detail = '') {
    window.MODEL_STATUS = status;
    window.MODEL_STATUS_DETAIL = detail;
    window.dispatchEvent(new CustomEvent('model-status'));
  }
  function session() {
    if (!sessionPromise) sessionPromise = ort.InferenceSession.create(MODEL_URL, { executionProviders:['wasm'] });
    return sessionPromise;
  }
  async function evaluate(games) {
    if (!games.length) return { policy:new Float32Array(), value:new Float32Array(), stride:0 };
    const n = games[0].size;
    const sess = await session();
    async function runWithPlanes(planes) {
      const planeSize = planes * n * n;
      const data = new Float32Array(games.length * planeSize);
      games.forEach((g, k) => data.set(CORE.encodeStateForModel(g, planes), k * planeSize));
      return sess.run({ board:new ort.Tensor('float32', data, [games.length, planes, n, n]) });
    }
    let out;
    if (cachedInputPlanes !== null) out = await runWithPlanes(cachedInputPlanes);
    else {
      let err13;
      try { out = await runWithPlanes(13); cachedInputPlanes = 13; }
      catch (err) {
        err13 = err;
        try { out = await runWithPlanes(16); cachedInputPlanes = 16; }
        catch (err16) { throw new Error(`ONNX 입력 규격 확인 실패 (13-plane: ${err13.message}; 16-plane: ${err16.message})`); }
      }
    }
    return { policy:out.policy.data, value:out.value.data, stride:n * n + 1 };
  }
  function stateKey(game) {
    return [game.current, game.placedThisTurn, game.stones[BLACK], game.stones[WHITE], game.stonesOnBoard, Array.from(game.board).join('')].join('|');
  }
  function rootPerspectiveTerminal(child, rootColor) {
    if (!child.isOver) return null;
    if (child.winner === null) return 0;
    return child.winner === rootColor ? 1 : -1;
  }
  function openingCandidates(game, probs) {
    const endIndex = game.size * game.size;
    if (!game.canPlace()) return [endIndex];
    let best = 0;
    for (let i = 0; i < endIndex; i++) if (game.board[i] === EMPTY) best = Math.max(best, Number(probs[i]) || 0);
    const threshold = best * OPENING_POLICY_CUTOFF_REL;
    const out = [];
    for (let i = 0; i < endIndex; i++) if (game.board[i] === EMPTY && (Number(probs[i]) || 0) >= threshold) out.push(i);
    out.push(endIndex);
    return out;
  }
  function actionLabel(action, n) { return action === n * n ? 'END_TURN' : `${Math.floor(action / n) + 1}·${action % n + 1}`; }
  function reasonFor(game, action, immediateWin) {
    if (action === game.size * game.size) return 'END_TURN';
    if (action === immediateWin) return '즉시승 강제수';
    if (game.stonesOnBoard === 0) return '오프닝 35% policy cutoff 통과';
    return 'policy 20%/전술 필터 통과';
  }

  async function analyzeUncached(game) {
    if (game.size !== MODEL_BOARD_SIZE) throw new Error(`모델은 ${MODEL_BOARD_SIZE}줄 판만 지원합니다.`);
    const n = game.size, endIndex = n * n, rootColor = game.current;
    const rootEval = await evaluate([game]);
    const logits = rootEval.policy;
    const probs = CORE.legalProbs(logits, game, 0, rootEval.stride);
    const rootValue = Number(rootEval.value[0]);

    const legalActions = [];
    if (game.canPlace()) for (let i = 0; i < endIndex; i++) if (game.board[i] === EMPTY) legalActions.push(i);
    legalActions.push(endIndex);
    const policyOrder = [...legalActions].sort((a,b) => ((Number(probs[b])||0) - (Number(probs[a])||0)) || (a-b));
    const policyRank = new Map(policyOrder.map((action,i) => [action, i+1]));

    const immediateWin = game.canPlace() && game.stonesOnBoard > 0 ? CORE.findImmediateWin(game) : -1;
    let candidates, decisionPath, needsChildValue = true;
    if (!game.canPlace()) { candidates=[endIndex]; decisionPath='착수 불가 → END_TURN'; needsChildValue=false; }
    else if (game.stonesOnBoard === 0) { candidates=openingCandidates(game, probs); decisionPath='오프닝 root policy'; needsChildValue=false; }
    else if (immediateWin >= 0) { candidates=[immediateWin]; decisionPath='즉시승 강제수'; needsChildValue=false; }
    else { candidates=CORE.candidateActions(game, probs, DEFAULT_POLICY_CUTOFF_REL); decisionPath='1-ply P×V'; }

    const childValues = new Map();
    let inferredChildren = 0;
    if (immediateWin >= 0 && candidates.length === 1 && candidates[0] === immediateWin) childValues.set(immediateWin, 1);
    else if (needsChildValue) {
      const children = candidates.map((action) => CORE.playEvaluationAction(game, action));
      const pending = [], pendingActions = [];
      children.forEach((child, idx) => {
        const terminal = rootPerspectiveTerminal(child, rootColor);
        if (terminal !== null) childValues.set(candidates[idx], terminal);
        else { pending.push(child); pendingActions.push(candidates[idx]); }
      });
      inferredChildren = pending.length;
      if (pending.length) {
        const ev = await evaluate(pending);
        pending.forEach((child,j) => {
          const v = Number(ev.value[j]);
          childValues.set(pendingActions[j], child.current === rootColor ? v : -v);
        });
      }
    }

    const scored = candidates.map((action) => {
      const policy = Number(probs[action]) || 0;
      const modifier = CORE.actionModifier(game, action);
      const childValue = childValues.has(action) ? Number(childValues.get(action)) : null;
      const winProb = childValue === null ? null : CORE.valueWinProb(childValue);
      const pv = childValue === null ? null : CORE.policyValueWeight(policy, childValue);
      const score = game.stonesOnBoard === 0 ? policy * modifier : (childValue === null ? policy * modifier : pv * modifier);
      return { action, label:actionLabel(action,n), rawLogit:Number(logits[action]), policy, policyRank:policyRank.get(action)||null, childValue, winProb, pv, modifier, score, reason:reasonFor(game, action, immediateWin) };
    }).sort((a,b) => (b.score-a.score) || (a.action-b.action));
    scored.forEach((row,i) => { row.candidateRank=i+1; });
    const greedy = scored[0] || { action:endIndex, label:'END_TURN', rawLogit:0, policy:0, policyRank:null, childValue:null, winProb:null, pv:null, modifier:1, score:0, reason:'착수 불가', candidateRank:1 };

    return {
      modelUrl:MODEL_URL, inputPlanes:cachedInputPlanes, rootValue, rootWinProb:CORE.valueWinProb(rootValue), legalCount:legalActions.length,
      candidateCount:scored.length, childEvaluatedCount:inferredChildren, decisionPath,
      policyCutoffRel:game.stonesOnBoard === 0 ? OPENING_POLICY_CUTOFF_REL : DEFAULT_POLICY_CUTOFF_REL,
      greedyMove:greedy.action === endIndex ? END_TURN : [Math.floor(greedy.action/n), greedy.action%n], greedy,
      candidates:needsChildValue ? scored : [],
    };
  }

  async function analyze(game) {
    const key = stateKey(game);
    if (key === lastKey && lastPromise) return lastPromise;
    lastKey = key;
    lastPromise = analyzeUncached(game).catch((err) => { if (lastKey === key) { lastKey=null; lastPromise=null; } throw err; });
    return lastPromise;
  }

  window.MODEL_ANALYZE_POSITION = analyze;
  window.MODEL_PLAN_TURN = async (game) => {
    const result = await analyze(game.clone ? game.clone() : game);
    return result.greedyMove === END_TURN ? [] : [result.greedyMove];
  };
  announce('loading');
  session().then(() => announce('ready')).catch((err) => announce('failed', err.message));
})();