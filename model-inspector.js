(() => {
  'use strict';

  const CORE = window.SIXMOK_MODEL_CORE;
  const MODEL_URL = 'sixmok.onnx?v=6.2-policy20-opening35-t1-inspector';
  const MODEL_BOARD_SIZE = 19;
  const DEFAULT_POLICY_CUTOFF_REL = 0.20;
  const OPENING_POLICY_CUTOFF_REL = 0.35;
  const OPENING_TEMPERATURE = 1.0;

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
    if (!sessionPromise) {
      sessionPromise = ort.InferenceSession.create(MODEL_URL, { executionProviders: ['wasm'] });
    }
    return sessionPromise;
  }

  async function evaluate(games) {
    if (!games.length) return { policy: new Float32Array(), value: new Float32Array(), stride: 0 };
    const n = games[0].size;
    const sess = await session();

    async function runWithPlanes(planes) {
      const planeSize = planes * n * n;
      const data = new Float32Array(games.length * planeSize);
      games.forEach((g, k) => data.set(CORE.encodeStateForModel(g, planes), k * planeSize));
      const tensor = new ort.Tensor('float32', data, [games.length, planes, n, n]);
      return sess.run({ board: tensor });
    }

    let out;
    if (cachedInputPlanes !== null) {
      out = await runWithPlanes(cachedInputPlanes);
    } else {
      let err13;
      try {
        out = await runWithPlanes(13);
        cachedInputPlanes = 13;
      } catch (err) {
        err13 = err;
        try {
          out = await runWithPlanes(16);
          cachedInputPlanes = 16;
        } catch (err16) {
          throw new Error(`ONNX 입력 규격 확인 실패 (13-plane: ${err13.message}; 16-plane: ${err16.message})`);
        }
      }
    }
    return { policy: out.policy.data, value: out.value.data, stride: n * n + 1 };
  }

  function stateKey(game) {
    return [
      game.current,
      game.placedThisTurn,
      game.stones[BLACK],
      game.stones[WHITE],
      game.stonesOnBoard,
      Array.from(game.board).join(''),
    ].join('|');
  }

  function rootPerspectiveTerminal(child, rootColor) {
    if (!child.isOver) return null;
    if (child.winner === null) return 0;
    return child.winner === rootColor ? 1 : -1;
  }

  function levelDefs(game) {
    const out = {};
    for (const key of ['easy', 'normal', 'hard']) {
      const src = (typeof AI_LEVELS !== 'undefined' && AI_LEVELS[key]) ? AI_LEVELS[key] : {};
      out[key] = {
        label: src.label || key,
        baseTemperature: Number.isFinite(src.temperature) ? src.temperature : 0,
        effectiveTemperature: game.stonesOnBoard === 0
          ? OPENING_TEMPERATURE
          : CORE.effectiveTemperature(game, src.temperature),
        cutoffRel: Number.isFinite(src.cutoffRel) ? src.cutoffRel : 0,
      };
    }
    return out;
  }

  function normalizedTemperatureWeights(rows, eligibleSet, temperature) {
    const result = new Map();
    if (!eligibleSet.size) return result;
    if (!(temperature > 0)) {
      let best = null;
      for (const row of rows) {
        if (!eligibleSet.has(row.action)) continue;
        if (!best || row.selectionWeight > best.selectionWeight ||
            (row.selectionWeight === best.selectionWeight && row.action < best.action)) best = row;
      }
      if (best) result.set(best.action, 1);
      return result;
    }
    let total = 0;
    for (const row of rows) {
      if (!eligibleSet.has(row.action)) continue;
      const scaled = Math.pow(Math.max(row.selectionWeight, 1e-12), 1 / temperature);
      result.set(row.action, scaled);
      total += scaled;
    }
    if (!(total > 0) || !Number.isFinite(total)) return result;
    for (const [action, weight] of result) result.set(action, weight / total);
    return result;
  }

  function cutoffSet(game, rows, actualCandidateSet, cutoffRel) {
    const candidateRows = rows.filter((row) => actualCandidateSet.has(row.action));
    if (game.stonesOnBoard === 0 || candidateRows.length <= 1) return new Set(candidateRows.map((row) => row.action));
    const winning = candidateRows.filter((row) => row.childValue >= 1 - 1e-9);
    if (winning.length) return new Set(winning.map((row) => row.action));
    let best = 0;
    for (const row of candidateRows) best = Math.max(best, row.selectionWeight);
    if (!(best > 0)) return new Set(candidateRows.map((row) => row.action));
    const threshold = best * Math.max(0, Math.min(1, cutoffRel));
    const kept = candidateRows.filter((row) => row.selectionWeight >= threshold);
    return new Set((kept.length ? kept : candidateRows).map((row) => row.action));
  }

  function classifyAction(game, action, data) {
    const endIndex = game.size * game.size;
    if (action === endIndex) return 'END_TURN';
    if (data.immediateWin === action) return '즉시승 강제수';
    if (data.ownWinning.has(action)) return '한 턴 승리 후보';
    if (data.defenses.has(action)) return '필수 방어';
    if (data.foeWinning.has(action)) return '상대 승리점';
    if (game.stonesOnBoard === 0) return data.openingCandidates.has(action) ? '오프닝 35% 통과' : '오프닝 컷 제외';
    return data.actualCandidates.has(action) ? 'policy 20% 통과' : 'policy 컷 제외';
  }

  async function analyzeUncached(game) {
    if (game.size !== MODEL_BOARD_SIZE) throw new Error(`모델은 ${MODEL_BOARD_SIZE}줄 판만 지원합니다.`);
    const rootColor = game.current;
    const n = game.size;
    const endIndex = n * n;
    const rootEval = await evaluate([game]);
    const rootLogits = rootEval.policy;
    const probs = CORE.legalProbs(rootLogits, game, 0, rootEval.stride);
    const rootValue = Number(rootEval.value[0]);

    const legalActions = [];
    if (game.canPlace()) {
      for (let i = 0; i < endIndex; i++) if (game.board[i] === EMPTY) legalActions.push(i);
    }
    legalActions.push(endIndex);

    const children = legalActions.map((action) => CORE.playEvaluationAction(game, action));
    const childValues = new Array(children.length);
    const pending = [];
    const pendingAt = [];
    children.forEach((child, idx) => {
      const terminal = rootPerspectiveTerminal(child, rootColor);
      if (terminal !== null) childValues[idx] = terminal;
      else { pending.push(child); pendingAt.push(idx); }
    });
    if (pending.length) {
      const ev = await evaluate(pending);
      pendingAt.forEach((idx, j) => {
        const child = children[idx];
        const v = Number(ev.value[j]);
        childValues[idx] = child.current === rootColor ? v : -v;
      });
    }

    let bestPlacementPolicy = 0;
    for (let i = 0; i < endIndex; i++) {
      if (game.board[i] === EMPTY) bestPlacementPolicy = Math.max(bestPlacementPolicy, Number(probs[i]) || 0);
    }
    const openingThreshold = bestPlacementPolicy * OPENING_POLICY_CUTOFF_REL;
    const openingCandidates = new Set([endIndex]);
    if (game.canPlace()) {
      for (let i = 0; i < endIndex; i++) {
        if (game.board[i] === EMPTY && (Number(probs[i]) || 0) >= openingThreshold) openingCandidates.add(i);
      }
    }

    const actualCandidateArray = game.stonesOnBoard === 0
      ? [...openingCandidates]
      : CORE.candidateActions(game, probs, DEFAULT_POLICY_CUTOFF_REL);
    const actualCandidates = new Set(actualCandidateArray);
    const ownWinning = new Set(CORE.oneTurnWinningSquares(game, game.current));
    const foeColor = game.current === BLACK ? WHITE : BLACK;
    const foeWinning = new Set(CORE.oneTurnWinningSquares(game, foeColor));
    const defenses = new Set(foeWinning.size ? CORE.oneTurnDefenseSquares(game, foeColor) : []);
    const immediateWin = CORE.findImmediateWin(game);

    const rows = legalActions.map((action, idx) => {
      const policy = Number(probs[action]) || 0;
      const childValue = Number(childValues[idx]);
      const winProb = CORE.valueWinProb(childValue);
      const pv = CORE.policyValueWeight(policy, childValue);
      const modifier = CORE.actionModifier(game, action);
      const selectionWeight = game.stonesOnBoard === 0 ? policy * modifier : pv * modifier;
      return {
        action,
        isEnd: action === endIndex,
        row: action === endIndex ? null : Math.floor(action / n),
        col: action === endIndex ? null : action % n,
        rawLogit: Number(rootLogits[action]),
        policy,
        childValue,
        winProb,
        pv,
        modifier,
        selectionWeight,
        actualCandidate: actualCandidates.has(action),
      };
    });

    const byPolicy = [...rows].sort((a, b) => (b.policy - a.policy) || (a.action - b.action));
    byPolicy.forEach((row, i) => { row.policyRank = i + 1; });
    const byScore = [...rows].sort((a, b) => (b.selectionWeight - a.selectionWeight) || (a.action - b.action));
    byScore.forEach((row, i) => { row.greedyRankAll = i + 1; });

    let greedyRow = null;
    let decisionPath = game.stonesOnBoard === 0 ? '오프닝 root policy' : '1-ply P×V';
    if (!game.canPlace()) {
      greedyRow = rows.find((row) => row.action === endIndex) || null;
      decisionPath = '착수 불가 → END_TURN';
    } else if (immediateWin >= 0) {
      greedyRow = rows.find((row) => row.action === immediateWin) || null;
      decisionPath = '즉시승 강제수';
    } else {
      const pool = rows.filter((row) => actualCandidates.has(row.action));
      greedyRow = [...pool].sort((a, b) => (b.selectionWeight - a.selectionWeight) || (a.action - b.action))[0] || null;
    }

    const levels = levelDefs(game);
    const levelSets = {};
    const levelProbMaps = {};
    for (const key of Object.keys(levels)) {
      const eligible = cutoffSet(game, rows, actualCandidates, levels[key].cutoffRel);
      levelSets[key] = eligible;
      levelProbMaps[key] = normalizedTemperatureWeights(rows, eligible, levels[key].effectiveTemperature);
      levels[key].eligibleCount = eligible.size;
    }

    const classData = { immediateWin, ownWinning, foeWinning, defenses, openingCandidates, actualCandidates };
    for (const row of rows) {
      row.reason = classifyAction(game, row.action, classData);
      row.greedy = !!greedyRow && row.action === greedyRow.action;
      row.levels = {};
      for (const key of Object.keys(levels)) {
        row.levels[key] = {
          eligible: levelSets[key].has(row.action),
          probability: levelProbMaps[key].get(row.action) || 0,
        };
      }
    }

    const rootPolicyEntropy = rows.reduce((sum, row) => row.policy > 0 ? sum - row.policy * Math.log(row.policy) : sum, 0);
    const result = {
      modelUrl: MODEL_URL,
      inputPlanes: cachedInputPlanes,
      rootColor,
      rootValue,
      rootWinProb: CORE.valueWinProb(rootValue),
      rootPolicyEntropy,
      legalCount: rows.length,
      actualCandidateCount: actualCandidates.size,
      decisionPath,
      policyCutoffRel: game.stonesOnBoard === 0 ? OPENING_POLICY_CUTOFF_REL : DEFAULT_POLICY_CUTOFF_REL,
      greedyAction: greedyRow ? greedyRow.action : endIndex,
      greedyMove: greedyRow && !greedyRow.isEnd ? [greedyRow.row, greedyRow.col] : END_TURN,
      levels,
      rows: byScore,
    };
    return result;
  }

  async function analyze(game) {
    const key = stateKey(game);
    if (key === lastKey && lastPromise) return lastPromise;
    lastKey = key;
    lastPromise = analyzeUncached(game).then((result) => {
      window.dispatchEvent(new CustomEvent('model-analysis', { detail: result }));
      return result;
    }).catch((err) => {
      if (lastKey === key) { lastKey = null; lastPromise = null; }
      throw err;
    });
    return lastPromise;
  }

  window.MODEL_ANALYZE_POSITION = analyze;
  window.MODEL_PLAN_TURN = async (game) => {
    const result = await analyze(game.clone ? game.clone() : game);
    return result.greedyMove === END_TURN ? [] : [result.greedyMove];
  };

  announce('loading');
  session()
    .then(() => announce('ready'))
    .catch((err) => announce('failed', err.message));
})();
