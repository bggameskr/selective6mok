/* 학습한 모델을 브라우저 게임에 붙인다.

   난이도에 따라 모델 정책과 가치만으로 제한 탐색한다.
     쉬움   8갈래 — 정책 상위 후보의 1-ply value×policy 가중치에서 온도로 샘플링
     보통   3갈래 — 정책 상위 후보의 1-ply value×policy 가중치에서 온도로 샘플링
     어려움 6갈래 — 상위 후보에서 현재 턴 전체와 상대 턴 전체를 제한 탐색한 뒤 value×policy로 선택
   모든 난이도는 최종 value×policy 가중치에 온도를 적용하고, 각 색의 첫 착수 턴에는 더 높은 온도를 쓴다.
   어려움은 한 번 고른 현재 턴의 수열을 그대로 실행해 같은 턴 안에서 재탐색하지 않는다.

   policy는 19x19 착수 361개와 턴 마치기 1개, 총 362개 행동을 하나의 분포로 다룬다.
   턴 마치기도 다른 합법 행동과 동일하게 softmax, 후보 선정, 온도 샘플링에 참여한다.
   value는 각 후보를 실제 적용한 후의 후속 국면을 평가하며, 현재 플레이어 관점으로 되돌린 뒤
   [-1, 1]을 [0, 1] 승산으로 변환해 policy 확률과 곱한다.

   즉시승 강제 선택, 강제방어 후보 삽입, 삼삼/사사 후보 보정, 전술 leaf 값 덮어쓰기를
   사용하지 않는다. 실제로 탐색한 수가 게임 종료 상태에 도달한 경우의 승패만 확정값으로 쓴다.
   오프닝도 별도로 변형하거나 대칭 이동하지 않는다.

   입력을 만드는 encodeState 는 engine.js 안에 있다. 학습에 쓴 encoding.py 와
   규격이 같아야 하며, parity_check 로 확인할 수 있다. */

(() => {
  'use strict';

  const MODEL_URL = 'sixmok.onnx';
  const MODEL_BOARD_SIZE = 19;
  const POLICY_TIE_REL_EPS = 1e-5;
  const EASY_CANDIDATE_WIDTH = 8;
  const DEEP_ROOTS = 3;
  const DEEP_WIDTH = 3;
  const TURN_NODE_CAP = 64;
  const NORMAL_OPENING_TEMPERATURE = 1.0;
  const HARD_TEMPERATURE = 0.2;
  const HARD_OPENING_TEMPERATURE = 0.8;
  const VALUE_WEIGHT_FLOOR = 1e-6;

  let sessionPromise = null;

  function announce(status, detail) {
    window.MODEL_STATUS = status;
    window.MODEL_STATUS_DETAIL = detail || '';
    window.dispatchEvent(new CustomEvent('model-status'));
  }

  announce('loading');

  if (typeof ort === 'undefined') {
    announce('failed', 'onnxruntime-web 을 불러오지 못했습니다. 인터넷 연결이나 CDN 차단을 확인하세요.');
    console.warn('onnxruntime-web 이 없습니다. 규칙 기반 AI로 둡니다.');
    return;
  }

  function session() {
    if (!sessionPromise) {
      sessionPromise = ort.InferenceSession.create(MODEL_URL, {
        executionProviders: ['wasm'],
      });
    }
    return sessionPromise;
  }

  async function evaluate(games) {
    const n = games[0].size;
    const planeSize = N_PLANES * n * n;
    const data = new Float32Array(games.length * planeSize);
    games.forEach((g, k) => data.set(encodeState(g), k * planeSize));

    const sess = await session();
    const tensor = new ort.Tensor('float32', data, [games.length, N_PLANES, n, n]);
    const out = await sess.run({ board: tensor });
    return {
      policy: out.policy.data,
      value: out.value.data,
      stride: n * n + 1,
    };
  }

  function legalProbs(logits, game, offset, stride) {
    const n = game.size;
    const endIndex = n * n;
    const probs = new Float64Array(stride);

    if (!game.canPlace()) {
      probs[endIndex] = 1;
      return probs;
    }

    const legal = [endIndex];
    for (let i = 0; i < endIndex; i++) {
      if (game.board[i] === 0) legal.push(i);
    }

    let max = -Infinity;
    for (const action of legal) {
      const raw = logits[offset + action];
      const v = Number.isFinite(raw) ? raw : -1e9;
      if (v > max) max = v;
    }

    let total = 0;
    for (const action of legal) {
      const raw = logits[offset + action];
      const v = Number.isFinite(raw) ? raw : -1e9;
      const e = Math.exp(v - max);
      probs[action] = e;
      total += e;
    }

    if (!(total > 0) || !Number.isFinite(total)) {
      const uniform = 1 / legal.length;
      for (const action of legal) probs[action] = uniform;
      return probs;
    }

    for (const action of legal) probs[action] /= total;
    return probs;
  }

  function toMove(index, n) {
    return index === n * n ? END_TURN : [Math.floor(index / n), index % n];
  }

  function otherColor(color) {
    return color === BLACK ? WHITE : BLACK;
  }

  function rootValue(game, rootColor) {
    if (!game.isOver) return null;
    if (game.winner === null) return 0;
    return game.winner === rootColor ? 1 : -1;
  }

  function hasColorStone(game, color) {
    for (let i = 0; i < game.board.length; i++) {
      if (game.board[i] === color) return true;
    }
    return false;
  }

  function rankActions(actions, probs, rng = Math.random) {
    const ranked = actions.slice().sort((a, b) => probs[b] - probs[a]);
    if (ranked.length < 2) return ranked;

    const maxProb = Math.max(...ranked.map((action) => Math.abs(probs[action])));
    const tolerance = Math.max(1e-12, maxProb * POLICY_TIE_REL_EPS);

    let start = 0;
    while (start < ranked.length) {
      const anchor = probs[ranked[start]];
      let end = start + 1;
      while (end < ranked.length && anchor - probs[ranked[end]] <= tolerance) end += 1;

      for (let i = end - 1; i > start; i--) {
        const j = start + Math.floor(rng() * (i - start + 1));
        [ranked[i], ranked[j]] = [ranked[j], ranked[i]];
      }
      start = end;
    }
    return ranked;
  }

  function candidateActions(game, probs, width, rng = Math.random) {
    const n = game.size;
    const endIndex = n * n;
    if (!game.canPlace()) return [endIndex];

    const legal = [endIndex];
    for (let i = 0; i < endIndex; i++) {
      if (game.board[i] === 0) legal.push(i);
    }

    const ranked = rankActions(legal, probs, rng);
    return ranked.slice(0, Math.max(1, width));
  }

  function sampleActions(probs, actions, temperature, rng = Math.random) {
    if (!actions.length) return -1;

    if (!(temperature > 0)) {
      let best = actions[0];
      for (const action of actions.slice(1)) {
        if (probs[action] > probs[best]) best = action;
      }
      return best;
    }

    const weights = actions.map((action) => Math.pow(Math.max(probs[action], 1e-12), 1 / temperature));
    const total = weights.reduce((sum, value) => sum + value, 0);
    let roll = rng() * total;

    for (let i = 0; i < actions.length; i++) {
      roll -= weights[i];
      if (roll <= 0) return actions[i];
    }
    return actions[actions.length - 1];
  }

  function sampleScoredIndex(indices, scoreFn, temperature, rng = Math.random) {
    if (!indices.length) return -1;
    if (!(temperature > 0)) {
      let best = indices[0];
      for (const k of indices.slice(1)) {
        if (scoreFn(k) > scoreFn(best)) best = k;
      }
      return best;
    }

    let maxScore = -Infinity;
    for (const k of indices) maxScore = Math.max(maxScore, scoreFn(k));
    const weights = indices.map((k) => Math.exp((scoreFn(k) - maxScore) / temperature));
    const total = weights.reduce((sum, value) => sum + value, 0);
    let roll = rng() * total;
    for (let i = 0; i < indices.length; i++) {
      roll -= weights[i];
      if (roll <= 0) return indices[i];
    }
    return indices[indices.length - 1];
  }

  function valueWinProb(value) {
    const raw = Number.isFinite(value) ? value : -1;
    const clipped = Math.max(-1, Math.min(1, raw));
    return Math.max(VALUE_WEIGHT_FLOOR, Math.min(1, (clipped + 1) * 0.5));
  }

  function policyValueWeight(policyProb, rootValueScore) {
    return Math.max(Number(policyProb) || 0, 0) * valueWinProb(rootValueScore);
  }

  function sampleWeightedIndex(indices, weightFn, temperature, rng = Math.random) {
    if (!indices.length) return -1;

    if (!(temperature > 0)) {
      let best = indices[0];
      for (const k of indices.slice(1)) {
        if (weightFn(k) > weightFn(best)) best = k;
      }
      return best;
    }

    const weights = indices.map((k) => (
      Math.pow(Math.max(weightFn(k), Number.MIN_VALUE), 1 / temperature)
    ));
    const total = weights.reduce((sum, value) => sum + value, 0);
    if (!(total > 0) || !Number.isFinite(total)) {
      let best = indices[0];
      for (const k of indices.slice(1)) {
        if (weightFn(k) > weightFn(best)) best = k;
      }
      return best;
    }

    let roll = rng() * total;
    for (let i = 0; i < indices.length; i++) {
      roll -= weights[i];
      if (roll <= 0) return indices[i];
    }
    return indices[indices.length - 1];
  }

  async function expandWholeTurn(nodes, turnColor, evaluateFn, width, recordTurnActions = false) {
    let frontier = nodes.slice();
    const done = [];

    for (let depth = 0; depth < 4 && frontier.length; depth++) {
      const active = [];

      for (const node of frontier) {
        const g = node.game;
        if (g.isOver || g.current !== turnColor) {
          done.push(node);
          continue;
        }
        if (!g.canPlace()) {
          const child = g.clone();
          child.play(END_TURN);
          done.push({ ...node, game: child });
          continue;
        }
        active.push(node);
      }

      if (!active.length) {
        frontier = [];
        break;
      }

      const { policy, stride } = await evaluateFn(active.map((node) => node.game));
      const next = [];
      active.forEach((node, j) => {
        const g = node.game;
        const probs = legalProbs(policy, g, j * stride, stride);
        const actions = candidateActions(g, probs, width);
        const endIndex = g.size * g.size;

        for (const action of actions) {
          const child = g.clone();
          child.play(toMove(action, g.size));
          const prior = node.prior * Math.max(probs[action], 1e-12);
          const nextNode = { ...node, game: child, prior };
          if (recordTurnActions) {
            nextNode.turnActions = (node.turnActions || []).slice();
            if (action !== endIndex) nextNode.turnActions.push(action);
          }
          next.push(nextNode);
        }
      });

      next.sort((a, b) => b.prior - a.prior);
      frontier = next.slice(0, TURN_NODE_CAP);
    }

    for (const node of frontier) {
      const g = node.game.clone();
      if (!g.isOver && g.current === turnColor) g.play(END_TURN);
      done.push({ ...node, game: g });
    }
    return done;
  }

  async function evaluateLeaves(nodes, rootColor, evaluateFn) {
    const scores = new Array(nodes.length);
    const pending = [];
    const pendingAt = [];

    nodes.forEach((node, i) => {
      const terminal = rootValue(node.game, rootColor);
      if (terminal !== null) scores[i] = terminal;
      else {
        pending.push(node.game);
        pendingAt.push(i);
      }
    });

    if (pending.length) {
      const { value } = await evaluateFn(pending);
      pendingAt.forEach((i, j) => {
        const g = nodes[i].game;
        scores[i] = g.current === rootColor ? value[j] : -value[j];
      });
    }
    return scores;
  }

  async function pickWithLookahead(
    game, probs, width, evaluateFn, deep = false, returnPlan = false,
    temperature = 0, rng = Math.random
  ) {
    const n = game.size;
    const rootColor = game.current;
    const actions = candidateActions(game, probs, width, rng);

    const children = actions.map((action) => {
      const child = game.clone();
      child.play(toMove(action, n));
      return child;
    });

    const scores = new Array(actions.length);
    const pending = [];
    const pendingAt = [];
    children.forEach((child, k) => {
      const terminal = rootValue(child, rootColor);
      if (terminal !== null) scores[k] = terminal;
      else {
        pending.push(child);
        pendingAt.push(k);
      }
    });

    if (pending.length) {
      const { value } = await evaluateFn(pending);
      pendingAt.forEach((k, j) => {
        const child = children[k];
        scores[k] = child.current === rootColor ? value[j] : -value[j];
      });
    }

    let deepPlans = null;
    if (deep) {
      const expandable = actions
        .map((action, k) => ({
          action,
          k,
          weight: policyValueWeight(probs[action], scores[k]),
        }))
        .filter(({ k }) => !children[k].isOver)
        .sort((a, b) => b.weight - a.weight)
        .slice(0, DEEP_ROOTS);

      const endIndex = n * n;
      const rootStarts = expandable.map(({ action, k }) => ({
        game: children[k],
        rootIndex: k,
        prior: Math.max(probs[action], 1e-12),
        turnActions: action === endIndex ? [] : [action],
      }));

      const alreadyEnded = [];
      const stillRootTurn = [];
      for (const node of rootStarts) {
        if (node.game.current === rootColor) stillRootTurn.push(node);
        else alreadyEnded.push(node);
      }

      const rootTurnEnds = alreadyEnded.concat(
        stillRootTurn.length
          ? await expandWholeTurn(stillRootTurn, rootColor, evaluateFn, DEEP_WIDTH, true)
          : []
      );
      rootTurnEnds.forEach((node, seqId) => { node.seqId = seqId; });

      const terminalRootEnds = [];
      const opponentStarts = [];
      for (const node of rootTurnEnds) {
        if (node.game.isOver) terminalRootEnds.push(node);
        else opponentStarts.push(node);
      }

      const opponentColor = otherColor(rootColor);
      const opponentTurnEnds = opponentStarts.length
        ? await expandWholeTurn(opponentStarts, opponentColor, evaluateFn, DEEP_WIDTH, false)
        : [];

      const allLeaves = terminalRootEnds.concat(opponentTurnEnds);
      const leafScores = await evaluateLeaves(allLeaves, rootColor, evaluateFn);
      const seqWorst = new Map();
      allLeaves.forEach((node, i) => {
        const prev = seqWorst.get(node.seqId);
        const score = leafScores[i];
        if (prev === undefined || score < prev) seqWorst.set(node.seqId, score);
      });

      const rootBest = new Map();
      for (const node of rootTurnEnds) {
        const seqScore = seqWorst.get(node.seqId);
        if (seqScore === undefined) continue;
        const prev = rootBest.get(node.rootIndex);
        if (prev === undefined || seqScore > prev.score) {
          rootBest.set(node.rootIndex, {
            score: seqScore,
            turnActions: (node.turnActions || []).slice(),
          });
        }
      }

      if (rootBest.size) {
        deepPlans = rootBest;
        for (const [k, result] of rootBest) scores[k] = result.score;
      }
    }

    const indices = deepPlans && deepPlans.size ? [...deepPlans.keys()] : actions.map((_, k) => k);
    const finalWeight = (k) => policyValueWeight(probs[actions[k]], scores[k]);
    const terminalWins = indices.filter((k) => rootValue(children[k], rootColor) === 1);
    const rankedIndices = terminalWins.length ? terminalWins : indices;
    const best = sampleWeightedIndex(rankedIndices, finalWeight, temperature, rng);
    const endIndex = n * n;

    if (returnPlan) {
      if (deepPlans && deepPlans.has(best)) {
        return {
          action: actions[best],
          turnActions: deepPlans.get(best).turnActions,
        };
      }
      return {
        action: actions[best],
        turnActions: actions[best] === endIndex ? [] : [actions[best]],
      };
    }
    return actions[best];
  }

  async function planTurnWithModel(game, level, evaluateFn, options = {}) {
    const n = game.size;
    const endIndex = n * n;
    const configuredTemperature = level.temperature || 0;
    const width = level.lookahead || 0;
    const deep = width >= 6;
    const openingTurn = game.placedThisTurn === 0 && !hasColorStone(game, game.current);
    const rng = options.rng || Math.random;

    let temperature = configuredTemperature;
    if (deep) {
      temperature = openingTurn
        ? HARD_OPENING_TEMPERATURE
        : Math.max(configuredTemperature, HARD_TEMPERATURE);
    } else if (width > 0 && openingTurn) {
      temperature = Math.max(configuredTemperature, NORMAL_OPENING_TEMPERATURE);
    }

    const sim = game.clone();
    const moves = [];

    if (deep) {
      const { policy, stride } = await evaluateFn([sim]);
      const probs = legalProbs(policy, sim, 0, stride);
      const plan = await pickWithLookahead(
        sim, probs, width, evaluateFn, true, true, temperature, rng
      );

      for (const index of plan.turnActions) {
        if (index === endIndex || !sim.canPlace()) break;
        const r = Math.floor(index / n);
        const c = index % n;
        sim.place(r, c);
        moves.push([r, c]);
        if (sim.isOver) break;
      }
      return moves;
    }

    while (sim.canPlace()) {
      const { policy, stride } = await evaluateFn([sim]);
      const probs = legalProbs(policy, sim, 0, stride);

      const candidateWidth = width > 0 ? width : EASY_CANDIDATE_WIDTH;
      const index = await pickWithLookahead(
        sim, probs, candidateWidth, evaluateFn, false, false, temperature, rng
      );

      if (index === endIndex || index < 0) break;
      const r = Math.floor(index / n);
      const c = index % n;
      sim.place(r, c);
      moves.push([r, c]);
      if (sim.isOver) break;
    }
    return moves;
  }

  function actionIndex(move, n) {
    return move === END_TURN ? n * n : move[0] * n + move[1];
  }

  function policyRank(probs, game, selected) {
    const endIndex = game.size * game.size;
    const legal = [endIndex];
    if (game.canPlace()) {
      for (let i = 0; i < endIndex; i++) {
        if (game.board[i] === EMPTY) legal.push(i);
      }
    }
    legal.sort((a, b) => probs[b] - probs[a]);
    const rank = legal.indexOf(selected);
    return rank < 0 ? null : rank + 1;
  }

  function moveJson(move, n) {
    if (move === END_TURN) return { type: 'end_turn', index: n * n };
    return {
      type: 'place',
      row: move[0],
      col: move[1],
      row1: move[0] + 1,
      col1: move[1] + 1,
      index: move[0] * n + move[1],
    };
  }

  async function buildKifu(game) {
    if (!game) throw new Error('진행 중인 대국이 없습니다.');
    if (game.size !== MODEL_BOARD_SIZE) {
      throw new Error(`모델 가중치는 ${MODEL_BOARD_SIZE}줄 판에서만 저장할 수 있습니다.`);
    }

    const history = game.history.map(([color, move]) => [
      color,
      Array.isArray(move) ? [move[0], move[1]] : move,
    ]);
    const replay = new SixMok({ ...game.config });
    const states = [];
    let turnIndex = 1;

    for (const [color, move] of history) {
      states.push({
        game: replay.clone(),
        color,
        move,
        turnIndex,
      });
      replay.play(move);
      if (move === END_TURN) turnIndex += 1;
    }

    let policy = new Float32Array(0);
    let value = new Float32Array(0);
    let stride = game.size * game.size + 1;
    if (states.length) {
      const out = await evaluate(states.map((entry) => entry.game));
      policy = out.policy;
      value = out.value;
      stride = out.stride;
    }

    const actions = states.map((entry, i) => {
      const state = entry.game;
      const offset = i * stride;
      const probs = legalProbs(policy, state, offset, stride);
      const selected = actionIndex(entry.move, state.size);
      const endIndex = state.size * state.size;
      let best = endIndex;
      for (let k = 0; k < stride; k++) {
        if (probs[k] > probs[best]) best = k;
      }

      return {
        ply: i + 1,
        turn: entry.turnIndex,
        color: state.current,
        history_color: entry.color,
        action_in_turn: entry.move === END_TURN ? 'end' : state.placedThisTurn + 1,
        move: moveJson(entry.move, state.size),
        stones_before: {
          black: state.stones[BLACK],
          white: state.stones[WHITE],
        },
        placed_this_turn_before: state.placedThisTurn,
        value: Number(value[i]),
        selected_policy_logit: Number(policy[offset + selected]),
        selected_policy_prob: Number(probs[selected]),
        selected_policy_rank: policyRank(probs, state, selected),
        best_policy_action: moveJson(toMove(best, state.size), state.size),
        best_policy_prob: Number(probs[best]),
        end_turn_logit: Number(policy[offset + endIndex]),
        end_turn_prob: Number(probs[endIndex]),
        policy_logits: Array.from(policy.slice(offset, offset + stride), Number),
        policy_probs: Array.from(probs.slice(0, stride), Number),
      };
    });

    const level = typeof window.AI_LEVEL === 'function' ? window.AI_LEVEL() : null;
    const side = document.getElementById('side');
    return {
      format: 'selective6mok-web-kifu-v1',
      created_at: new Date().toISOString(),
      model: MODEL_URL,
      note: 'policy 값은 361개 착수와 턴 마치기를 합친 362개 신경망 원출력과 합법수 기준 softmax 확률입니다. 실제 착수는 후보별 후속 value를 현재 플레이어 승산으로 바꾼 뒤 policy 확률과 곱한 최종 가중치에 temperature를 적용합니다.',
      board_size: game.size,
      config: { ...game.config },
      difficulty: level ? {
        label: level.label || null,
        lookahead: level.lookahead || 0,
        temperature: level.temperature || 0,
      } : null,
      human_color: side ? Number(side.value) : null,
      actions,
      result: {
        winner: game.winner,
        is_draw: game.isDraw,
        is_over: game.isOver,
        stones_on_board: game.stonesOnBoard,
      },
    };
  }

  async function downloadKifu() {
    const game = typeof window.__game === 'function' ? window.__game() : null;
    const kifu = await buildKifu(game);
    const blob = new Blob([JSON.stringify(kifu)], { type: 'application/json;charset=utf-8' });
    const stamp = new Date().toISOString().replace(/[-:]/g, '').replace(/\.\d{3}Z$/, 'Z');
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `sixmok-kifu-${stamp}.json`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 0);
  }

  function installKifuButton() {
    if (document.getElementById('kifu-download')) return;
    const actions = document.querySelector('.rail .actions');
    if (!actions) return;

    const button = document.createElement('button');
    button.id = 'kifu-download';
    button.type = 'button';
    button.textContent = '기보 다운로드';
    button.title = '현재 대국과 각 행동 직전의 모델 policy/value를 JSON으로 저장합니다.';
    button.style.width = '100%';
    actions.insertAdjacentElement('afterend', button);

    button.addEventListener('click', async () => {
      const previous = button.textContent;
      button.disabled = true;
      button.textContent = '기보 만드는 중…';
      try {
        await downloadKifu();
      } catch (err) {
        console.error('기보 다운로드 실패', err);
        window.alert(`기보를 만들지 못했습니다: ${err.message}`);
      } finally {
        button.disabled = false;
        button.textContent = previous;
      }
    });
  }

  window.MODEL_BUILD_KIFU = buildKifu;
  window.MODEL_DOWNLOAD_KIFU = downloadKifu;

  window.MODEL_PLAN_TURN = async (game) => {
    if (game.size !== MODEL_BOARD_SIZE) {
      const msg = `모델은 ${MODEL_BOARD_SIZE}줄 판으로만 학습돼 있습니다.`;
      announce('failed', msg);
      throw new Error(msg);
    }

    const level = (typeof window.AI_LEVEL === 'function' ? window.AI_LEVEL() : null) || {};
    return planTurnWithModel(game, level, evaluate);
  };

  installKifuButton();

  session()
    .then(() => {
      announce('ready');
      console.log(`모델을 불러왔습니다: ${MODEL_URL}`);
    })
    .catch((err) => {
      announce('failed', err.message);
      console.warn('모델을 불러오지 못했습니다. 규칙 기반 AI로 둡니다.', err);
    });

  if (typeof module !== 'undefined') {
    module.exports = {
      planTurnWithModel,
      pickWithLookahead,
      candidateActions,
      legalProbs,
      rankActions,
      rankBoardActions: rankActions,
      sampleActions,
      sampleScoredIndex,
      sampleWeightedIndex,
      valueWinProb,
      policyValueWeight,
      expandWholeTurn,
      evaluateLeaves,
      hasColorStone,
      buildKifu,
    };
  }
})();
