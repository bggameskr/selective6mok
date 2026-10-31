/* 학습한 모델을 브라우저 게임에 붙인다.

   모든 난이도는 강제승/필수방어를 보존한 1-ply policy×value 평가를 공통으로 쓴다.
   일반 상황에서는 root policy 최고값의 20% 이상인 착점을 후보로 삼고,
   살아남은 후보만 child value를 평가한다. 후보 개수 자체에는 상한을 두지 않는다.
   END_TURN은 policy 컷오프와 무관하게 항상 후보로 유지한다.
   빈 판은 child 탐색 없이 root policy만 사용하며, 착점 20% 컷과 END_TURN을 함께 T=1.0으로 선택한다.
   난이도 차이는 P×V 가중치 컷오프와 temperature로 조절한다.
   전체 착수 10개 미만은 기본 T×1.5를 적용한다.
   빈 판을 제외하면 난이도별 기준보다 낮은 P×V 가중치 후보는 샘플링에서 제외한다.

   13평면(v5.10+)과 구형 16평면 ONNX를 모두 지원한다.
   v6.2 배포 정합성:
     - 3번째 착수/보유돌 소진 child는 강제 END_TURN 뒤 상태로 value를 평가한다.
     - 0착수 자발 END_TURN만 보유돌 LUT로 감쇠한다. */

(() => {
  'use strict';

  const MODEL_URL = 'sixmok.onnx?v=6.2-policy20-opening-end-t1';
  const MODEL_BOARD_SIZE = 19;
  const VALUE_WEIGHT_FLOOR = 1e-6;
  const DEFAULT_WEIGHT_CUTOFF_REL = 0.05;
  const DEFAULT_POLICY_CUTOFF_REL = 0.20;
  const OPENING_TEMPERATURE = 1.0;
  const EARLY_ACTIONS = 10;
  const EARLY_TEMPERATURE_MULTIPLIER = 1.5;
  const STONE_SCALE = 6;
  const STONE_LINEAR_LIMIT = 12;
  const STONE_TAIL_SCALE = 48;
  const END_STOCK_LUT_MAX = 32;
  const POLICY_NEAR_TIE_REL_TOL = 1e-4;
  const EMPTY_END_STOCK_LUT = Array.from(
    { length: END_STOCK_LUT_MAX + 1 },
    (_, stones) => (stones <= 2 ? 1.0 : Math.pow(0.5, stones - 2)),
  );

  let sessionPromise = null;
  let cachedInputPlanes = null;

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
      sessionPromise = ort.InferenceSession.create(MODEL_URL, { executionProviders: ['wasm'] });
    }
    return sessionPromise;
  }

  function encodeStoneCount(stones) {
    if (stones <= STONE_LINEAR_LIMIT) return stones / STONE_SCALE;
    return 2 + (stones - STONE_LINEAR_LIMIT) / STONE_TAIL_SCALE;
  }

  function encodeStateForModel(game, planes) {
    const n = game.size;
    const plane = n * n;
    const me = game.current;
    const foe = otherColor(me);
    const need = game.config.winLength;
    const data = new Float32Array(planes * plane);

    for (let i = 0; i < plane; i++) {
      const v = game.board[i];
      if (v === me) data[i] = 1;
      else if (v === foe) data[plane + i] = 1;
    }

    if (planes === 13) {
      data.fill(encodeStoneCount(game.stones[me]), 2 * plane, 3 * plane);
      data.fill(game.placedThisTurn / game.config.maxPlacesPerTurn, 3 * plane, 4 * plane);
      data.fill(encodeStoneCount(game.stones[foe]), 4 * plane, 5 * plane);
      for (const [offset, color] of [[5, me], [9, foe]]) {
        const { best, forks3 } = threatGrids(game, color);
        for (let i = 0; i < plane; i++) {
          if (game.board[i] !== EMPTY) continue;
          if (best[i] >= need - 1) data[(offset + 0) * plane + i] = 1;
          if (best[i] >= need - 2) data[(offset + 1) * plane + i] = 1;
          if (best[i] >= need - 3) data[(offset + 2) * plane + i] = 1;
          if (forks3[i] >= 2) data[(offset + 3) * plane + i] = 1;
        }
      }
      return data;
    }

    data.fill(Math.min(game.stones[me], 12) / STONE_SCALE, 2 * plane, 3 * plane);
    data.fill(game.placedThisTurn / game.config.maxPlacesPerTurn, 3 * plane, 4 * plane);
    data.fill(Math.min(game.stones[foe], 12) / STONE_SCALE, 4 * plane, 5 * plane);
    data.fill(me === BLACK ? 1 : 0, 5 * plane, 6 * plane);
    for (const [offset, color] of [[6, me], [11, foe]]) {
      const { best, forks4, forks3 } = threatGrids(game, color);
      for (let i = 0; i < plane; i++) {
        if (game.board[i] !== EMPTY) continue;
        if (best[i] >= need - 1) data[(offset + 0) * plane + i] = 1;
        if (best[i] >= need - 2) data[(offset + 1) * plane + i] = 1;
        if (best[i] >= need - 3) data[(offset + 2) * plane + i] = 1;
        if (forks4[i] >= 2) data[(offset + 3) * plane + i] = 1;
        if (forks3[i] >= 2) data[(offset + 4) * plane + i] = 1;
      }
    }
    return data;
  }

  async function evaluate(games) {
    if (!games.length) return { policy: new Float32Array(), value: new Float32Array(), stride: 0 };
    const n = games[0].size;
    const sess = await session();

    async function runWithPlanes(planes) {
      const planeSize = planes * n * n;
      const data = new Float32Array(games.length * planeSize);
      games.forEach((g, k) => data.set(encodeStateForModel(g, planes), k * planeSize));
      const tensor = new ort.Tensor('float32', data, [games.length, planes, n, n]);
      return sess.run({ board: tensor });
    }

    let out;
    if (cachedInputPlanes !== null) {
      out = await runWithPlanes(cachedInputPlanes);
    } else {
      let err13 = null;
      try {
        out = await runWithPlanes(13);
        cachedInputPlanes = 13;
      } catch (err) {
        err13 = err;
        try {
          out = await runWithPlanes(16);
          cachedInputPlanes = 16;
          console.warn('구형 16평면 ONNX를 감지했습니다. 최신 모델은 13평면입니다.');
        } catch (err16) {
          throw new Error(`ONNX 입력 규격 확인 실패 (13-plane: ${err13.message}; 16-plane: ${err16.message})`);
        }
      }
    }

    return { policy: out.policy.data, value: out.value.data, stride: n * n + 1 };
  }

  function legalProbs(logits, game, offset, stride) {
    const n = game.size;
    const endIndex = n * n;
    const probs = new Float64Array(stride);
    let max = -Infinity;

    for (let i = 0; i <= endIndex; i++) {
      if (i < endIndex && game.board[i] !== 0) continue;
      if (i < endIndex && !game.canPlace()) continue;
      const v = logits[offset + i];
      if (Number.isFinite(v) && v > max) max = v;
    }
    if (!Number.isFinite(max)) return probs;

    let total = 0;
    for (let i = 0; i <= endIndex; i++) {
      if (i < endIndex && (game.board[i] !== 0 || !game.canPlace())) continue;
      const logit = Number.isFinite(logits[offset + i]) ? logits[offset + i] : -Infinity;
      const e = Math.exp(Math.max(-745, Math.min(0, logit - max)));
      probs[i] = e;
      total += e;
    }
    if (!(total > 0) || !Number.isFinite(total)) return probs;
    for (let i = 0; i <= endIndex; i++) probs[i] /= total;
    return probs;
  }

  function valueWinProb(value) {
    const v = Number.isFinite(value) ? Math.max(-1, Math.min(1, value)) : -1;
    return Math.max(VALUE_WEIGHT_FLOOR, Math.min(1, (v + 1) * 0.5));
  }

  function policyValueWeight(policyProb, rootValueScore) {
    return Math.max(Number.isFinite(policyProb) ? policyProb : 0, 0) * valueWinProb(rootValueScore);
  }

  function emptyEndStockMultiplier(game) {
    if (game.placedThisTurn > 0 || !game.canPlace()) return 1.0;
    const stones = Math.max(0, Math.floor(Number(game.stones[game.current]) || 0));
    return EMPTY_END_STOCK_LUT[Math.min(stones, END_STOCK_LUT_MAX)];
  }

  function actionModifier(game, action) {
    const endIndex = game.size * game.size;
    return action === endIndex ? emptyEndStockMultiplier(game) : 1.0;
  }

  function effectiveTemperature(game, baseTemperature) {
    const base = Number.isFinite(baseTemperature) ? Math.max(0, baseTemperature) : 0.5;
    if (game.stonesOnBoard === 0) return OPENING_TEMPERATURE;
    if (game.stonesOnBoard < EARLY_ACTIONS) return base * EARLY_TEMPERATURE_MULTIPLIER;
    return base;
  }

  function policyCutoffForLevel() {
    return DEFAULT_POLICY_CUTOFF_REL;
  }

  function applyWeightCutoff(game, indices, weightFn, cutoffRel = DEFAULT_WEIGHT_CUTOFF_REL) {
    if (game.stonesOnBoard === 0 || indices.length <= 1) return indices;

    const relative = Number.isFinite(cutoffRel)
      ? Math.max(0, Math.min(1, cutoffRel))
      : DEFAULT_WEIGHT_CUTOFF_REL;
    let best = 0;
    for (const i of indices) {
      const w = weightFn(i);
      if (Number.isFinite(w) && w > best) best = w;
    }
    if (!(best > 0)) return indices;

    const cutoff = best * relative;
    const kept = indices.filter((i) => {
      const w = weightFn(i);
      return Number.isFinite(w) && w >= cutoff;
    });
    return kept.length ? kept : indices;
  }

  function sampleWeightedIndices(indices, weightFn, temperature, rng = Math.random) {
    if (!indices.length) return -1;
    if (!(temperature > 0)) {
      const weights = indices.map((i) => {
        const w = weightFn(i);
        return Number.isFinite(w) ? w : 0;
      });
      const scale = Math.max(1e-12, ...weights.map((w) => Math.abs(w)));
      let bestPos = 0;
      let bestRank = weights[0] + rng() * scale * POLICY_NEAR_TIE_REL_TOL;
      for (let j = 1; j < indices.length; j++) {
        const rank = weights[j] + rng() * scale * POLICY_NEAR_TIE_REL_TOL;
        if (rank > bestRank) { bestPos = j; bestRank = rank; }
      }
      return indices[bestPos];
    }
    const scaled = indices.map((i) => Math.pow(Math.max(weightFn(i), 1e-12), 1 / temperature));
    const total = scaled.reduce((sum, w) => sum + w, 0);
    if (!(total > 0) || !Number.isFinite(total)) return indices[0];
    let roll = rng() * total;
    for (let j = 0; j < indices.length; j++) {
      roll -= scaled[j];
      if (roll <= 0) return indices[j];
    }
    return indices[indices.length - 1];
  }

  function openingActionFromPolicy(game, probs, policyCutoffRel = DEFAULT_POLICY_CUTOFF_REL, rng = Math.random) {
    if (game.stonesOnBoard !== 0 || !game.canPlace()) return -1;
    const endIndex = game.size * game.size;
    const placements = [];
    let bestPlacement = 0;
    for (let i = 0; i < endIndex; i++) {
      if (game.board[i] !== EMPTY) continue;
      const p = Number.isFinite(probs[i]) ? Math.max(0, probs[i]) : 0;
      if (p > bestPlacement) bestPlacement = p;
      placements.push(i);
    }

    const relative = Number.isFinite(policyCutoffRel)
      ? Math.max(0, Math.min(1, policyCutoffRel))
      : DEFAULT_POLICY_CUTOFF_REL;
    const threshold = bestPlacement * relative;
    const candidates = bestPlacement > 0
      ? placements.filter((i) => Number.isFinite(probs[i]) && probs[i] >= threshold)
      : placements.slice();
    candidates.push(endIndex);

    return sampleWeightedIndices(
      candidates,
      (i) => Math.max(Number.isFinite(probs[i]) ? probs[i] : 0, 0) * actionModifier(game, i),
      OPENING_TEMPERATURE,
      rng,
    );
  }

  function findImmediateWin(game) {
    if (!game.canPlace()) return -1;
    const { best } = threatGrids(game, game.current);
    const need = game.config.winLength;
    for (let i = 0; i < game.size * game.size; i++) {
      if (game.board[i] === 0 && best[i] >= need - 1) return i;
    }
    return -1;
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

  function playEvaluationAction(game, action) {
    const child = game.clone();
    const endIndex = game.size * game.size;
    child.play(toMove(action, game.size));
    if (action !== endIndex && !child.isOver && !child.canPlace()) child.play(END_TURN);
    return child;
  }

  function availablePlacements(game, color) {
    if (color === game.current) {
      return Math.max(0, Math.min(game.stones[color], game.config.maxPlacesPerTurn - game.placedThisTurn));
    }
    return Math.max(0, Math.min(game.stones[color], game.config.maxPlacesPerTurn));
  }

  function oneTurnWinningWindows(game, color) {
    if (game.isOver) return [];
    const moves = availablePlacements(game, color);
    if (moves <= 0) return [];
    const n = game.size;
    const need = game.config.winLength;
    const foe = otherColor(color);
    const windows = [];

    for (const [dr, dc] of DIRECTIONS) {
      for (let r = 0; r < n; r++) {
        for (let c = 0; c < n; c++) {
          const er = r + dr * (need - 1);
          const ec = c + dc * (need - 1);
          if (er < 0 || er >= n || ec < 0 || ec >= n) continue;
          const empties = [];
          let blocked = false;
          for (let k = 0; k < need; k++) {
            const i = (r + dr * k) * n + (c + dc * k);
            const v = game.board[i];
            if (v === foe) { blocked = true; break; }
            if (v === 0) empties.push(i);
          }
          if (!blocked && empties.length > 0 && empties.length <= moves) windows.push(empties);
        }
      }
    }
    return windows;
  }

  function oneTurnWinningSquares(game, color) {
    const out = new Set();
    for (const window of oneTurnWinningWindows(game, color)) for (const i of window) out.add(i);
    return [...out];
  }

  function oneTurnDefenseSquares(game, foeColor) {
    const windows = oneTurnWinningWindows(game, foeColor);
    if (!windows.length) return [];
    const moves = availablePlacements(game, game.current);
    if (moves <= 0) return [];
    const candidates = [...new Set(windows.flat())];

    function canCover(remaining, slots) {
      if (!remaining.length) return true;
      if (slots <= 0) return false;
      let target = remaining[0];
      for (const window of remaining) if (window.length < target.length) target = window;
      for (const cell of target) {
        const next = remaining.filter((window) => !window.includes(cell));
        if (canCover(next, slots - 1)) return true;
      }
      return false;
    }

    return candidates.filter((cell) => {
      const remaining = windows.filter((window) => !window.includes(cell));
      return canCover(remaining, moves - 1);
    });
  }

  function candidateActions(game, probs, policyCutoffRel = DEFAULT_POLICY_CUTOFF_REL) {
    const n = game.size;
    const endIndex = n * n;
    if (!game.canPlace()) return [endIndex];

    const empties = [];
    for (let i = 0; i < endIndex; i++) if (game.board[i] === 0) empties.push(i);
    empties.sort((a, b) => probs[b] - probs[a]);

    const ownWinning = oneTurnWinningSquares(game, game.current);
    if (ownWinning.length) return ownWinning.sort((a, b) => probs[b] - probs[a]);

    const foe = otherColor(game.current);
    const foeWinning = oneTurnWinningSquares(game, foe);
    if (foeWinning.length) {
      const defenses = oneTurnDefenseSquares(game, foe);
      const forced = defenses.length ? defenses : foeWinning;
      return forced.sort((a, b) => probs[b] - probs[a]);
    }

    const relative = Number.isFinite(policyCutoffRel)
      ? Math.max(0, Math.min(1, policyCutoffRel))
      : DEFAULT_POLICY_CUTOFF_REL;
    const bestPolicy = empties.length && Number.isFinite(probs[empties[0]])
      ? Math.max(0, probs[empties[0]])
      : 0;
    const threshold = bestPolicy * relative;
    const placements = bestPolicy > 0
      ? empties.filter((i) => Number.isFinite(probs[i]) && probs[i] >= threshold)
      : empties;

    return [...placements, endIndex];
  }

  async function pickWithLookahead(game, probs, policyCutoffRel, evaluateFn, temperature = 0, rng = Math.random, cutoffRel = DEFAULT_WEIGHT_CUTOFF_REL) {
    const rootColor = game.current;
    const actions = candidateActions(game, probs, policyCutoffRel);
    const children = actions.map((action) => playEvaluationAction(game, action));

    const scores = new Array(actions.length);
    const pending = [];
    const pendingAt = [];
    children.forEach((child, k) => {
      const terminal = rootValue(child, rootColor);
      if (terminal !== null) scores[k] = terminal;
      else { pending.push(child); pendingAt.push(k); }
    });

    if (pending.length) {
      const { value } = await evaluateFn(pending);
      pendingAt.forEach((k, j) => {
        const child = children[k];
        scores[k] = child.current === rootColor ? value[j] : -value[j];
      });
    }

    const actionWeight = (k) => (
      policyValueWeight(probs[actions[k]], scores[k]) * actionModifier(game, actions[k])
    );

    const indices = actions.map((_, k) => k);
    const winningIndices = indices.filter((k) => scores[k] >= 1);
    const rankedIndices = winningIndices.length
      ? winningIndices
      : applyWeightCutoff(game, indices, actionWeight, cutoffRel);

    const chosen = sampleWeightedIndices(
      rankedIndices,
      actionWeight,
      temperature,
      rng,
    );
    return chosen < 0 ? -1 : actions[chosen];
  }

  async function planTurnWithModel(game, level, evaluateFn, options = {}) {
    const n = game.size;
    const endIndex = n * n;
    const baseTemperature = Number.isFinite(level.temperature) ? level.temperature : 0.5;
    const cutoffRel = Number.isFinite(level.cutoffRel) ? level.cutoffRel : DEFAULT_WEIGHT_CUTOFF_REL;
    const policyCutoffRel = policyCutoffForLevel();
    const rng = options.rng || Math.random;
    const sim = game.clone();
    const moves = [];

    while (sim.canPlace()) {
      if (sim.stonesOnBoard === 0) {
        const { policy, stride } = await evaluateFn([sim]);
        const probs = legalProbs(policy, sim, 0, stride);
        const index = openingActionFromPolicy(sim, probs, policyCutoffRel, rng);
        if (index < 0 || index === endIndex) break;
        const r = Math.floor(index / n), c = index % n;
        sim.place(r, c);
        moves.push([r, c]);
        if (sim.isOver) break;
        continue;
      }

      const win = findImmediateWin(sim);
      if (win >= 0) {
        sim.place(Math.floor(win / n), win % n);
        moves.push([Math.floor(win / n), win % n]);
        if (sim.isOver) break;
        continue;
      }

      const { policy, stride } = await evaluateFn([sim]);
      const probs = legalProbs(policy, sim, 0, stride);
      const temperature = effectiveTemperature(sim, baseTemperature);
      const index = await pickWithLookahead(
        sim, probs, policyCutoffRel, evaluateFn, temperature, rng, cutoffRel,
      );

      if (index === endIndex || index < 0) break;
      const r = Math.floor(index / n), c = index % n;
      sim.place(r, c);
      moves.push([r, c]);
      if (sim.isOver) break;
    }
    return moves;
  }

  window.MODEL_PLAN_TURN = async (game) => {
    if (game.size !== MODEL_BOARD_SIZE) {
      const msg = `모델은 ${MODEL_BOARD_SIZE}줄 판으로만 학습돼 있습니다.`;
      announce('failed', msg);
      throw new Error(msg);
    }
    const level = (typeof window.AI_LEVEL === 'function' ? window.AI_LEVEL() : null) || {};
    return planTurnWithModel(game, level, evaluate);
  };

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
      planTurnWithModel, pickWithLookahead, candidateActions, openingActionFromPolicy,
      oneTurnWinningWindows, oneTurnWinningSquares, oneTurnDefenseSquares,
      availablePlacements, legalProbs, findImmediateWin,
      valueWinProb, policyValueWeight, emptyEndStockMultiplier,
      actionModifier, effectiveTemperature, policyCutoffForLevel, applyWeightCutoff,
      playEvaluationAction, encodeStateForModel,
    };
  }
})();