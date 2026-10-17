/* v5.10.15 experiment adapter.
   실제 모델 구현은 model-core.js에 두고, 보통 난이도만 policy 질량 기반
   adaptive 8→16→32 1-ply policy×value 평가로 교체한다.
   experiment/no-tactical-corrections 브랜치의 원칙대로 전술 강제 후보 삽입은 하지 않는다.
   각 색이 첫 돌을 둔 턴부터 3개 완료 턴까지 오프닝 temperature를 적용한다. */

(() => {
  'use strict';

  const CORE_URL = 'model-core.js';
  const MODEL_URL = 'sixmok.onnx';
  const MODEL_BOARD_SIZE = 19;
  const OPENING_TURNS = 3;
  const NORMAL_OPENING_TEMPERATURE = 1.0;
  const HARD_OPENING_TEMPERATURE = 0.8;
  const VALUE_WEIGHT_FLOOR = 1e-6;
  const POLICY_TIE_REL_EPS = 1e-4;
  const MEDIUM_STAGES = [8, 16, 32];
  const MEDIUM_MASS_8 = 0.80;
  const MEDIUM_MASS_16 = 0.95;

  let sharedSessionPromise = null;

  function installSharedSessionCache() {
    if (typeof ort === 'undefined' || !ort.InferenceSession || ort.InferenceSession.__sixmokShared) return;
    const originalCreate = ort.InferenceSession.create.bind(ort.InferenceSession);
    ort.InferenceSession.create = (url, options) => {
      if (url === MODEL_URL) {
        if (!sharedSessionPromise) sharedSessionPromise = originalCreate(url, options);
        return sharedSessionPromise;
      }
      return originalCreate(url, options);
    };
    ort.InferenceSession.__sixmokShared = true;
  }

  function inOpeningWindow(game, color = game.current) {
    let started = false;
    let completed = 0;
    for (const [playedColor, move] of game.history) {
      if (playedColor !== color) continue;
      if (move === END_TURN) {
        if (started) completed += 1;
      } else {
        started = true;
      }
    }
    return completed < OPENING_TURNS;
  }

  function rankActions(actions, probs, rng = Math.random) {
    const ranked = Array.from(actions).sort((a, b) => probs[b] - probs[a]);
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

  let currentAdaptiveProbs = null;
  let currentAdaptiveRng = Math.random;

  function prefixWithEnd(ordered, width, endIndex) {
    if (ordered.length <= width) return ordered.slice();
    const chosen = ordered.slice(0, Math.max(1, width));
    if (!chosen.includes(endIndex)) {
      chosen[chosen.length - 1] = endIndex;
      return rankActions(chosen, currentAdaptiveProbs, currentAdaptiveRng);
    }
    return chosen;
  }

  function adaptiveCandidates(game, probs, rng = Math.random) {
    const endIndex = game.size * game.size;
    if (!game.canPlace()) return [endIndex];
    const legal = [endIndex];
    for (let i = 0; i < endIndex; i++) if (game.board[i] === EMPTY) legal.push(i);
    const ordered = rankActions(legal, probs, rng);

    currentAdaptiveProbs = probs;
    currentAdaptiveRng = rng;
    const c8 = prefixWithEnd(ordered, MEDIUM_STAGES[0], endIndex);
    const mass8 = c8.reduce((s, a) => s + probs[a], 0);
    if (mass8 >= MEDIUM_MASS_8) return c8;

    const c16 = prefixWithEnd(ordered, MEDIUM_STAGES[1], endIndex);
    const mass16 = c16.reduce((s, a) => s + probs[a], 0);
    if (mass16 >= MEDIUM_MASS_16) return c16;

    return prefixWithEnd(ordered, MEDIUM_STAGES[2], endIndex);
  }

  async function evaluate(games) {
    if (!games.length) return { policy: new Float32Array(), value: new Float32Array(), stride: 0 };
    const n = games[0].size;
    const planeSize = N_PLANES * n * n;
    const data = new Float32Array(games.length * planeSize);
    games.forEach((g, k) => data.set(encodeState(g), k * planeSize));
    if (!sharedSessionPromise) {
      sharedSessionPromise = ort.InferenceSession.create(MODEL_URL, { executionProviders: ['wasm'] });
    }
    const sess = await sharedSessionPromise;
    const tensor = new ort.Tensor('float32', data, [games.length, N_PLANES, n, n]);
    const out = await sess.run({ board: tensor });
    return { policy: out.policy.data, value: out.value.data, stride: n * n + 1 };
  }

  function legalProbs(logits, game, offset, stride) {
    const endIndex = game.size * game.size;
    const probs = new Float64Array(stride);
    if (!game.canPlace()) {
      probs[endIndex] = 1;
      return probs;
    }
    const legal = [endIndex];
    for (let i = 0; i < endIndex; i++) if (game.board[i] === EMPTY) legal.push(i);
    let max = -Infinity;
    for (const a of legal) max = Math.max(max, Number.isFinite(logits[offset + a]) ? logits[offset + a] : -1e9);
    let total = 0;
    for (const a of legal) {
      const raw = Number.isFinite(logits[offset + a]) ? logits[offset + a] : -1e9;
      const e = Math.exp(raw - max);
      probs[a] = e;
      total += e;
    }
    if (!(total > 0) || !Number.isFinite(total)) {
      const u = 1 / legal.length;
      for (const a of legal) probs[a] = u;
      return probs;
    }
    for (const a of legal) probs[a] /= total;
    return probs;
  }

  function rootValue(game, rootColor) {
    if (!game.isOver) return null;
    if (game.winner === null) return 0;
    return game.winner === rootColor ? 1 : -1;
  }

  function valueWinProb(value) {
    const raw = Number.isFinite(value) ? value : -1;
    const clipped = Math.max(-1, Math.min(1, raw));
    return Math.max(VALUE_WEIGHT_FLOOR, Math.min(1, (clipped + 1) * 0.5));
  }

  function policyValueWeight(policyProb, rootValueScore) {
    return Math.max(Number(policyProb) || 0, 0) * valueWinProb(rootValueScore);
  }

  function sampleWeighted(indices, weightFn, temperature, rng = Math.random) {
    if (!indices.length) return -1;
    if (!(temperature > 0)) {
      const values = indices.map((i) => weightFn(i));
      const scale = Math.max(1e-12, ...values.map((v) => Math.abs(v)));
      let bestPos = 0;
      let bestRank = values[0] + rng() * scale * POLICY_TIE_REL_EPS;
      for (let k = 1; k < indices.length; k++) {
        const rank = values[k] + rng() * scale * POLICY_TIE_REL_EPS;
        if (rank > bestRank) { bestPos = k; bestRank = rank; }
      }
      return indices[bestPos];
    }
    const weights = indices.map((i) => Math.pow(Math.max(weightFn(i), Number.MIN_VALUE), 1 / temperature));
    const total = weights.reduce((s, w) => s + w, 0);
    if (!(total > 0) || !Number.isFinite(total)) return indices[0];
    let roll = rng() * total;
    for (let k = 0; k < indices.length; k++) {
      roll -= weights[k];
      if (roll <= 0) return indices[k];
    }
    return indices[indices.length - 1];
  }

  async function pickAdaptive(game, probs, temperature, rng = Math.random) {
    const n = game.size;
    const endIndex = n * n;
    const rootColor = game.current;
    const actions = adaptiveCandidates(game, probs, rng);
    const children = actions.map((action) => {
      const child = game.clone();
      child.play(action === endIndex ? END_TURN : [Math.floor(action / n), action % n]);
      return child;
    });

    const scores = new Array(actions.length);
    const pending = [];
    const pendingAt = [];
    children.forEach((child, k) => {
      const terminal = rootValue(child, rootColor);
      if (terminal !== null) scores[k] = terminal;
      else { pending.push(child); pendingAt.push(k); }
    });
    if (pending.length) {
      const out = await evaluate(pending);
      pendingAt.forEach((k, j) => {
        const child = children[k];
        scores[k] = child.current === rootColor ? out.value[j] : -out.value[j];
      });
    }

    const indices = actions.map((_, k) => k);
    const terminalWins = indices.filter((k) => rootValue(children[k], rootColor) === 1);
    const eligible = terminalWins.length ? terminalWins : indices;
    const chosen = sampleWeighted(
      eligible,
      (k) => policyValueWeight(probs[actions[k]], scores[k]),
      temperature,
      rng,
    );
    return chosen < 0 ? -1 : actions[chosen];
  }

  async function planAdaptiveMedium(game, level, rng = Math.random) {
    const n = game.size;
    const endIndex = n * n;
    const sim = game.clone();
    const moves = [];
    let temperature = Number(level.temperature) || 0;
    if (inOpeningWindow(sim, sim.current)) temperature = Math.max(temperature, NORMAL_OPENING_TEMPERATURE);

    while (sim.canPlace()) {
      const out = await evaluate([sim]);
      const probs = legalProbs(out.policy, sim, 0, out.stride);
      const action = await pickAdaptive(sim, probs, temperature, rng);
      if (action === endIndex || action < 0) break;
      const r = Math.floor(action / n), c = action % n;
      sim.place(r, c);
      moves.push([r, c]);
      if (sim.isOver) break;
    }
    return moves;
  }

  function installOpeningPatch() {
    const corePlanTurn = window.MODEL_PLAN_TURN;
    if (typeof corePlanTurn !== 'function') {
      console.warn('model-core.js가 MODEL_PLAN_TURN을 등록하지 못했습니다.');
      return;
    }

    window.MODEL_PLAN_TURN = (game) => {
      const originalAILevel = typeof window.AI_LEVEL === 'function' ? window.AI_LEVEL : null;
      const level = originalAILevel ? { ...originalAILevel() } : {};
      const width = Number(level.lookahead) || 0;

      // 보통 난이도만 adaptive 8→16→32로 교체한다. 쉬움/어려움은 core를 그대로 쓴다.
      if (width > 0 && width < 6) {
        return planAdaptiveMedium(game, level);
      }

      if (!inOpeningWindow(game, game.current) || !originalAILevel) {
        return corePlanTurn(game);
      }

      window.AI_LEVEL = () => {
        const patched = { ...level };
        const baseTemperature = Number(patched.temperature) || 0;
        if (width >= 6) patched.temperature = Math.max(baseTemperature, HARD_OPENING_TEMPERATURE);
        else if (width > 0) patched.temperature = Math.max(baseTemperature, NORMAL_OPENING_TEMPERATURE);
        return patched;
      };
      try {
        return corePlanTurn(game);
      } finally {
        window.AI_LEVEL = originalAILevel;
      }
    };

    window.MODEL_IN_OPENING_WINDOW = inOpeningWindow;
  }

  installSharedSessionCache();

  const script = document.createElement('script');
  script.src = CORE_URL;
  script.async = false;
  script.onload = installOpeningPatch;
  script.onerror = () => {
    window.MODEL_STATUS = 'failed';
    window.MODEL_STATUS_DETAIL = `${CORE_URL}을 불러오지 못했습니다.`;
    window.dispatchEvent(new CustomEvent('model-status'));
  };
  document.head.appendChild(script);
})();
