/* 학습한 모델을 브라우저 게임에 붙인다.

   쓰는 법
   1. 코랩 노트북에서 받은 sixmok.onnx 를 index.html 과 같은 폴더에 둔다.
   2. 그 폴더를 웹에 올리거나 간단한 서버로 연다 (file:// 로 열면 모델을 못 읽는다).

   난이도에 따라 모델 정책과 가치만으로 제한 탐색한다.
     쉬움   0갈래 — 정책 상위 후보 안에서 온도로 샘플링
     보통   3갈래 — 정책 상위 후보의 1-ply 가치 탐색
     어려움 6갈래 — 상위 후보에서 현재 턴 전체와 상대 턴 전체를 제한 탐색
   어려움은 한 번 고른 현재 턴의 수열을 그대로 실행해 같은 턴 안에서 재탐색하지 않는다.

   즉시승 강제 선택, 강제방어 후보 삽입, 삼삼/사사 후보 보정, 전술 leaf 값 덮어쓰기를
   사용하지 않는다. 실제로 탐색한 수가 게임 종료 상태에 도달한 경우의 승패만 확정값으로 쓴다.
   오프닝도 별도로 변형하거나 대칭 이동하지 않는다.

   입력을 만드는 encodeState 는 engine.js 안에 있다. 학습에 쓴 encoding.py 와
   규격이 같아야 하며, parity_check 로 확인할 수 있다. */

(() => {
  'use strict';

  const MODEL_URL = 'sixmok.onnx';
  const MODEL_BOARD_SIZE = 19;
  const POLICY_BLEND = 0.1;
  const POLICY_TIE_REL_EPS = 1e-5;
  const EASY_CANDIDATE_WIDTH = 8;
  const DEEP_ROOTS = 3;
  const DEEP_WIDTH = 3;
  const TURN_NODE_CAP = 64;

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

  function sigmoid(value) {
    const z = Math.max(-60, Math.min(60, Number.isFinite(value) ? value : 0));
    return 1 / (1 + Math.exp(-z));
  }

  /**
   * 최신 계층형 policy를 결합확률로 바꾼다.
   * 앞 n*n개는 CONTINUE 이후의 조건부 착점 logits이고,
   * 마지막 하나는 END_TURN Bernoulli logit이다.
   */
  function legalProbs(logits, game, offset, stride) {
    const n = game.size;
    const endIndex = n * n;
    const probs = new Float64Array(stride);

    if (!game.canPlace()) {
      probs[endIndex] = 1;
      return probs;
    }

    const boardLegal = [];
    let max = -Infinity;
    for (let i = 0; i < endIndex; i++) {
      if (game.board[i] !== 0) continue;
      boardLegal.push(i);
      const raw = logits[offset + i];
      const v = Number.isFinite(raw) ? raw : -1e9;
      if (v > max) max = v;
    }

    if (!boardLegal.length) {
      probs[endIndex] = 1;
      return probs;
    }

    let total = 0;
    for (const i of boardLegal) {
      const raw = logits[offset + i];
      const v = Number.isFinite(raw) ? raw : -1e9;
      const e = Math.exp(v - max);
      probs[i] = e;
      total += e;
    }

    if (!(total > 0) || !Number.isFinite(total)) {
      total = boardLegal.length;
      for (const i of boardLegal) probs[i] = 1;
    }

    const pEnd = sigmoid(logits[offset + endIndex]);
    const continueMass = 1 - pEnd;
    for (const i of boardLegal) probs[i] = continueMass * probs[i] / total;
    probs[endIndex] = pEnd;
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

  /** 정책값이 사실상 같은 좌표들은 배열 인덱스가 작은 쪽이 계속 이기지 않도록 섞는다. */
  function rankBoardActions(actions, probs, rng = Math.random) {
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

    const boardActions = [];
    for (let i = 0; i < endIndex; i++) {
      if (game.board[i] === 0) boardActions.push(i);
    }

    const rankedBoard = rankBoardActions(boardActions, probs, rng);
    const limit = Math.max(1, width);
    if (limit === 1) {
      return probs[endIndex] > 0.5 ? [endIndex] : rankedBoard.slice(0, 1);
    }

    // END_TURN을 개별 착점 361개와 직접 순위 비교하지 않는다.
    // 탐색 폭 하나는 END에 남기고 나머지를 조건부 착점 상위 후보로 채운다.
    return [endIndex, ...rankedBoard.slice(0, limit - 1)];
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

  /** END/CONTINUE를 먼저 고른 뒤, CONTINUE일 때만 착점 위치를 고른다. */
  function chooseHierarchicalAction(probs, actions, endIndex, temperature, rng = Math.random) {
    if (!actions.length) return -1;

    const hasEnd = actions.includes(endIndex);
    const boardActions = actions.filter((action) => action !== endIndex);
    if (!boardActions.length) return hasEnd ? endIndex : -1;
    if (!hasEnd) return sampleActions(probs, boardActions, temperature, rng);

    const pEnd = Math.max(0, Math.min(1, probs[endIndex]));
    if (!(temperature > 0)) {
      if (pEnd > 0.5) return endIndex;
      return sampleActions(probs, boardActions, 0, rng);
    }

    const safeEnd = Math.max(1e-12, Math.min(1 - 1e-12, pEnd));
    const endLogit = Math.log(safeEnd / (1 - safeEnd));
    const temperedEnd = sigmoid(endLogit / temperature);
    if (rng() < temperedEnd) return endIndex;
    return sampleActions(probs, boardActions, temperature, rng);
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
    game, probs, width, evaluateFn, deep = false, returnPlan = false
  ) {
    const n = game.size;
    const rootColor = game.current;
    const actions = candidateActions(game, probs, width);

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
          score: scores[k] + POLICY_BLEND * probs[action],
        }))
        .filter(({ k }) => !children[k].isOver)
        .sort((a, b) => b.score - a.score)
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

    const indices = deepPlans && deepPlans.size
      ? [...deepPlans.keys()]
      : actions.map((_, k) => k);
    const finalScore = (k) => scores[k] + POLICY_BLEND * probs[actions[k]];

    const terminalWins = indices.filter((k) => rootValue(children[k], rootColor) === 1);
    const rankedIndices = terminalWins.length ? terminalWins : indices;
    let best = rankedIndices[0];
    for (const k of rankedIndices.slice(1)) {
      if (finalScore(k) > finalScore(best)) best = k;
    }

    if (returnPlan) {
      if (deepPlans && deepPlans.has(best)) {
        return {
          action: actions[best],
          turnActions: deepPlans.get(best).turnActions,
        };
      }
      return {
        action: actions[best],
        turnActions: actions[best] === n * n ? [] : [actions[best]],
      };
    }

    return actions[best];
  }

  async function planTurnWithModel(game, level, evaluateFn, options = {}) {
    const n = game.size;
    const endIndex = n * n;
    const temperature = level.temperature || 0;
    const width = level.lookahead || 0;
    const deep = width >= 6;
    const rng = options.rng || Math.random;

    const sim = game.clone();
    const moves = [];

    if (deep) {
      const { policy, stride } = await evaluateFn([sim]);
      const probs = legalProbs(policy, sim, 0, stride);
      const plan = await pickWithLookahead(sim, probs, width, evaluateFn, true, true);

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

      let index;
      if (width > 0) {
        index = await pickWithLookahead(sim, probs, width, evaluateFn, false, false);
      } else {
        const easyActions = candidateActions(sim, probs, EASY_CANDIDATE_WIDTH, rng);
        index = chooseHierarchicalAction(probs, easyActions, endIndex, temperature, rng);
      }

      if (index === endIndex || index < 0) break;
      const r = Math.floor(index / n);
      const c = index % n;
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
      planTurnWithModel,
      pickWithLookahead,
      candidateActions,
      legalProbs,
      rankBoardActions,
      sampleActions,
      chooseHierarchicalAction,
      expandWholeTurn,
      evaluateLeaves,
    };
  }
})();
