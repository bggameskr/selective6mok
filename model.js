/* 학습한 모델을 브라우저 게임에 붙인다.

   쓰는 법
   1. 코랩 노트북에서 받은 sixmok.onnx 를 index.html 과 같은 폴더에 둔다.
   2. 그 폴더를 웹에 올리거나 간단한 서버로 연다 (file:// 로 열면 모델을 못 읽는다).

   난이도에 따라 제한 탐색을 한다.
     쉬움   0갈래 — 신경망이 낸 수를 그대로 (온도만 높여 흔든다)
     보통   3갈래 — 전술 필수 후보를 보존한 1-ply 가치 탐색
     어려움 6갈래 — 상위 후보에서 현재 턴 전체와 상대 턴 전체를 제한 탐색
   후보 국면은 가능한 한 배치로 묶어 가치 헤드로 평가한다.

   입력을 만드는 encodeState 는 engine.js 안에 있다. 학습에 쓴 encoding.py 와
   규격이 같아야 하며, parity_check 로 확인할 수 있다. */

(() => {
  'use strict';

  const MODEL_URL = 'sixmok.onnx';
  const MODEL_BOARD_SIZE = 19;
  const POLICY_BLEND = 0.1;
  const DEEP_ROOTS = 4;          // 어려움에서 턴 단위 탐색을 적용할 첫 수 개수
  const DEEP_WIDTH = 3;          // 한 턴 내부 각 착수에서 펼칠 후보 폭
  const TURN_NODE_CAP = 384;     // 브라우저 폭발 방지용 frontier 상한

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
    let max = -Infinity;

    for (let i = 0; i <= endIndex; i++) {
      if (i < endIndex && game.board[i] !== 0) continue;
      if (i < endIndex && !game.canPlace()) continue;
      const v = logits[offset + i];
      if (v > max) max = v;
    }

    let total = 0;
    for (let i = 0; i <= endIndex; i++) {
      if (i < endIndex && (game.board[i] !== 0 || !game.canPlace())) continue;
      const e = Math.exp(logits[offset + i] - max);
      probs[i] = e;
      total += e;
    }
    for (let i = 0; i <= endIndex; i++) probs[i] /= total;
    return probs;
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

  function availablePlacements(game, color) {
    if (color === game.current) {
      return Math.max(
        0,
        Math.min(
          game.stones[color],
          game.config.maxPlacesPerTurn - game.placedThisTurn
        )
      );
    }
    return Math.max(
      0,
      Math.min(game.stones[color], game.config.maxPlacesPerTurn)
    );
  }

  /** color가 다음 한 턴 안에 완성할 수 있는 실제 6칸 승리창들. */
  function oneTurnWinningWindows(game, color) {
    if (game.isOver) return [];

    const moves = Math.min(availablePlacements(game, color), 3);
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
            if (v === foe) {
              blocked = true;
              break;
            }
            if (v === 0) empties.push(i);
          }

          if (!blocked && empties.length > 0 && empties.length <= moves) {
            windows.push(empties);
          }
        }
      }
    }
    return windows;
  }

  function oneTurnWinningSquares(game, color) {
    const out = new Set();
    for (const window of oneTurnWinningWindows(game, color)) {
      for (const i of window) out.add(i);
    }
    return [...out];
  }

  /** 현재 플레이어가 남은 착수 수 안에 foeColor의 모든 한 턴 승리창을 막을 수 있는 첫 수. */
  function oneTurnDefenseSquares(game, foeColor) {
    const windows = oneTurnWinningWindows(game, foeColor);
    if (!windows.length) return [];

    const moves = Math.min(availablePlacements(game, game.current), 3);
    if (moves <= 0) return [];

    const candidates = [...new Set(windows.flat())];

    function canCover(remaining, slots) {
      if (!remaining.length) return true;
      if (slots <= 0) return false;

      let target = remaining[0];
      for (const window of remaining) {
        if (window.length < target.length) target = window;
      }

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

  function candidateActions(game, probs, width) {
    const n = game.size;
    const endIndex = n * n;

    if (!game.canPlace()) return [endIndex];

    const empties = [];
    for (let i = 0; i < endIndex; i++) {
      if (game.board[i] === 0) empties.push(i);
    }
    empties.sort((a, b) => probs[b] - probs[a]);

    const ownWinning = oneTurnWinningSquares(game, game.current);
    if (ownWinning.length) {
      return ownWinning.sort((a, b) => probs[b] - probs[a]);
    }

    const foe = otherColor(game.current);
    const foeWinning = oneTurnWinningSquares(game, foe);
    if (foeWinning.length) {
      const defenses = oneTurnDefenseSquares(game, foe);
      const forced = defenses.length ? defenses : foeWinning;
      return forced.sort((a, b) => probs[b] - probs[a]);
    }

    const foeThreat = threatGrids(game, foe);
    const ownThreat = threatGrids(game, game.current);
    const foeForks = [];
    const ownForks = [];

    for (const i of empties) {
      if (foeThreat.forks4[i] >= 2 || foeThreat.forks3[i] >= 2) foeForks.push(i);
      if (ownThreat.forks4[i] >= 2 || ownThreat.forks3[i] >= 2) ownForks.push(i);
    }

    if (foeForks.length || ownForks.length) {
      const tactical = new Set([...foeForks, ...ownForks]);
      const policyAllowance = Math.max(2, Math.floor(width / 2));
      for (const a of empties.slice(0, policyAllowance)) tactical.add(a);
      return [...tactical].sort((a, b) => probs[b] - probs[a]);
    }

    const chosen = new Set(empties.slice(0, width));
    chosen.add(endIndex);
    return [...chosen].sort((a, b) => {
      if (a === endIndex) return 1;
      if (b === endIndex) return -1;
      return probs[b] - probs[a];
    });
  }

  function sample(probs, stride, temperature, rng = Math.random) {
    if (!(temperature > 0)) {
      let best = 0;
      for (let i = 1; i < stride; i++) if (probs[i] > probs[best]) best = i;
      return best;
    }

    const weights = new Float64Array(stride);
    let total = 0;
    for (let i = 0; i < stride; i++) {
      if (probs[i] <= 0) continue;
      weights[i] = Math.pow(probs[i], 1 / temperature);
      total += weights[i];
    }

    let roll = rng() * total;
    for (let i = 0; i < stride; i++) {
      roll -= weights[i];
      if (roll <= 0 && weights[i] > 0) return i;
    }
    return stride - 1;
  }

  /**
   * 같은 사람의 턴이 끝날 때까지 후보 수열을 펼친다.
   * 각 node는 { game, rootIndex, seqId?, prior } 형태이며 메타데이터를 보존한다.
   */
  async function expandWholeTurn(nodes, turnColor, evaluateFn, width) {
    let frontier = nodes.slice();
    const done = [];

    // 한 턴 최대 3착수 + END_TURN이므로 네 단계면 충분하다.
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

        for (const action of actions) {
          const child = g.clone();
          child.play(toMove(action, g.size));
          const prior = node.prior * Math.max(probs[action], 1e-12);
          next.push({ ...node, game: child, prior });
        }
      });

      next.sort((a, b) => b.prior - a.prior);
      frontier = next.slice(0, TURN_NODE_CAP);
    }

    // 안전장치: 상한에 걸려 아직 턴이 안 끝난 상태는 강제로 턴 종료시켜 평가한다.
    for (const node of frontier) {
      const g = node.game.clone();
      if (!g.isOver && g.current === turnColor) g.play(END_TURN);
      done.push({ ...node, game: g });
    }

    return done;
  }

  /** 네트워크 가치보다 확실한 한 턴 강제승/강제패를 우선한다. */
  function tacticalLeafValue(game, rootColor) {
    const terminal = rootValue(game, rootColor);
    if (terminal !== null) return terminal;

    // 두 턴 탐색이 끝나면 보통 rootColor의 차례다.
    if (game.current !== rootColor) return null;

    // 지금 한 턴 안에 이길 수 있으면 확정 승리로 본다.
    if (oneTurnWinningSquares(game, rootColor).length) return 1;

    // 상대가 다음 자기 턴에 이길 수 있고, 이번 턴 안에 모든 승리창을 막을 방법이 없으면 확정 패배다.
    const foe = otherColor(rootColor);
    if (oneTurnWinningWindows(game, foe).length && !oneTurnDefenseSquares(game, foe).length) {
      return -1;
    }
    return null;
  }

  async function evaluateLeaves(nodes, rootColor, evaluateFn) {
    const scores = new Array(nodes.length);
    const pending = [];
    const pendingAt = [];

    nodes.forEach((node, i) => {
      const forced = tacticalLeafValue(node.game, rootColor);
      if (forced !== null) scores[i] = forced;
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

  /**
   * 보통: 기존 1-ply.
   * 어려움: 상위 첫 수들에 대해 '내 남은 턴 전체 → 상대 턴 전체'를 펼쳐 minimax한다.
   */
  async function pickWithLookahead(game, probs, width, evaluateFn, deep = false) {
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

      const rootStarts = expandable.map(({ action, k }) => ({
        game: children[k],
        rootIndex: k,
        prior: Math.max(probs[action], 1e-12),
      }));

      const alreadyEnded = [];
      const stillRootTurn = [];
      for (const node of rootStarts) {
        if (node.game.current === rootColor) stillRootTurn.push(node);
        else alreadyEnded.push(node);
      }

      const rootTurnEnds = alreadyEnded.concat(
        stillRootTurn.length
          ? await expandWholeTurn(stillRootTurn, rootColor, evaluateFn, DEEP_WIDTH)
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
        ? await expandWholeTurn(opponentStarts, opponentColor, evaluateFn, DEEP_WIDTH)
        : [];

      const allLeaves = terminalRootEnds.concat(opponentTurnEnds);
      const leafScores = await evaluateLeaves(allLeaves, rootColor, evaluateFn);

      // 같은 root-turn 수열에서는 상대가 최악의 결과를 선택한다.
      const seqWorst = new Map();
      allLeaves.forEach((node, i) => {
        const prev = seqWorst.get(node.seqId);
        const score = leafScores[i];
        if (prev === undefined || score < prev) seqWorst.set(node.seqId, score);
      });

      // 첫 수 이후의 같은 턴 추가 착수는 우리 선택이므로 가장 좋은 수열을 택한다.
      const rootBest = new Map();
      for (const node of rootTurnEnds) {
        const seqScore = seqWorst.get(node.seqId);
        if (seqScore === undefined) continue;
        const prev = rootBest.get(node.rootIndex);
        if (prev === undefined || seqScore > prev) rootBest.set(node.rootIndex, seqScore);
      }

      for (const [k, score] of rootBest) scores[k] = score;
    }

    let best = 0;
    for (let k = 1; k < actions.length; k++) {
      const a = scores[k] + POLICY_BLEND * probs[actions[k]];
      const b = scores[best] + POLICY_BLEND * probs[actions[best]];
      if (a > b) best = k;
    }
    return actions[best];
  }

  async function planTurnWithModel(game, level, evaluateFn) {
    const n = game.size;
    const endIndex = n * n;
    const temperature = level.temperature || 0;
    const width = level.lookahead || 0;
    const deep = width >= 6;

    const sim = game.clone();
    const moves = [];

    while (sim.canPlace()) {
      const win = findImmediateWin(sim);
      if (win >= 0) {
        sim.place(Math.floor(win / n), win % n);
        moves.push([Math.floor(win / n), win % n]);
        if (sim.isOver) break;
        continue;
      }

      const { policy, stride } = await evaluateFn([sim]);
      const probs = legalProbs(policy, sim, 0, stride);

      const index = width > 0
        ? await pickWithLookahead(sim, probs, width, evaluateFn, deep)
        : sample(probs, stride, temperature);

      if (index === endIndex) break;
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
      oneTurnWinningWindows,
      oneTurnWinningSquares,
      oneTurnDefenseSquares,
      availablePlacements,
      legalProbs,
      sample,
      findImmediateWin,
      expandWholeTurn,
      tacticalLeafValue,
    };
  }
})();
