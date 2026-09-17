/* 학습한 모델을 브라우저 게임에 붙인다.

   쓰는 법
   1. 코랩 노트북에서 받은 sixmok.onnx 를 index.html 과 같은 폴더에 둔다.
   2. 그 폴더를 웹에 올리거나 간단한 서버로 연다 (file:// 로 열면 모델을 못 읽는다).

   난이도에 따라 제한 탐색을 한다.
     쉬움   0갈래 — 신경망이 낸 수를 그대로 (온도만 높여 흔든다)
     보통   3갈래 — 전술 필수 후보를 보존한 1-ply 가치 탐색
     어려움 6갈래 — 필수 후보 + 상위 후보 일부를 한 단계 더 읽는 제한 2-ply 탐색
   후보 국면은 가능한 한 배치로 묶어 가치 헤드로 평가한다.

   입력을 만드는 encodeState 는 engine.js 안에 있다. 학습에 쓴 encoding.py 와
   규격이 같아야 하며, parity_check 로 확인할 수 있다. */

(() => {
  'use strict';

  const MODEL_URL = 'sixmok.onnx';
  const MODEL_BOARD_SIZE = 19;   // 학습할 때 쓴 판 크기
  const POLICY_BLEND = 0.1;      // 가치가 비슷할 때 정책으로 우열을 가르는 정도
  const DEEP_ROOTS = 4;          // 어려움에서 2-ply로 다시 펼칠 뿌리 후보 수
  const DEEP_WIDTH = 3;          // 2-ply의 두 번째 단계 후보 폭

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

  /** 여러 국면을 한 번에 계산한다. 갈래를 늘려도 호출 횟수를 최소화한다. */
  async function evaluate(games) {
    const n = games[0].size;
    const planeSize = N_PLANES * n * n;
    const data = new Float32Array(games.length * planeSize);
    games.forEach((g, k) => data.set(encodeState(g), k * planeSize));

    const sess = await session();
    const tensor = new ort.Tensor('float32', data, [games.length, N_PLANES, n, n]);
    const out = await sess.run({ board: tensor });
    return {
      policy: out.policy.data,          // (B, n*n+1)
      value: out.value.data,            // (B,)
      stride: n * n + 1,
    };
  }

  /** 둘 수 있는 자리만 남긴 확률 분포. */
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

  /** 지금 두면 곧바로 6목이 되는 자리. 없으면 -1.
      탐색에 맡기지 않는다. 이기는 수를 놓치는 일만은 없어야 한다. */
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

  function rootValue(game, rootColor) {
    if (!game.isOver) return null;
    if (game.winner === null) return 0;
    return game.winner === rootColor ? 1 : -1;
  }

  /** color가 현재/다음 자기 턴에 연속으로 둘 수 있는 최대 돌 수. */
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

  /** color가 다음 한 턴 안에 완성할 수 있는 실제 6칸 승리창들.
      상대 돌이 하나라도 들어간 창은 제외하고, 남은 빈칸 수가 이번 턴에 둘 수 있는
      돌 수 이하인 창만 반환한다. 돌이 띄어져 있어도 같은 6칸 창이면 정확히 잡는다. */
  function oneTurnWinningWindows(game, color) {
    if (game.isOver) return [];

    const moves = Math.min(availablePlacements(game, color), 3);
    if (moves <= 0) return [];

    const n = game.size;
    const need = game.config.winLength;
    const foe = color === BLACK ? WHITE : BLACK;
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

  /** color가 한 턴 안에 6목을 완성할 수 있는 첫 착수 후보.
      실제 승리창의 빈칸 합집합이므로, 띄어진 형태도 연속된 6칸 창 안에만 있으면 잡힌다. */
  function oneTurnWinningSquares(game, color) {
    const out = new Set();
    for (const window of oneTurnWinningWindows(game, color)) {
      for (const i of window) out.add(i);
    }
    return [...out];
  }

  /** 상대의 모든 한 턴 승리창을 현재 남은 착수 수 안에 실제로 막을 수 있는 첫 수만 반환한다.
      단순히 위협 칸을 합치는 대신 각 승리창을 하나 이상 때리는 최소 차단 집합을 찾는다.
      최대 3수까지만 보므로 재귀 분기는 작다. */
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

  /** policy 상위 후보에 놓치면 안 되는 전술 후보를 합친다.
      1) 내가 이번 턴 안에 이길 수 있으면 그 승리 첫 수들만
      2) 상대가 다음 자기 턴 안에 이길 수 있으면 모든 승리창을 실제로 막을 수 있는 수들만
      3) 그 외에는 fork 공격/방어 + policy
      4) 평온한 국면에서만 END_TURN 허용 */
  function candidateActions(game, probs, width) {
    const n = game.size;
    const endIndex = n * n;

    if (!game.canPlace()) return [endIndex];

    const empties = [];
    for (let i = 0; i < endIndex; i++) {
      if (game.board[i] === 0) empties.push(i);
    }
    empties.sort((a, b) => probs[b] - probs[a]);

    // 내가 같은 턴 안에 2~3수를 이어서 끝낼 수 있다면 방어보다 먼저 끝낸다.
    const ownWinning = oneTurnWinningSquares(game, game.current);
    if (ownWinning.length) {
      return ownWinning.sort((a, b) => probs[b] - probs[a]);
    }

    const foe = game.current === BLACK ? WHITE : BLACK;

    // 상대가 다음 자기 턴 안에 끝낼 수 있으면, 현재 남은 착수 수 안에
    // 모든 승리창을 차단할 수 있는 첫 수만 강제 방어 후보로 남긴다.
    const foeWinning = oneTurnWinningSquares(game, foe);
    if (foeWinning.length) {
      const defenses = oneTurnDefenseSquares(game, foe);
      // 이미 필패라 모든 창을 막는 조합이 없다면 기존처럼 위협 칸 안에서 최선을 찾는다.
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

  /** 온도에 따라 하나를 고른다. 0이면 언제나 최선. */
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

  /** 후보마다 한 수 놓아보고 가치로 고른다.
      어려움에서는 1-ply 평가 상위 DEEP_ROOTS개만 다시 DEEP_WIDTH갈래로 펼친다.
      자식에서도 같은 사람이 계속 두는 경우에는 최대화하고, 턴이 넘어가 상대가 두는
      경우에는 최소화한다. 이 게임의 한 턴 2~3연속 착수도 자연스럽게 읽힌다. */
  async function pickWithLookahead(game, probs, width, evaluateFn, deep = false) {
    const n = game.size;
    const rootColor = game.current;
    const actions = candidateActions(game, probs, width);

    const children = actions.map((a) => {
      const child = game.clone();
      child.play(toMove(a, n));
      return child;
    });

    const scores = new Array(actions.length);
    const childPolicies = new Array(actions.length);
    const pending = [], pendingAt = [];
    children.forEach((child, k) => {
      const terminal = rootValue(child, rootColor);
      if (terminal !== null) {
        scores[k] = terminal;
      } else {
        pending.push(child);
        pendingAt.push(k);
      }
    });

    if (pending.length) {
      const { policy, value, stride } = await evaluateFn(pending);
      pendingAt.forEach((k, j) => {
        const child = children[k];
        const v = value[j];
        scores[k] = child.current === rootColor ? v : -v;
        childPolicies[k] = legalProbs(policy, child, j * stride, stride);
      });
    }

    if (deep) {
      const expandable = actions
        .map((action, k) => ({ action, k, score: scores[k] + POLICY_BLEND * probs[action] }))
        .filter(({ k }) => !children[k].isOver && childPolicies[k])
        .sort((a, b) => b.score - a.score)
        .slice(0, DEEP_ROOTS);

      const leaves = [];
      const leafMeta = [];

      for (const { k } of expandable) {
        const child = children[k];
        const replyProbs = childPolicies[k];
        const replies = candidateActions(child, replyProbs, DEEP_WIDTH);
        for (const reply of replies) {
          const leaf = child.clone();
          leaf.play(toMove(reply, n));
          leaves.push(leaf);
          leafMeta.push({ parent: k, reply });
        }
      }

      const leafScores = new Array(leaves.length);
      const leafPending = [], leafPendingAt = [];
      leaves.forEach((leaf, j) => {
        const terminal = rootValue(leaf, rootColor);
        if (terminal !== null) leafScores[j] = terminal;
        else {
          leafPending.push(leaf);
          leafPendingAt.push(j);
        }
      });

      if (leafPending.length) {
        const { value } = await evaluateFn(leafPending);
        leafPendingAt.forEach((j, p) => {
          const leaf = leaves[j];
          leafScores[j] = leaf.current === rootColor ? value[p] : -value[p];
        });
      }

      for (const { k } of expandable) {
        const child = children[k];
        const candidates = [];
        for (let j = 0; j < leafMeta.length; j++) {
          if (leafMeta[j].parent !== k) continue;
          const reply = leafMeta[j].reply;
          const replyPrior = childPolicies[k][reply] || 0;
          candidates.push(
            leafScores[j]
            + (child.current === rootColor ? 1 : -1) * POLICY_BLEND * replyPrior
          );
        }
        if (candidates.length) {
          scores[k] = child.current === rootColor
            ? Math.max(...candidates)
            : Math.min(...candidates);
        }
      }
    }

    let best = 0;
    for (let k = 1; k < actions.length; k++) {
      const a = scores[k] + POLICY_BLEND * probs[actions[k]];
      const b = scores[best] + POLICY_BLEND * probs[actions[best]];
      if (a > b) best = k;
    }
    return actions[best];
  }

  /** 한 턴 분량의 착수 목록. 빈 배열이면 이번 턴은 쉬고 돌을 모은다는 뜻. */
  async function planTurnWithModel(game, level, evaluateFn) {
    const n = game.size;
    const endIndex = n * n;
    const temperature = level.temperature || 0;
    const width = level.lookahead || 0;
    const deep = width >= 6;                    // 현재 난이도 설정에서 '어려움'만 2-ply

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

      if (index === endIndex) break;              // 턴을 마치기로 했다
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
    };
  }
})();
