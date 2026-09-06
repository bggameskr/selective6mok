/* 학습한 모델을 브라우저 게임에 붙인다.

   쓰는 법
   1. 코랩 노트북에서 받은 sixmok.onnx 를 index.html 과 같은 폴더에 둔다.
   2. 그 폴더를 웹에 올리거나 간단한 서버로 연다 (file:// 로 열면 모델을 못 읽는다).

   난이도에 따라 한 수 앞을 내다본다.
     쉬움   0갈래 — 신경망이 낸 수를 그대로 (온도만 높여 흔든다)
     보통   3갈래
     어려움 6갈래
   후보마다 실제로 돌을 놓아본 뒤 가치 헤드로 평가한다. 한 번에 묶어 계산하므로
   갈래를 늘려도 신경망 호출은 한 번만 늘어난다.

   입력을 만드는 encodeState 는 engine.js 안에 있다. 학습에 쓴 encoding.py 와
   규격이 같아야 하며, parity_check 로 확인할 수 있다. */

(() => {
  'use strict';

  const MODEL_URL = 'sixmok.onnx';
  const MODEL_BOARD_SIZE = 19;   // 학습할 때 쓴 판 크기
  const POLICY_BLEND = 0.1;      // 가치가 비슷할 때 정책으로 우열을 가르는 정도

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

  /** 여러 국면을 한 번에 계산한다. 갈래를 늘려도 호출 횟수는 그대로다. */
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
      돌을 놓아도 차례가 넘어가지 않으므로, 자식 국면의 가치는 대개 내 관점 그대로다.
      턴을 마치는 갈래만 상대 관점이라 부호를 뒤집는다. */
  async function pickWithLookahead(game, probs, width, evaluateFn) {
    const n = game.size;
    const endIndex = n * n;

    const empties = [];
    for (let i = 0; i < endIndex; i++) if (game.board[i] === 0) empties.push(i);
    empties.sort((a, b) => probs[b] - probs[a]);
    const actions = empties.slice(0, width);
    actions.push(endIndex);                       // 턴 마치기도 후보에 넣는다

    const children = actions.map((a) => {
      const child = game.clone();
      child.play(toMove(a, n));
      return child;
    });

    const scores = new Array(actions.length);
    const pending = [], pendingAt = [];
    children.forEach((child, k) => {
      if (child.isOver) {
        scores[k] = child.winner === null ? 0 : (child.winner === game.current ? 1 : -1);
      } else {
        pending.push(child);
        pendingAt.push(k);
      }
    });

    if (pending.length) {
      const { value } = await evaluateFn(pending);
      pendingAt.forEach((k, j) => {
        const v = value[j];
        scores[k] = pending[j].current === game.current ? v : -v;
      });
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
        ? await pickWithLookahead(sim, probs, width, evaluateFn)
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
    module.exports = { planTurnWithModel, pickWithLookahead, legalProbs, sample, findImmediateWin };
  }
})();
