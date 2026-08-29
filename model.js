/* 학습한 모델을 브라우저 게임에 붙인다.

   쓰는 법
   1. 코랩 노트북에서 받은 sixmok.onnx 를 index.html 과 같은 폴더에 둔다.
   2. index.html 아래쪽의 주석 처리된 <script> 두 줄을 살린다.
   3. 로컬 파일을 그대로 열면 브라우저가 모델 파일을 막으므로 간단한 서버를 띄운다.
        python -m http.server 8000
      그리고 http://localhost:8000 으로 접속한다.

   입력을 만드는 encodeState 는 engine.js 안에 있다. 학습에 쓴 encoding.py 와
   규격이 같아야 하며, parity_check 로 확인할 수 있다. */

(() => {
  'use strict';

  const MODEL_URL = 'sixmok.onnx';
  const MODEL_BOARD_SIZE = 19;   // 학습할 때 쓴 판 크기

  let sessionPromise = null;

  /* 모델을 쓰고 있는지 화면에 표시할 수 있게 상태를 알린다.
     'loading' -> 'ready' 또는 'failed' */
  function announce(status, detail) {
    window.MODEL_STATUS = status;
    window.MODEL_STATUS_DETAIL = detail || '';
    window.dispatchEvent(new CustomEvent('model-status'));
  }

  announce('loading');

  function session() {
    if (!sessionPromise) {
      sessionPromise = ort.InferenceSession.create(MODEL_URL, {
        executionProviders: ['wasm'],
      });
    }
    return sessionPromise;
  }

  async function policyFor(game) {
    const n = game.size;
    const sess = await session();
    const tensor = new ort.Tensor('float32', encodeState(game), [1, N_PLANES, n, n]);
    const out = await sess.run({ board: tensor });
    return out.policy.data;   // 길이 n*n+1 로짓
  }

  /** 둘 수 있는 자리 중에서 하나를 고른다. temperature가 0이면 언제나 최선의 수. */
  function pickAction(logits, sim, endIndex, temperature) {
    const indices = [endIndex];
    const values = [logits[endIndex]];
    for (let i = 0; i < endIndex; i++) {
      if (sim.board[i] !== 0) continue;
      indices.push(i);
      values.push(logits[i]);
    }

    if (!(temperature > 0)) {
      let best = 0;
      for (let k = 1; k < values.length; k++) if (values[k] > values[best]) best = k;
      return indices[best];
    }

    const max = Math.max(...values);
    const weights = values.map((v) => Math.exp((v - max) / temperature));
    const total = weights.reduce((a, b) => a + b, 0);
    let roll = Math.random() * total;
    for (let k = 0; k < weights.length; k++) {
      roll -= weights[k];
      if (roll <= 0) return indices[k];
    }
    return indices[indices.length - 1];
  }

  /** 한 턴 분량의 착수 목록. 빈 배열이면 이번 턴은 쉬고 돌을 모은다는 뜻. */
  window.MODEL_PLAN_TURN = async (game) => {
    if (game.size !== MODEL_BOARD_SIZE) {
      const msg = `모델은 ${MODEL_BOARD_SIZE}줄 판으로만 학습돼 있습니다.`;
      announce('failed', msg);
      throw new Error(msg);
    }

    const level = (typeof window.AI_LEVEL === 'function' ? window.AI_LEVEL() : null) || {};
    const temperature = level.temperature || 0;

    const sim = game.clone();
    const n = sim.size;
    const endIndex = n * n;
    const moves = [];

    while (sim.canPlace()) {
      const logits = await policyFor(sim);
      const index = pickAction(logits, sim, endIndex, temperature);

      if (index === endIndex) break;   // 모델이 턴을 마치기로 했다
      const r = Math.floor(index / n);
      const c = index % n;
      sim.place(r, c);
      moves.push([r, c]);
      if (sim.isOver) break;
    }

    return moves;
  };

  // 모델 파일을 미리 받아둔다. 실패하면 index.html 이 규칙 기반 AI로 되돌린다.
  session()
    .then(() => {
      announce('ready');
      console.log(`모델을 불러왔습니다: ${MODEL_URL}`);
    })
    .catch((err) => {
      announce('failed', err.message);
      console.warn('모델을 불러오지 못했습니다. 규칙 기반 AI로 둡니다.', err);
    });
})();
