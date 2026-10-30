(() => {
  'use strict';

  const MODEL_BOARD_SIZE = 19;
  const MAX_EMPTY_TURNS = 16;
  const MAX_TURNS = 512;
  const LEVELS = {
    easy: { label: '쉬움', temperature: 0.7, lookahead: 3 },
    normal: { label: '보통', temperature: 0.5, lookahead: 3 },
    hard: { label: '어려움', temperature: 0.3, lookahead: 3 },
  };

  const core = window.SIXMOK_MODEL_CORE;
  if (!core || typeof core.planTurnWithModel !== 'function' || typeof core.encodeStateForModel !== 'function') {
    throw new Error('현재 웹 모델 코어를 불러오지 못했습니다. model.js 로드를 확인하세요.');
  }

  const $ = (id) => document.getElementById(id);
  const colorName = { [BLACK]: '흑', [WHITE]: '백' };

  class ArenaModel {
    constructor(name, session, fileName) {
      this.name = name;
      this.session = session;
      this.fileName = fileName;
      this.inputPlanes = null;
      this.inferenceMs = 0;
      this.inferenceCalls = 0;
      this.positions = 0;
    }

    resetTiming() {
      this.inferenceMs = 0;
      this.inferenceCalls = 0;
      this.positions = 0;
    }

    async evaluate(games) {
      if (!games.length) return { policy: new Float32Array(), value: new Float32Array(), stride: 0 };
      const n = games[0].size;

      const runWithPlanes = async (planes) => {
        const planeSize = planes * n * n;
        const data = new Float32Array(games.length * planeSize);
        games.forEach((g, k) => data.set(core.encodeStateForModel(g, planes), k * planeSize));
        const tensor = new ort.Tensor('float32', data, [games.length, planes, n, n]);
        const t0 = performance.now();
        const out = await this.session.run({ board: tensor });
        this.inferenceMs += performance.now() - t0;
        this.inferenceCalls += 1;
        this.positions += games.length;
        return out;
      };

      let out;
      if (this.inputPlanes !== null) {
        out = await runWithPlanes(this.inputPlanes);
      } else {
        let err13 = null;
        try {
          out = await runWithPlanes(13);
          this.inputPlanes = 13;
        } catch (err) {
          err13 = err;
          try {
            out = await runWithPlanes(16);
            this.inputPlanes = 16;
          } catch (err16) {
            throw new Error(`${this.name}: ONNX 입력 규격 확인 실패 (13-plane: ${err13.message}; 16-plane: ${err16.message})`);
          }
        }
      }

      if (!out.policy || !out.value) throw new Error(`${this.name}: ONNX 출력에 policy/value가 없습니다.`);
      return { policy: out.policy.data, value: out.value.data, stride: n * n + 1 };
    }
  }

  function mulberry32(seed) {
    let a = seed >>> 0;
    return function rng() {
      a |= 0;
      a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function hashSeed(base, salt) {
    let x = (base >>> 0) ^ Math.imul((salt + 1) >>> 0, 0x9E3779B1);
    x ^= x >>> 16;
    x = Math.imul(x, 0x85EBCA6B);
    x ^= x >>> 13;
    x = Math.imul(x, 0xC2B2AE35);
    return (x ^ (x >>> 16)) >>> 0;
  }

  function escapeHtml(text) {
    return String(text).replace(/[&<>'\"]/g, (ch) => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;',
    })[ch]);
  }

  let modelA = null;
  let modelB = null;
  let running = false;
  let stopRequested = false;
  let lastResults = null;

  async function loadModel(which) {
    const input = $(`model-${which}-file`);
    const status = $(`model-${which}-status`);
    const file = input.files && input.files[0];
    if (!file) {
      status.textContent = 'ONNX 파일을 선택하세요.';
      status.className = 'status error';
      return null;
    }

    status.textContent = '불러오는 중…';
    status.className = 'status';
    try {
      const bytes = new Uint8Array(await file.arrayBuffer());
      const session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] });
      const model = new ArenaModel(`모델 ${which.toUpperCase()}`, session, file.name);
      await model.evaluate([new SixMok({ boardSize: MODEL_BOARD_SIZE })]);
      model.resetTiming();
      status.textContent = `${file.name} · ${model.inputPlanes}평면 · 준비됨`;
      status.className = 'status ok';
      return model;
    } catch (err) {
      status.textContent = err.message;
      status.className = 'status error';
      return null;
    }
  }

  function updateControls() {
    $('start').disabled = running;
    $('stop').disabled = !running;
    for (const id of ['model-a-file', 'model-b-file', 'level-a', 'level-b', 'games', 'seed']) {
      $(id).disabled = running;
    }
  }

  function clearResults() {
    $('summary').innerHTML = '<div class="empty">대국을 시작하면 결과가 표시됩니다.</div>';
    $('games-body').innerHTML = '';
    $('progress-bar').style.width = '0%';
    $('progress-text').textContent = '대기 중';
    $('export').disabled = true;
    lastResults = null;
  }

  function outcomeLabel(result) {
    if (result.reason === 'empty_turn_limit') return '무착수 중단';
    if (result.reason === 'turn_limit') return '턴 한도';
    if (result.reason === 'error') return '오류';
    if (result.winnerModel === 'A') return 'A 승';
    if (result.winnerModel === 'B') return 'B 승';
    return '무승부';
  }

  function appendGameRow(result) {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${result.index}</td>
      <td>${escapeHtml(result.blackModel)} (${escapeHtml(result.blackLevel)})</td>
      <td>${escapeHtml(result.whiteModel)} (${escapeHtml(result.whiteLevel)})</td>
      <td class="${result.winnerModel === 'A' ? 'a-win' : result.winnerModel === 'B' ? 'b-win' : ''}">${outcomeLabel(result)}</td>
      <td>${result.turns}</td><td>${result.stones}</td><td>${result.emptyTurns}</td>
      <td>${result.elapsedMs.toFixed(0)} ms</td>`;
    const body = $('games-body');
    body.prepend(tr);
    while (body.children.length > 50) body.removeChild(body.lastChild);
  }

  function renderSummary(results, planned) {
    const done = results.games.length;
    const aWins = results.games.filter((g) => g.winnerModel === 'A').length;
    const bWins = results.games.filter((g) => g.winnerModel === 'B').length;
    const draws = results.games.filter((g) => !g.winnerModel && g.reason === 'draw').length;
    const abnormal = results.games.filter((g) => ['empty_turn_limit', 'turn_limit', 'error'].includes(g.reason)).length;
    const decisive = aWins + bWins;
    const aRate = decisive ? 100 * aWins / decisive : 0;
    const bRate = decisive ? 100 * bWins / decisive : 0;
    const avgTurns = done ? results.games.reduce((s, g) => s + g.turns, 0) / done : 0;
    const avgStones = done ? results.games.reduce((s, g) => s + g.stones, 0) / done : 0;
    const aBlack = results.games.filter((g) => g.blackModel === 'A');
    const bBlack = results.games.filter((g) => g.blackModel === 'B');
    const aBlackWins = aBlack.filter((g) => g.winnerModel === 'A').length;
    const aWhiteWins = bBlack.filter((g) => g.winnerModel === 'A').length;
    const bBlackWins = bBlack.filter((g) => g.winnerModel === 'B').length;
    const bWhiteWins = aBlack.filter((g) => g.winnerModel === 'B').length;
    const aMs = modelA && modelA.positions ? modelA.inferenceMs / modelA.positions : 0;
    const bMs = modelB && modelB.positions ? modelB.inferenceMs / modelB.positions : 0;

    $('summary').innerHTML = `
      <div class="score a"><span>A</span><strong>${aWins}</strong><small>${aRate.toFixed(1)}% of decisive</small></div>
      <div class="score draw"><span>무승부/비정상</span><strong>${draws} / ${abnormal}</strong><small>${done} / ${planned} 완료</small></div>
      <div class="score b"><span>B</span><strong>${bWins}</strong><small>${bRate.toFixed(1)}% of decisive</small></div>
      <div class="metric"><span>A 색별 승</span><strong>흑 ${aBlackWins} · 백 ${aWhiteWins}</strong></div>
      <div class="metric"><span>B 색별 승</span><strong>흑 ${bBlackWins} · 백 ${bWhiteWins}</strong></div>
      <div class="metric"><span>평균 길이</span><strong>${avgTurns.toFixed(1)}턴 · ${avgStones.toFixed(1)}돌</strong></div>
      <div class="metric"><span>A 추론</span><strong>${aMs.toFixed(2)} ms/position</strong></div>
      <div class="metric"><span>B 추론</span><strong>${bMs.toFixed(2)} ms/position</strong></div>`;
  }

  async function playOneGame(index, config) {
    const even = (index % 2) === 0;
    const blackModel = even ? modelA : modelB;
    const whiteModel = even ? modelB : modelA;
    const blackId = even ? 'A' : 'B';
    const whiteId = even ? 'B' : 'A';
    const blackLevel = even ? config.levelA : config.levelB;
    const whiteLevel = even ? config.levelB : config.levelA;
    const pairSeed = hashSeed(config.seed, Math.floor(index / 2));
    const rngA = mulberry32(hashSeed(pairSeed, 1001));
    const rngB = mulberry32(hashSeed(pairSeed, 2002));
    const game = new SixMok({ boardSize: MODEL_BOARD_SIZE });
    let turns = 0;
    let emptyStreak = 0;
    let emptyTurns = 0;
    const started = performance.now();
    let reason = 'draw';
    let errorMessage = null;

    try {
      while (!game.isOver && turns < MAX_TURNS && emptyStreak < MAX_EMPTY_TURNS) {
        if (stopRequested) break;
        const blackToMove = game.current === BLACK;
        const model = blackToMove ? blackModel : whiteModel;
        const modelId = blackToMove ? blackId : whiteId;
        const level = blackToMove ? blackLevel : whiteLevel;
        const rng = modelId === 'A' ? rngA : rngB;
        const evaluate = model.evaluate.bind(model);
        const moves = await core.planTurnWithModel(game, level, evaluate, { rng });

        let placed = 0;
        for (const [r, c] of moves) {
          if (!game.canPlace()) break;
          game.place(r, c);
          placed += 1;
          if (game.isOver) break;
        }
        if (placed === 0) { emptyTurns += 1; emptyStreak += 1; }
        else emptyStreak = 0;
        turns += 1;
        if (!game.isOver) game.endTurn();
      }

      if (game.isOver) reason = game.winner ? 'win' : 'draw';
      else if (emptyStreak >= MAX_EMPTY_TURNS) reason = 'empty_turn_limit';
      else if (turns >= MAX_TURNS) reason = 'turn_limit';
      else if (stopRequested) reason = 'stopped';
    } catch (err) {
      reason = 'error';
      errorMessage = err.message;
    }

    let winnerModel = null;
    if (game.winner === BLACK) winnerModel = blackId;
    else if (game.winner === WHITE) winnerModel = whiteId;

    return {
      index: index + 1,
      pair: Math.floor(index / 2) + 1,
      seed: pairSeed,
      blackModel: blackId,
      whiteModel: whiteId,
      blackFile: blackModel.fileName,
      whiteFile: whiteModel.fileName,
      blackLevel: blackLevel.label,
      whiteLevel: whiteLevel.label,
      winnerColor: game.winner ? colorName[game.winner] : null,
      winnerModel,
      reason,
      errorMessage,
      turns,
      stones: game.stonesOnBoard,
      emptyTurns,
      elapsedMs: performance.now() - started,
    };
  }

  async function startArena() {
    if (running) return;
    if (!$('model-a-file').files[0] || !$('model-b-file').files[0]) {
      $('progress-text').textContent = '모델 A와 B의 ONNX 파일을 모두 선택하세요.';
      return;
    }

    const games = Math.max(1, Math.min(1000, parseInt($('games').value, 10) || 20));
    const seed = (parseInt($('seed').value, 10) || 1) >>> 0;
    const levelA = LEVELS[$('level-a').value] || LEVELS.normal;
    const levelB = LEVELS[$('level-b').value] || LEVELS.normal;

    running = true;
    stopRequested = false;
    updateControls();
    clearResults();
    $('progress-text').textContent = '모델 불러오는 중…';

    modelA = await loadModel('a');
    if (!modelA || stopRequested) {
      running = false; updateControls();
      $('progress-text').textContent = stopRequested ? '중단됨' : '모델 A 로드 실패';
      return;
    }
    modelB = await loadModel('b');
    if (!modelB || stopRequested) {
      running = false; updateControls();
      $('progress-text').textContent = stopRequested ? '중단됨' : '모델 B 로드 실패';
      return;
    }

    modelA.resetTiming();
    modelB.resetTiming();
    const results = {
      createdAt: new Date().toISOString(),
      settings: {
        games, seed,
        modelA: { file: modelA.fileName, difficulty: levelA.label, temperature: levelA.temperature },
        modelB: { file: modelB.fileName, difficulty: levelB.label, temperature: levelB.temperature },
        environment: 'current-web-model.js',
        maxEmptyTurns: MAX_EMPTY_TURNS,
        maxTurns: MAX_TURNS,
      },
      games: [],
    };
    lastResults = results;
    $('export').disabled = false;

    for (let i = 0; i < games; i++) {
      if (stopRequested) break;
      $('progress-text').textContent = `${i + 1}/${games} 대국 중 · ${i % 2 === 0 ? 'A 흑 / B 백' : 'B 흑 / A 백'}`;
      const result = await playOneGame(i, { seed, levelA, levelB });
      if (result.reason === 'stopped') break;
      results.games.push(result);
      appendGameRow(result);
      renderSummary(results, games);
      $('progress-bar').style.width = `${100 * results.games.length / games}%`;
      await new Promise((resolve) => setTimeout(resolve, 0));
    }

    running = false;
    updateControls();
    const done = results.games.length;
    $('progress-text').textContent = stopRequested ? `중단됨 · ${done}/${games}판 완료` : `완료 · ${done}/${games}판`;
  }

  function exportResults() {
    if (!lastResults) return;
    const blob = new Blob([JSON.stringify(lastResults, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `sixmok-arena-${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  $('start').addEventListener('click', startArena);
  $('stop').addEventListener('click', () => {
    stopRequested = true;
    $('progress-text').textContent = '현재 대국 종료 후 중단…';
  });
  $('export').addEventListener('click', exportResults);
  $('clear').addEventListener('click', clearResults);

  clearResults();
  updateControls();
})();
