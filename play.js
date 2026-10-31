(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const mode = document.body.dataset.mode || 'ai';
  const canvas = $('board');
  const ctx = canvas.getContext('2d');
  const colorName = { [BLACK]: '흑', [WHITE]: '백' };
  const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const finePointer = matchMedia('(pointer:fine)').matches;

  let game = null;
  let humanColor = BLACK;
  let busy = false;
  let turnStart = null;
  let hover = null;
  let cursor = null;
  let logs = [];
  let resultShown = false;
  let geom = { pad: 0, cell: 0, px: 0 };
  let recommendation = null;
  let recommendationText = '';
  let recommendationSeq = 0;

  const isAiMode = mode === 'ai';
  const isSelfMode = mode === 'self';
  const isPvpMode = mode === 'pvp';

  function level() {
    const key = $('level')?.value || 'normal';
    return (typeof AI_LEVELS !== 'undefined' && AI_LEVELS[key]) ? AI_LEVELS[key] : {};
  }

  window.AI_LEVEL = level;

  function css(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function xy(r, c) {
    return [geom.pad + c * geom.cell, geom.pad + r * geom.cell];
  }

  function stoneRadius() {
    return geom.cell * 0.44;
  }

  function roundRect(x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  function drawStone(r, c, color, alpha = 1) {
    const [x, y] = xy(r, c);
    const rad = stoneRadius();
    ctx.save();
    ctx.globalAlpha = alpha;
    const grad = ctx.createRadialGradient(x - rad * .3, y - rad * .35, rad * .1, x, y, rad);
    if (color === BLACK) {
      grad.addColorStop(0, '#4a5350');
      grad.addColorStop(1, '#0d1110');
    } else {
      grad.addColorStop(0, '#fff');
      grad.addColorStop(1, '#bcb4a3');
    }
    ctx.beginPath();
    ctx.arc(x, y, rad, 0, Math.PI * 2);
    ctx.fillStyle = grad;
    ctx.fill();
    ctx.restore();
  }

  function draw() {
    if (!game || !geom.px) return;
    const n = game.size;
    const size = geom.px;
    ctx.clearRect(0, 0, size, size);
    ctx.fillStyle = css('--field');
    roundRect(0, 0, size, size, 10);
    ctx.fill();

    ctx.strokeStyle = css('--line');
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const [x0, y] = xy(i, 0);
      const [x1] = xy(i, n - 1);
      const [x, y0] = xy(0, i);
      const [, y1] = xy(n - 1, i);
      ctx.moveTo(x0, y); ctx.lineTo(x1, y);
      ctx.moveTo(x, y0); ctx.lineTo(x, y1);
    }
    ctx.stroke();

    for (let r = 0; r < n; r++) {
      for (let c = 0; c < n; c++) {
        const v = game.at(r, c);
        if (v !== EMPTY) drawStone(r, c, v);
      }
    }

    ctx.strokeStyle = css('--accent');
    ctx.lineWidth = 1.7;
    for (const [r, c] of game.lastPlaced) {
      const [x, y] = xy(r, c);
      ctx.beginPath();
      ctx.arc(x, y, stoneRadius() * .42, 0, Math.PI * 2);
      ctx.stroke();
    }

    if (game.winningLine) {
      ctx.lineWidth = 2.4;
      for (const [r, c] of game.winningLine) {
        const [x, y] = xy(r, c);
        ctx.beginPath();
        ctx.arc(x, y, stoneRadius() * 1.12, 0, Math.PI * 2);
        ctx.stroke();
      }
    }

    if (isSelfMode && recommendation && recommendation !== END_TURN && !game.isOver) {
      const [x, y] = xy(recommendation[0], recommendation[1]);
      ctx.save();
      ctx.strokeStyle = css('--brass');
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.arc(x, y, stoneRadius() * 1.22, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
    }

    const preview = hover || cursor;
    if (preview && canUserAct() && game.at(preview[0], preview[1]) === EMPTY && game.canPlace()) {
      drawStone(preview[0], preview[1], game.current, .4);
    }
    if (cursor) {
      const [x, y] = xy(cursor[0], cursor[1]);
      ctx.strokeStyle = css('--accent');
      ctx.strokeRect(x - geom.cell * .5, y - geom.cell * .5, geom.cell, geom.cell);
    }
  }

  function resize() {
    if (!game) return;
    const wrap = canvas.parentElement;
    const style = getComputedStyle(wrap);
    const availW = wrap.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight);
    const availH = wrap.clientHeight - parseFloat(style.paddingTop) - parseFloat(style.paddingBottom);
    const minSize = matchMedia('(max-width:820px) and (orientation:portrait)').matches ? 160 : 200;
    const size = Math.max(minSize, Math.floor(Math.min(availW, availH)));
    const dpr = devicePixelRatio || 1;
    canvas.style.width = size + 'px';
    canvas.style.height = size + 'px';
    canvas.width = Math.round(size * dpr);
    canvas.height = Math.round(size * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    geom.pad = Math.max(14, size * .035);
    geom.cell = (size - geom.pad * 2) / (game.size - 1);
    geom.px = size;
    draw();
  }

  function canUserAct() {
    if (busy || !game || game.isOver) return false;
    if (isAiMode) return game.current === humanColor;
    return true;
  }

  function renderBank(node, color) {
    if (!node) return;
    node.textContent = '';
    const who = document.createElement('span');
    who.className = 'who';
    who.textContent = colorName[color];
    node.appendChild(who);
    for (let i = 0; i < Math.min(game.stones[color], 8); i++) {
      const token = document.createElement('span');
      token.className = 'token ' + (color === BLACK ? 'b' : 'w');
      node.appendChild(token);
    }
    const count = document.createElement('span');
    count.className = 'count';
    count.textContent = game.stones[color] + '개';
    node.appendChild(count);
  }

  function updateBadge() {
    const badge = $('ai-badge');
    if (!badge) return;
    if (window.MODEL_STATUS === 'ready') {
      badge.textContent = '학습 모델';
      badge.className = 'badge ready';
    } else if (window.MODEL_STATUS === 'failed') {
      badge.textContent = '모델 실패';
      badge.className = 'badge error';
      badge.title = window.MODEL_STATUS_DETAIL || '';
    } else {
      badge.textContent = '모델 불러오는 중';
      badge.className = 'badge';
    }
  }

  function turnText() {
    if (game.isOver) return '대국 종료';
    if (isAiMode) return `${colorName[game.current]} 차례 — ${game.current === humanColor ? '내 턴' : 'AI 턴'}`;
    if (isPvpMode) return `${colorName[game.current]} 차례 — 플레이어 ${game.current === BLACK ? '1' : '2'}`;
    return `${colorName[game.current]} 차례`;
  }

  function render() {
    draw();
    $('turn-dot').style.background = game.current === BLACK
      ? 'radial-gradient(circle at 34% 30%,#4a5350,#0d1110 70%)'
      : 'radial-gradient(circle at 34% 30%,#fff,#bdb5a4 72%)';
    $('turn-text').textContent = turnText();
    renderBank($('bank-black'), BLACK);
    renderBank($('bank-white'), WHITE);

    const slots = $('slots');
    slots.textContent = '';
    for (let i = 0; i < game.config.maxPlacesPerTurn; i++) {
      const slot = document.createElement('span');
      slot.className = 'slot';
      if (i < game.placedThisTurn) slot.classList.add('on');
      else if (i >= game.placedThisTurn + game.stones[game.current]) slot.classList.add('lock');
      slots.appendChild(slot);
    }

    $('end-turn').disabled = !canUserAct();
    $('undo').disabled = !canUserAct() || !turnStart || game.placedThisTurn === 0;

    const recommend = $('recommend');
    if (recommend) {
      recommend.hidden = !isSelfMode;
      if (isSelfMode) recommend.innerHTML = recommendationText || '<strong>Greedy</strong> 모델 계산 중…';
    }

    $('log').innerHTML = logs.slice(-45).join('<br>');
    $('log').scrollTop = $('log').scrollHeight;
    updateBadge();
    if (game.isOver) showResult();
  }

  function setHint() {
    if (game.isOver) return $('hint').textContent = '';
    if (!canUserAct()) return $('hint').textContent = 'AI가 생각하는 중입니다.';
    if (!game.canPlace()) return $('hint').textContent = '더 둘 수 없습니다. 턴을 마치세요.';
    if (isSelfMode) return $('hint').textContent = '흑과 백을 모두 직접 둡니다. 점선 원은 난수 없는 greedy 추천수입니다.';
    if (isPvpMode) return $('hint').textContent = '현재 차례의 플레이어가 돌을 두거나 턴을 마칩니다.';
    $('hint').textContent = '이번 턴의 돌을 배치하거나 턴을 마치세요.';
  }

  async function refreshRecommendation() {
    recommendation = null;
    recommendationText = '';
    const seq = ++recommendationSeq;
    if (!isSelfMode || !game || game.isOver) {
      render();
      return;
    }
    if (window.MODEL_STATUS !== 'ready' || typeof window.MODEL_PLAN_TURN !== 'function') {
      recommendationText = '<strong>Greedy</strong> 모델 준비 중…';
      render();
      return;
    }

    const stamp = `${game.current}|${game.placedThisTurn}|${game.stonesOnBoard}|${game.history.length}`;
    recommendationText = '<strong>Greedy</strong> 계산 중…';
    render();
    try {
      const moves = await window.MODEL_PLAN_TURN(game.clone());
      if (seq !== recommendationSeq || !isSelfMode) return;
      const now = `${game.current}|${game.placedThisTurn}|${game.stonesOnBoard}|${game.history.length}`;
      if (now !== stamp) return refreshRecommendation();
      if (moves && moves.length) {
        recommendation = moves[0];
        recommendationText = `<strong>Greedy</strong> ${recommendation[0] + 1}·${recommendation[1] + 1}`;
      } else {
        recommendation = END_TURN;
        recommendationText = '<strong>Greedy</strong> 턴 마치기';
      }
    } catch (err) {
      recommendationText = '<strong>Greedy</strong> 계산 실패';
      console.warn(err);
    }
    render();
  }

  function showResult() {
    if (resultShown) return;
    resultShown = true;
    $('result-title').textContent = game.winner ? `${colorName[game.winner]} 승리` : '무승부';
    $('result-detail').textContent = game.winner
      ? '여섯 개를 이었습니다.'
      : `판의 70%가 채워졌습니다. (${game.stonesOnBoard}개)`;
    setTimeout(() => { $('result').hidden = false; }, reduceMotion ? 0 : 500);
  }

  function pointFromEvent(e) {
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const c = Math.round((x - geom.pad) / geom.cell);
    const r = Math.round((y - geom.pad) / geom.cell);
    if (!game.inside(r, c)) return null;
    const [px, py] = xy(r, c);
    const radius = geom.cell * (finePointer ? .55 : .72);
    return Math.hypot(px - x, py - y) <= radius ? [r, c] : null;
  }

  function isCurrentTurnStone(r, c) {
    return canUserAct() && game.lastPlaced.some(([rr, cc]) => rr === r && cc === c);
  }

  function reclaim(r, c) {
    if (!turnStart || !isCurrentTurnStone(r, c)) return;
    const keep = game.lastPlaced
      .filter(([rr, cc]) => rr !== r || cc !== c)
      .map(([rr, cc]) => [rr, cc]);
    game.restore(turnStart);
    for (const [rr, cc] of keep) game.place(rr, cc);
    setHint();
    render();
    refreshRecommendation();
  }

  function place(r, c) {
    if (!canUserAct()) return;
    if (isCurrentTurnStone(r, c)) return reclaim(r, c);
    if (!game.canPlace()) return;
    try {
      game.place(r, c);
    } catch (err) {
      $('hint').textContent = err.message;
      return;
    }
    if (game.isOver) {
      logTurn(game.current, game.lastPlaced);
      finishLog();
    }
    setHint();
    render();
    refreshRecommendation();
  }

  function logTurn(color, moves) {
    const where = moves.length ? moves.map(([r, c]) => `${r + 1}·${c + 1}`).join('  ') : '쉼';
    logs.push(`<b>${colorName[color]}</b> ${where}`);
  }

  function finishLog() {
    logs.push(game.winner ? `— ${colorName[game.winner]} 승리` : '— 무승부');
  }

  async function runAiTurn() {
    if (!isAiMode || game.isOver || game.current === humanColor) {
      afterTurn();
      return;
    }
    busy = true;
    render();
    await new Promise((resolve) => setTimeout(resolve, reduceMotion ? 0 : 220));
    let moves = [];
    try {
      moves = typeof window.MODEL_PLAN_TURN === 'function'
        ? await window.MODEL_PLAN_TURN(game)
        : planTurn(game, level());
    } catch (err) {
      console.warn(err);
      moves = planTurn(game, level());
    }

    const color = game.current;
    const placed = [];
    for (const [r, c] of moves) {
      if (!game.canPlace()) break;
      game.place(r, c);
      placed.push([r, c]);
      render();
      if (game.isOver) break;
      await new Promise((resolve) => setTimeout(resolve, reduceMotion ? 0 : 140));
    }
    logTurn(color, placed);
    if (!game.isOver) game.endTurn();
    else finishLog();
    busy = false;
    afterTurn();
  }

  function afterTurn() {
    turnStart = game.isOver ? null : game.snapshot();
    setHint();
    render();
    refreshRecommendation();
  }

  function endTurn() {
    if (!canUserAct()) return;
    const color = game.current;
    logTurn(color, game.lastPlaced);
    game.endTurn();
    turnStart = game.snapshot();
    setHint();
    render();
    if (isAiMode) runAiTurn();
    else refreshRecommendation();
  }

  function newGame() {
    humanColor = parseInt($('side')?.value || BLACK, 10);
    game = new SixMok({ boardSize: 19 });
    busy = false;
    turnStart = game.snapshot();
    hover = null;
    cursor = null;
    logs = [];
    resultShown = false;
    recommendation = null;
    recommendationText = '';
    recommendationSeq++;
    $('result').hidden = true;
    resize();
    setHint();
    render();
    if (isAiMode && game.current !== humanColor) runAiTurn();
    else refreshRecommendation();
  }

  canvas.addEventListener('click', (e) => {
    const p = pointFromEvent(e);
    if (p) place(p[0], p[1]);
  });

  if (finePointer) {
    canvas.addEventListener('mousemove', (e) => { hover = pointFromEvent(e); draw(); });
    canvas.addEventListener('mouseleave', () => { hover = null; draw(); });
  }

  canvas.addEventListener('keydown', (e) => {
    const n = game.size;
    if (!cursor) cursor = [Math.floor(n / 2), Math.floor(n / 2)];
    const delta = {
      ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1],
    }[e.key];
    if (delta) {
      cursor = [
        Math.max(0, Math.min(n - 1, cursor[0] + delta[0])),
        Math.max(0, Math.min(n - 1, cursor[1] + delta[1])),
      ];
      e.preventDefault();
      draw();
    } else if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      place(cursor[0], cursor[1]);
    }
  });

  $('end-turn').addEventListener('click', endTurn);
  $('undo').addEventListener('click', () => {
    if (!turnStart || !canUserAct()) return;
    game.restore(turnStart);
    setHint();
    render();
    refreshRecommendation();
  });
  $('new-game').addEventListener('click', newGame);
  $('result-new').addEventListener('click', newGame);
  $('side')?.addEventListener('change', newGame);
  $('level')?.addEventListener('change', () => { setHint(); refreshRecommendation(); });
  window.addEventListener('model-status', () => { updateBadge(); refreshRecommendation(); });
  window.addEventListener('resize', resize);

  newGame();
})();
