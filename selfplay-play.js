(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const canvas = $('board');
  const ctx = canvas.getContext('2d');
  const colorName = { [BLACK]: '흑', [WHITE]: '백' };
  const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const finePointer = matchMedia('(pointer:fine)').matches;

  let game = null;
  let turnStart = null;
  let hover = null;
  let cursor = null;
  let logs = [];
  let resultShown = false;
  let geom = { pad: 0, cell: 0, px: 0 };
  let recommendation = null;
  let recommendationText = '';
  let recommendationSeq = 0;

  let reviewEntries = [];
  let reviewIndex = 0;
  let reviewView = null;

  function css(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function shownGame() {
    return reviewView || game;
  }

  function isLatest() {
    return reviewIndex === reviewEntries.length - 1;
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
    const g = shownGame();
    if (!g || !geom.px) return;
    const n = g.size;
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
        const v = g.at(r, c);
        if (v !== EMPTY) drawStone(r, c, v);
      }
    }

    ctx.strokeStyle = css('--accent');
    ctx.lineWidth = 1.7;
    for (const [r, c] of g.lastPlaced) {
      const [x, y] = xy(r, c);
      ctx.beginPath();
      ctx.arc(x, y, stoneRadius() * .42, 0, Math.PI * 2);
      ctx.stroke();
    }

    if (g.winningLine) {
      ctx.lineWidth = 2.4;
      for (const [r, c] of g.winningLine) {
        const [x, y] = xy(r, c);
        ctx.beginPath();
        ctx.arc(x, y, stoneRadius() * 1.12, 0, Math.PI * 2);
        ctx.stroke();
      }
    }

    if (recommendation && recommendation !== END_TURN && !g.isOver) {
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
    if (preview && canUserAct() && g.at(preview[0], preview[1]) === EMPTY && g.canPlace()) {
      drawStone(preview[0], preview[1], g.current, .4);
    }
    if (cursor) {
      const [x, y] = xy(cursor[0], cursor[1]);
      ctx.strokeStyle = css('--accent');
      ctx.strokeRect(x - geom.cell * .5, y - geom.cell * .5, geom.cell, geom.cell);
    }
  }

  function resize() {
    const g = shownGame();
    if (!g) return;
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
    geom.cell = (size - geom.pad * 2) / (g.size - 1);
    geom.px = size;
    draw();
  }

  function canUserAct() {
    return !!game && isLatest() && !game.isOver;
  }

  function renderBank(node, color, g) {
    node.textContent = '';
    const who = document.createElement('span');
    who.className = 'who';
    who.textContent = colorName[color];
    node.appendChild(who);
    for (let i = 0; i < Math.min(g.stones[color], 8); i++) {
      const token = document.createElement('span');
      token.className = 'token ' + (color === BLACK ? 'b' : 'w');
      node.appendChild(token);
    }
    const count = document.createElement('span');
    count.className = 'count';
    count.textContent = g.stones[color] + '개';
    node.appendChild(count);
  }

  function updateBadge() {
    const badge = $('ai-badge');
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

  function render() {
    const g = shownGame();
    draw();
    $('turn-dot').style.background = g.current === BLACK
      ? 'radial-gradient(circle at 34% 30%,#4a5350,#0d1110 70%)'
      : 'radial-gradient(circle at 34% 30%,#fff,#bdb5a4 72%)';
    $('turn-text').textContent = g.isOver
      ? '대국 종료'
      : `${colorName[g.current]} 차례${isLatest() ? '' : ' — 기보 탐색 중'}`;
    renderBank($('bank-black'), BLACK, g);
    renderBank($('bank-white'), WHITE, g);

    const slots = $('slots');
    slots.textContent = '';
    for (let i = 0; i < g.config.maxPlacesPerTurn; i++) {
      const slot = document.createElement('span');
      slot.className = 'slot';
      if (i < g.placedThisTurn) slot.classList.add('on');
      else if (i >= g.placedThisTurn + g.stones[g.current]) slot.classList.add('lock');
      slots.appendChild(slot);
    }

    $('end-turn').disabled = !canUserAct();
    $('undo').disabled = !canUserAct() || !turnStart || game.placedThisTurn === 0;
    $('recommend').innerHTML = recommendationText || '<strong>Greedy</strong> 모델 계산 중…';
    $('log').innerHTML = logs.slice(-45).join('<br>');
    $('log').scrollTop = $('log').scrollHeight;
    updateBadge();

    if (isLatest() && game.isOver) showResult();
    else $('result').hidden = true;
  }

  function setHint() {
    const g = shownGame();
    if (!isLatest()) {
      $('hint').innerHTML = '<span class="reviewing-note">기보 탐색 중입니다. 최신 수로 이동하면 다시 둘 수 있습니다.</span>';
      return;
    }
    if (g.isOver) return $('hint').textContent = '';
    if (!g.canPlace()) return $('hint').textContent = '더 둘 수 없습니다. 턴을 마치세요.';
    $('hint').textContent = '흑과 백을 모두 직접 둡니다. 점선 원은 난수 없는 greedy 추천수입니다.';
  }

  function reviewDetail() {
    let stoneNo = 0;
    const items = game.history.map(([color, move], i) => {
      const isEnd = move === END_TURN;
      if (!isEnd) stoneNo += 1;
      return {
        index: i + 1,
        stoneNo: isEnd ? null : stoneNo,
        color: colorName[color],
        isEnd,
        label: isEnd ? '턴 끝' : `${move[0] + 1}·${move[1] + 1}`,
      };
    });
    return { index: reviewIndex, total: reviewEntries.length - 1, items };
  }

  function dispatchReview() {
    window.dispatchEvent(new CustomEvent('kifu-update', { detail: reviewDetail() }));
  }

  function rebuildReview() {
    const replay = new SixMok(game.config);
    const entries = [{ snapshot: replay.snapshot() }];
    for (const [, move] of game.history) {
      replay.play(move);
      entries.push({ snapshot: replay.snapshot() });
    }
    reviewEntries = entries;
    reviewIndex = reviewEntries.length - 1;
    reviewView = null;
    dispatchReview();
  }

  function setReviewIndex(index) {
    if (!reviewEntries.length) return;
    const next = Math.max(0, Math.min(reviewEntries.length - 1, index));
    reviewIndex = next;
    if (isLatest()) {
      reviewView = null;
    } else {
      reviewView = new SixMok(game.config);
      reviewView.restore(reviewEntries[reviewIndex].snapshot);
    }
    recommendationSeq += 1;
    recommendation = null;
    recommendationText = '';
    setHint();
    render();
    dispatchReview();
    refreshRecommendation();
  }

  window.SELFPLAY_REVIEW = {
    first: () => setReviewIndex(0),
    prev: () => setReviewIndex(reviewIndex - 1),
    next: () => setReviewIndex(reviewIndex + 1),
    last: () => setReviewIndex(reviewEntries.length - 1),
    goto: (index) => setReviewIndex(index),
  };

  async function refreshRecommendation() {
    recommendation = null;
    recommendationText = '';
    const seq = ++recommendationSeq;
    const target = shownGame();
    if (!target || target.isOver) {
      render();
      return;
    }
    if (window.MODEL_STATUS !== 'ready' || typeof window.MODEL_ANALYZE_POSITION !== 'function') {
      recommendationText = '<strong>Greedy</strong> 모델 준비 중…';
      render();
      return;
    }

    const targetKey = `${reviewIndex}|${target.current}|${target.placedThisTurn}|${target.stonesOnBoard}|${Array.from(target.board).join('')}`;
    recommendationText = '<strong>Greedy</strong> 계산 중…';
    render();
    try {
      const result = await window.MODEL_ANALYZE_POSITION(target.clone());
      if (seq !== recommendationSeq) return;
      const now = shownGame();
      const nowKey = `${reviewIndex}|${now.current}|${now.placedThisTurn}|${now.stonesOnBoard}|${Array.from(now.board).join('')}`;
      if (nowKey !== targetKey) return refreshRecommendation();
      recommendation = result.greedyMove;
      recommendationText = recommendation === END_TURN
        ? '<strong>Greedy</strong> 턴 마치기'
        : `<strong>Greedy</strong> ${recommendation[0] + 1}·${recommendation[1] + 1}`;
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
    setTimeout(() => {
      if (isLatest()) $('result').hidden = false;
    }, reduceMotion ? 0 : 500);
  }

  function pointFromEvent(e) {
    const g = shownGame();
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const c = Math.round((x - geom.pad) / geom.cell);
    const r = Math.round((y - geom.pad) / geom.cell);
    if (!g.inside(r, c)) return null;
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
    rebuildReview();
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
    rebuildReview();
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

  function endTurn() {
    if (!canUserAct()) return;
    const color = game.current;
    logTurn(color, game.lastPlaced);
    game.endTurn();
    turnStart = game.snapshot();
    rebuildReview();
    setHint();
    render();
    refreshRecommendation();
  }

  function newGame() {
    game = new SixMok({ boardSize: 19 });
    turnStart = game.snapshot();
    hover = null;
    cursor = null;
    logs = [];
    resultShown = false;
    recommendation = null;
    recommendationText = '';
    recommendationSeq += 1;
    $('result').hidden = true;
    rebuildReview();
    resize();
    setHint();
    render();
    refreshRecommendation();
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
    const n = shownGame().size;
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
    rebuildReview();
    setHint();
    render();
    refreshRecommendation();
  });
  $('new-game').addEventListener('click', newGame);
  $('result-new').addEventListener('click', newGame);
  window.addEventListener('model-status', () => { updateBadge(); refreshRecommendation(); });
  window.addEventListener('resize', resize);

  newGame();
})();
