(() => {
  'use strict';

  const BOARD_SIZE = 19;
  const isSelfMode = document.body.dataset.mode === 'self';
  let latestKifu = null;

  function boardGeometry() {
    const canvas = document.getElementById('board');
    const wrap = canvas?.parentElement;
    if (!canvas || !wrap) return null;
    const canvasRect = canvas.getBoundingClientRect();
    const wrapRect = wrap.getBoundingClientRect();
    const size = canvasRect.width;
    const pad = Math.max(14, size * 0.035);
    const cell = (size - pad * 2) / (BOARD_SIZE - 1);
    return { wrap, canvasRect, wrapRect, pad, cell };
  }

  function previousTurnFromHistory() {
    const game = window.SIXMOK_PLAY?.getShownGame?.();
    if (!game || !Array.isArray(game.history)) return [];

    let lastEnd = -1;
    for (let i = game.history.length - 1; i >= 0; i--) {
      if (game.history[i]?.[1] === END_TURN) {
        lastEnd = i;
        break;
      }
    }
    if (lastEnd < 0) return [];

    const moves = [];
    for (let i = lastEnd - 1; i >= 0; i--) {
      const move = game.history[i]?.[1];
      if (move === END_TURN) break;
      if (Array.isArray(move) && move.length === 2) moves.push([...move]);
    }
    moves.reverse();
    return moves;
  }

  function previousTurnFromKifu() {
    if (!latestKifu || latestKifu.index <= 0) return [];
    const items = latestKifu.items || [];
    const active = items.find((item) => item.positionIndex === latestKifu.index);
    if (active?.complete) return (active.moves || []).map((move) => [...move]);

    for (let i = items.length - 1; i >= 0; i--) {
      const item = items[i];
      if (item.positionIndex < latestKifu.index && item.complete) {
        return (item.moves || []).map((move) => [...move]);
      }
    }
    return [];
  }

  function previousTurnMoves() {
    if (isSelfMode && latestKifu) return previousTurnFromKifu();
    return previousTurnFromHistory();
  }

  function ensureLayer(wrap) {
    let layer = document.getElementById('previous-turn-markers');
    if (!layer) {
      layer = document.createElement('div');
      layer.id = 'previous-turn-markers';
      layer.className = 'previous-turn-markers';
      wrap.appendChild(layer);
    }
    return layer;
  }

  function renderPreviousTurnMarkers() {
    const geom = boardGeometry();
    if (!geom) return;
    const layer = ensureLayer(geom.wrap);
    layer.textContent = '';

    for (const [r, c] of previousTurnMoves()) {
      const marker = document.createElement('span');
      marker.className = 'previous-turn-marker';
      marker.style.left = `${geom.canvasRect.left - geom.wrapRect.left + geom.pad + c * geom.cell}px`;
      marker.style.top = `${geom.canvasRect.top - geom.wrapRect.top + geom.pad + r * geom.cell}px`;
      marker.style.width = `${Math.max(14, geom.cell * 0.66)}px`;
      marker.style.height = marker.style.width;
      layer.appendChild(marker);
    }
  }

  const scheduleRender = () => requestAnimationFrame(renderPreviousTurnMarkers);

  window.addEventListener('kifu-update', (event) => {
    latestKifu = event.detail || null;
    scheduleRender();
  });
  window.addEventListener('resize', scheduleRender);
  window.addEventListener('load', scheduleRender);

  const slots = document.getElementById('slots');
  if (slots) new MutationObserver(scheduleRender).observe(slots, { childList: true });

  scheduleRender();
})();
