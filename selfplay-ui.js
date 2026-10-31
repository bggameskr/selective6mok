(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const BOARD_SIZE = 19;
  let latestAnalysis = null;
  let selectedAction = null;

  function pct(v, digits = 2) { return Number.isFinite(v) ? (v * 100).toFixed(digits) + '%' : '—'; }
  function num(v, digits = 4) { return Number.isFinite(v) ? v.toFixed(digits) : '—'; }

  function ensureBoardOverlays() {
    const wrap = document.querySelector('.board-wrap');
    if (!wrap) return {};
    let marker = $('candidate-marker');
    if (!marker) {
      marker = document.createElement('div');
      marker.id = 'candidate-marker';
      marker.className = 'candidate-marker';
      marker.hidden = true;
      wrap.appendChild(marker);
    }
    let info = $('board-info-overlay');
    if (!info) {
      info = document.createElement('div');
      info.id = 'board-info-overlay';
      info.className = 'board-info-overlay';
      info.hidden = true;
      wrap.appendChild(info);
    }
    return { wrap, marker, info };
  }

  function canvasGeometry() {
    const canvas = $('board');
    const wrap = canvas?.parentElement;
    if (!canvas || !wrap) return null;
    const canvasRect = canvas.getBoundingClientRect();
    const wrapRect = wrap.getBoundingClientRect();
    const size = canvasRect.width;
    const pad = Math.max(14, size * 0.035);
    const cell = (size - pad * 2) / (BOARD_SIZE - 1);
    return { canvas, wrap, canvasRect, wrapRect, size, pad, cell };
  }

  function actionPoint(action) {
    if (!Number.isInteger(action) || action < 0 || action >= BOARD_SIZE * BOARD_SIZE) return null;
    const geom = canvasGeometry();
    if (!geom) return null;
    const r = Math.floor(action / BOARD_SIZE);
    const c = action % BOARD_SIZE;
    return {
      x: geom.canvasRect.left - geom.wrapRect.left + geom.pad + c * geom.cell,
      y: geom.canvasRect.top - geom.wrapRect.top + geom.pad + r * geom.cell,
      cell: geom.cell,
      row: r,
      col: c,
    };
  }

  function showCandidateMarker(action) {
    selectedAction = action;
    const { marker } = ensureBoardOverlays();
    const point = actionPoint(action);
    if (!marker || !point) {
      if (marker) marker.hidden = true;
      return;
    }
    marker.hidden = false;
    marker.style.left = `${point.x}px`;
    marker.style.top = `${point.y}px`;
    marker.style.width = `${Math.max(18, point.cell * 0.88)}px`;
    marker.style.height = marker.style.width;
  }

  function refreshMarker() {
    if (selectedAction !== null) showCandidateMarker(selectedAction);
  }

  function candidateByAction(action) {
    return latestAnalysis?.candidates?.find((row) => row.action === action) || null;
  }

  function overlayHtml(row, r, c) {
    if (!row) {
      return `<strong>${r + 1}·${c + 1}</strong><span>현재 cutoff 후보 아님</span>`;
    }
    return `
      <strong>${row.label}</strong>
      <span>P ${pct(row.policy, 3)} · 전체 ${row.policyRank || '—'}위</span>
      <span>V ${num(row.childValue, 4)} · P×V ${num(row.pv, 6)}</span>
      <span>Greedy score ${num(row.score, 6)}</span>
      <small>${row.reason}</small>`;
  }

  function handleBoardPointer(e) {
    if (!latestAnalysis) return;
    const geom = canvasGeometry();
    const { info } = ensureBoardOverlays();
    if (!geom || !info) return;
    const x = e.clientX - geom.canvasRect.left;
    const y = e.clientY - geom.canvasRect.top;
    const c = Math.round((x - geom.pad) / geom.cell);
    const r = Math.round((y - geom.pad) / geom.cell);
    if (r < 0 || r >= BOARD_SIZE || c < 0 || c >= BOARD_SIZE) {
      info.hidden = true;
      return;
    }
    const px = geom.pad + c * geom.cell;
    const py = geom.pad + r * geom.cell;
    if (Math.hypot(px - x, py - y) > geom.cell * 0.58) {
      info.hidden = true;
      return;
    }
    const action = r * BOARD_SIZE + c;
    const row = candidateByAction(action);
    info.innerHTML = overlayHtml(row, r, c);
    info.hidden = false;
    const localX = e.clientX - geom.wrapRect.left;
    const localY = e.clientY - geom.wrapRect.top;
    info.style.left = `${localX}px`;
    info.style.top = `${localY}px`;
  }

  function renderAnalysis(data) {
    latestAnalysis = data;
    const g = data.greedy;
    const summary = $('analysis-summary');
    if (summary) {
      summary.innerHTML = `
        <div class="metric hero"><span>Greedy 추천</span><strong>${g.label}</strong><small>${g.reason}</small></div>
        <div class="metric"><span>Greedy score</span><strong>${num(g.score, 6)}</strong><small>보정 ${num(g.modifier, 4)}</small></div>
        <div class="metric"><span>Policy</span><strong>${pct(g.policy, 3)}</strong><small>전체 합법수 중 ${g.policyRank || '—'}위 · logit ${num(g.rawLogit, 3)}</small></div>
        <div class="metric"><span>Child value</span><strong>${num(g.childValue, 4)}</strong><small>승률 환산 ${pct(g.winProb)}</small></div>
        <div class="metric"><span>P×V</span><strong>${num(g.pv, 6)}</strong><small>${data.decisionPath}</small></div>
        <div class="metric"><span>Root value</span><strong>${num(data.rootValue, 4)}</strong><small>승률 환산 ${pct(data.rootWinProb)}</small></div>
      `;
    }
    const meta = $('analysis-meta');
    if (meta) meta.textContent = `${data.inputPlanes || '?'}-plane · policy cutoff ${pct(data.policyCutoffRel, 0)} · child 평가 ${data.childEvaluatedCount} · 난수 없음`;
    renderCandidates(data.candidates || [], data);
    if (selectedAction !== null && !candidateByAction(selectedAction)) {
      selectedAction = null;
      const marker = $('candidate-marker');
      if (marker) marker.hidden = true;
    } else {
      refreshMarker();
    }
  }

  function renderCandidates(rows, data) {
    const body = $('candidate-body');
    const count = $('candidate-count');
    if (count) count.textContent = `cutoff/전술 필터 통과 ${rows.length}개 · child 평가 ${data.childEvaluatedCount}개`;
    if (!body) return;
    body.textContent = '';
    if (!rows.length) {
      body.innerHTML = '<tr><td colspan="10">표시할 후보가 없습니다.</td></tr>';
      return;
    }
    for (const row of rows) {
      const tr = document.createElement('tr');
      tr.dataset.action = String(row.action);
      tr.tabIndex = 0;
      tr.title = row.action < BOARD_SIZE * BOARD_SIZE ? '클릭하면 판에서 위치를 표시합니다.' : 'END_TURN';
      if (row.candidateRank === 1) tr.classList.add('greedy-row');
      if (row.action === selectedAction) tr.classList.add('selected-row');
      tr.innerHTML = `
        <td>${row.candidateRank}</td><td><strong>${row.label}</strong></td><td>${row.policyRank || '—'}</td>
        <td>${pct(row.policy, 3)}</td><td>${num(row.rawLogit, 3)}</td><td>${num(row.childValue, 4)}</td>
        <td>${pct(row.winProb)}</td><td>${num(row.pv, 6)}</td><td>${num(row.modifier, 4)}</td>
        <td>${num(row.score, 6)}<br><small>${row.reason}</small></td>`;
      body.appendChild(tr);
    }
  }

  function endLabel(item) {
    if (!item.complete) return '진행 중';
    if (item.endReason === 'forced') return '강제 종료';
    if (item.endReason === 'game-over') return '대국 종료';
    return item.moves.length ? '턴 종료' : '무착수 END_TURN';
  }

  function renderKifu(detail) {
    const pos = $('kifu-pos');
    if (pos) pos.textContent = detail.label;
    for (const [id, disabled] of [
      ['kifu-first', detail.index <= 0], ['kifu-prev', detail.index <= 0],
      ['kifu-next', detail.index >= detail.total], ['kifu-last', detail.index >= detail.total],
    ]) {
      const node = $(id); if (node) node.disabled = disabled;
    }

    const list = $('kifu-list');
    if (!list) return;
    const wasLatest = detail.index >= detail.total;
    list.textContent = '';
    if (!detail.items.length) {
      list.innerHTML = '<span class="kifu-empty">아직 완료되거나 진행 중인 턴이 없습니다.</span>';
      return;
    }
    for (const item of detail.items) {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'kifu-item' + (item.active ? ' active' : '');
      button.dataset.index = String(item.positionIndex);
      const moves = item.moveLabels.length ? item.moveLabels.join(' · ') : '착수 없음';
      button.innerHTML = `
        <span class="kifu-turn">${item.turnNo}턴</span>
        <span class="kifu-color">${item.colorName}</span>
        <span class="kifu-moves"><strong>${moves}</strong><span class="kifu-end">${endLabel(item)}</span></span>`;
      list.appendChild(button);
    }
    requestAnimationFrame(() => {
      if (wasLatest) list.scrollTop = list.scrollHeight;
      else list.querySelector('.active')?.scrollIntoView({ block:'nearest' });
    });
  }

  window.addEventListener('model-analysis', (e) => renderAnalysis(e.detail));
  window.addEventListener('kifu-update', (e) => renderKifu(e.detail));
  window.addEventListener('resize', refreshMarker);

  $('kifu-first')?.addEventListener('click', () => window.SELFPLAY_REVIEW?.first());
  $('kifu-prev')?.addEventListener('click', () => window.SELFPLAY_REVIEW?.prev());
  $('kifu-next')?.addEventListener('click', () => window.SELFPLAY_REVIEW?.next());
  $('kifu-last')?.addEventListener('click', () => window.SELFPLAY_REVIEW?.last());
  $('kifu-list')?.addEventListener('click', (e) => {
    const button = e.target.closest('[data-index]');
    if (button) window.SELFPLAY_REVIEW?.goto(Number(button.dataset.index));
  });

  function selectCandidateRow(row) {
    const action = Number(row.dataset.action);
    if (!Number.isInteger(action) || action >= BOARD_SIZE * BOARD_SIZE) return;
    selectedAction = action;
    document.querySelectorAll('#candidate-body tr.selected-row').forEach((tr) => tr.classList.remove('selected-row'));
    row.classList.add('selected-row');
    showCandidateMarker(action);
  }

  $('candidate-body')?.addEventListener('click', (e) => {
    const row = e.target.closest('tr[data-action]');
    if (row) selectCandidateRow(row);
  });
  $('candidate-body')?.addEventListener('keydown', (e) => {
    if (e.key !== 'Enter' && e.key !== ' ') return;
    const row = e.target.closest('tr[data-action]');
    if (!row) return;
    e.preventDefault();
    selectCandidateRow(row);
  });

  const canvas = $('board');
  canvas?.addEventListener('mousemove', handleBoardPointer);
  canvas?.addEventListener('mouseleave', () => {
    const info = $('board-info-overlay');
    if (info) info.hidden = true;
  });
})();