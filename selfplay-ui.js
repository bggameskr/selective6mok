(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  let latest = null;

  function pct(v, digits = 2) {
    return Number.isFinite(v) ? (v * 100).toFixed(digits) + '%' : '—';
  }
  function num(v, digits = 4) {
    return Number.isFinite(v) ? v.toFixed(digits) : '—';
  }
  function actionLabel(row) {
    return row.isEnd ? 'END_TURN' : `${row.row + 1}·${row.col + 1}`;
  }

  function renderSummary(data) {
    const node = $('analysis-summary');
    if (!node) return;
    const greedy = data.rows.find((row) => row.greedy);
    const levelCards = ['easy', 'normal', 'hard'].map((key) => {
      const level = data.levels[key];
      return `<div class="metric"><span>${level.label} T</span><strong>${num(level.effectiveTemperature, 2)}</strong><small>base ${num(level.baseTemperature, 2)} · 컷 ${pct(level.cutoffRel, 0)} · ${level.eligibleCount}수</small></div>`;
    }).join('');
    node.innerHTML = `
      <div class="metric hero"><span>Greedy 추천</span><strong>${greedy ? actionLabel(greedy) : '—'}</strong><small>${data.decisionPath}</small></div>
      <div class="metric"><span>Root value</span><strong>${num(data.rootValue, 4)}</strong><small>승률 환산 ${pct(data.rootWinProb)}</small></div>
      <div class="metric"><span>Policy entropy</span><strong>${num(data.rootPolicyEntropy, 4)}</strong><small>합법 ${data.legalCount} · 실제 후보 ${data.actualCandidateCount}</small></div>
      ${levelCards}
    `;
    const meta = $('analysis-meta');
    if (meta) meta.textContent = `${data.inputPlanes || '?'}-plane · policy cutoff ${pct(data.policyCutoffRel, 0)} · 난수 샘플링 없음`;
  }

  function sortedRows(data) {
    const key = $('analysis-sort')?.value || 'greedy';
    const rows = [...data.rows];
    const cmp = {
      greedy: (a, b) => (b.selectionWeight - a.selectionWeight) || (a.action - b.action),
      policy: (a, b) => (b.policy - a.policy) || (a.action - b.action),
      value: (a, b) => (b.childValue - a.childValue) || (b.policy - a.policy),
      easy: (a, b) => (b.levels.easy.probability - a.levels.easy.probability) || (a.action - b.action),
      normal: (a, b) => (b.levels.normal.probability - a.levels.normal.probability) || (a.action - b.action),
      hard: (a, b) => (b.levels.hard.probability - a.levels.hard.probability) || (a.action - b.action),
    }[key];
    return rows.sort(cmp);
  }

  function renderTable(data) {
    const body = $('analysis-body');
    if (!body) return;
    body.textContent = '';
    const rows = sortedRows(data);
    rows.forEach((row, idx) => {
      const tr = document.createElement('tr');
      if (row.greedy) tr.classList.add('greedy-row');
      if (!row.actualCandidate) tr.classList.add('excluded-row');
      tr.innerHTML = `
        <td>${idx + 1}</td>
        <td><strong>${actionLabel(row)}</strong>${row.greedy ? '<span class="greedy-mark">G</span>' : ''}</td>
        <td>${row.policyRank}</td>
        <td>${pct(row.policy, 3)}</td>
        <td>${num(row.rawLogit, 3)}</td>
        <td>${num(row.childValue, 4)}</td>
        <td>${pct(row.winProb)}</td>
        <td>${num(row.pv, 6)}</td>
        <td>${num(row.modifier, 4)}</td>
        <td>${num(row.selectionWeight, 6)}</td>
        <td>${row.actualCandidate ? '포함' : '제외'}<br><small>${row.reason}</small></td>
        <td class="temp ${row.levels.easy.eligible ? '' : 'off'}">${pct(row.levels.easy.probability, 3)}</td>
        <td class="temp ${row.levels.normal.eligible ? '' : 'off'}">${pct(row.levels.normal.probability, 3)}</td>
        <td class="temp ${row.levels.hard.eligible ? '' : 'off'}">${pct(row.levels.hard.probability, 3)}</td>
      `;
      body.appendChild(tr);
    });
    const count = $('analysis-count');
    if (count) count.textContent = `${rows.length}개 합법 행동 전체 표시`;
  }

  function render(data) {
    latest = data;
    renderSummary(data);
    renderTable(data);
  }

  window.addEventListener('model-analysis', (event) => render(event.detail));
  $('analysis-sort')?.addEventListener('change', () => { if (latest) renderTable(latest); });
})();
