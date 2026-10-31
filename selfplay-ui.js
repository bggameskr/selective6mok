(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);

  function pct(v, digits = 2) {
    return Number.isFinite(v) ? (v * 100).toFixed(digits) + '%' : '—';
  }
  function num(v, digits = 4) {
    return Number.isFinite(v) ? v.toFixed(digits) : '—';
  }

  function renderAnalysis(data) {
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
    if (meta) {
      meta.textContent = `${data.inputPlanes || '?'}-plane · policy cutoff ${pct(data.policyCutoffRel, 0)} · 후보 ${data.candidateCount}/${data.legalCount} · child 평가 ${data.childEvaluatedCount} · 난수 없음`;
    }
    renderCandidates(data.candidates || []);
  }

  function renderCandidates(rows) {
    const body = $('candidate-body');
    const count = $('candidate-count');
    if (count) count.textContent = `cutoff/전술 필터 통과 ${rows.length}개 후보`;
    if (!body) return;
    body.textContent = '';
    if (!rows.length) {
      body.innerHTML = '<tr><td colspan="10">표시할 후보가 없습니다.</td></tr>';
      return;
    }
    for (const row of rows) {
      const tr = document.createElement('tr');
      if (row.candidateRank === 1) tr.classList.add('greedy-row');
      tr.innerHTML = `
        <td>${row.candidateRank}</td>
        <td><strong>${row.label}</strong></td>
        <td>${row.policyRank || '—'}</td>
        <td>${pct(row.policy, 3)}</td>
        <td>${num(row.rawLogit, 3)}</td>
        <td>${num(row.childValue, 4)}</td>
        <td>${pct(row.winProb)}</td>
        <td>${num(row.pv, 6)}</td>
        <td>${num(row.modifier, 4)}</td>
        <td>${num(row.score, 6)}<br><small>${row.reason}</small></td>
      `;
      body.appendChild(tr);
    }
  }

  function renderKifu(detail) {
    const pos = $('kifu-pos');
    if (pos) pos.textContent = `${detail.index} / ${detail.total}`;
    for (const [id, disabled] of [
      ['kifu-first', detail.index <= 0], ['kifu-prev', detail.index <= 0],
      ['kifu-next', detail.index >= detail.total], ['kifu-last', detail.index >= detail.total],
    ]) {
      const node = $(id);
      if (node) node.disabled = disabled;
    }

    const list = $('kifu-list');
    if (!list) return;
    list.textContent = '';
    if (!detail.items.length) {
      list.innerHTML = '<span class="kifu-empty">아직 둔 수가 없습니다.</span>';
      return;
    }
    for (const item of detail.items) {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'kifu-item' + (item.index === detail.index ? ' active' : '') + (item.isEnd ? ' end' : '');
      button.dataset.index = String(item.index);
      button.innerHTML = item.isEnd
        ? `<span class="kifu-color">${item.color}</span><span>턴 끝</span>`
        : `<span class="kifu-no">${item.stoneNo}.</span><span class="kifu-color">${item.color}</span><strong>${item.label}</strong>`;
      list.appendChild(button);
    }
    list.querySelector('.active')?.scrollIntoView({ block: 'nearest' });
  }

  window.addEventListener('model-analysis', (e) => renderAnalysis(e.detail));
  window.addEventListener('kifu-update', (e) => renderKifu(e.detail));

  $('kifu-first')?.addEventListener('click', () => window.SELFPLAY_REVIEW?.first());
  $('kifu-prev')?.addEventListener('click', () => window.SELFPLAY_REVIEW?.prev());
  $('kifu-next')?.addEventListener('click', () => window.SELFPLAY_REVIEW?.next());
  $('kifu-last')?.addEventListener('click', () => window.SELFPLAY_REVIEW?.last());
  $('kifu-list')?.addEventListener('click', (e) => {
    const button = e.target.closest('[data-index]');
    if (button) window.SELFPLAY_REVIEW?.goto(Number(button.dataset.index));
  });
})();
