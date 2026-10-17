/* v5.10.2 web opening-window adapter.
   실제 모델 구현은 model-core.js에 두고, 자가대국/평가와 동일하게
   각 색이 첫 돌을 둔 턴부터 3개 완료 턴까지 오프닝 temperature를 적용한다.
   첫 착수 전 빈 턴은 오프닝 횟수를 소모하지 않는다. */

(() => {
  'use strict';

  const CORE_URL = 'model-core.js';
  const OPENING_TURNS = 3;
  const NORMAL_OPENING_TEMPERATURE = 1.0;
  const HARD_OPENING_TEMPERATURE = 0.8;

  function inOpeningWindow(game, color = game.current) {
    let started = false;
    let completed = 0;

    for (const [playedColor, move] of game.history) {
      if (playedColor !== color) continue;
      if (move === END_TURN) {
        if (started) completed += 1;
      } else {
        started = true;
      }
    }
    return completed < OPENING_TURNS;
  }

  function installOpeningPatch() {
    const corePlanTurn = window.MODEL_PLAN_TURN;
    if (typeof corePlanTurn !== 'function') {
      console.warn('model-core.js가 MODEL_PLAN_TURN을 등록하지 못했습니다.');
      return;
    }

    window.MODEL_PLAN_TURN = (game) => {
      if (!inOpeningWindow(game, game.current) || typeof window.AI_LEVEL !== 'function') {
        return corePlanTurn(game);
      }

      const originalAILevel = window.AI_LEVEL;
      window.AI_LEVEL = () => {
        const level = { ...originalAILevel() };
        const width = Number(level.lookahead) || 0;
        const baseTemperature = Number(level.temperature) || 0;
        if (width >= 6) {
          level.temperature = Math.max(baseTemperature, HARD_OPENING_TEMPERATURE);
        } else if (width > 0) {
          level.temperature = Math.max(baseTemperature, NORMAL_OPENING_TEMPERATURE);
        }
        return level;
      };

      try {
        // corePlanTurn은 호출 직후 AI_LEVEL()을 동기적으로 읽고 Promise를 반환한다.
        // 따라서 전역 함수는 즉시 복원해도 이번 수의 level 값은 이미 캡처돼 있다.
        return corePlanTurn(game);
      } finally {
        window.AI_LEVEL = originalAILevel;
      }
    };

    window.MODEL_IN_OPENING_WINDOW = inOpeningWindow;
  }

  const script = document.createElement('script');
  script.src = CORE_URL;
  script.async = false;
  script.onload = installOpeningPatch;
  script.onerror = () => {
    window.MODEL_STATUS = 'failed';
    window.MODEL_STATUS_DETAIL = `${CORE_URL}을 불러오지 못했습니다.`;
    window.dispatchEvent(new CustomEvent('model-status'));
  };
  document.head.appendChild(script);
})();
