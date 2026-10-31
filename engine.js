/* 6목 변형 게임 엔진 + 규칙 기반 AI.
   game.py / play.py 와 동일한 규칙으로 동작해야 한다.
   DOM에 의존하지 않는다. 학습한 모델을 붙일 때 이 파일만 그대로 두고 AI만 교체한다. */

const EMPTY = 0, BLACK = 1, WHITE = 2;
const END_TURN = 'end';
const DIRECTIONS = [[0, 1], [1, 0], [1, 1], [1, -1]];

const DEFAULTS = {
  boardSize: 19,
  winLength: 6,
  maxPlacesPerTurn: 3,
  stonesGainedPerTurn: 2,
  blackStartStones: 1,
  whiteStartStones: 2,
  drawFillRatio: 0.70,
  allowOverline: true,
};

const other = (color) => (color === BLACK ? WHITE : BLACK);

class SixMok {
  constructor(config = {}) {
    this.config = { ...DEFAULTS, ...config };
    const n = this.config.boardSize;
    if (n < 19 || n > 25) throw new Error('판 크기는 19 이상 25 이하여야 합니다.');
    this.board = new Int8Array(n * n);
    this.stones = { [BLACK]: this.config.blackStartStones, [WHITE]: this.config.whiteStartStones };
    this.current = BLACK;
    this.placedThisTurn = 0;
    this.stonesOnBoard = 0;
    this.winner = null;
    this.isDraw = false;
    this.winningLine = null;
    this.lastPlaced = [];
    this.history = [];
  }

  get size() { return this.config.boardSize; }
  get isOver() { return this.winner !== null || this.isDraw; }
  get drawThreshold() { return Math.floor(this.size * this.size * this.config.drawFillRatio); }

  at(r, c) { return this.board[r * this.size + c]; }
  inside(r, c) { return r >= 0 && r < this.size && c >= 0 && c < this.size; }

  canPlace() {
    return !this.isOver
      && this.placedThisTurn < this.config.maxPlacesPerTurn
      && this.stones[this.current] > 0;
  }

  emptyPoints() {
    const out = [];
    for (let r = 0; r < this.size; r++) {
      for (let c = 0; c < this.size; c++) if (this.at(r, c) === EMPTY) out.push([r, c]);
    }
    return out;
  }

  snapshot() {
    return {
      board: this.board.slice(),
      stones: { ...this.stones },
      current: this.current,
      placedThisTurn: this.placedThisTurn,
      stonesOnBoard: this.stonesOnBoard,
      winner: this.winner,
      isDraw: this.isDraw,
      winningLine: this.winningLine,
      lastPlaced: this.lastPlaced.slice(),
      historyLength: this.history.length,
    };
  }

  restore(s) {
    this.board = s.board.slice();
    this.stones = { ...s.stones };
    this.current = s.current;
    this.placedThisTurn = s.placedThisTurn;
    this.stonesOnBoard = s.stonesOnBoard;
    this.winner = s.winner;
    this.isDraw = s.isDraw;
    this.winningLine = s.winningLine;
    this.lastPlaced = s.lastPlaced.slice();
    this.history.length = s.historyLength;
  }

  clone() {
    const g = new SixMok(this.config);
    g.restore(this.snapshot());
    return g;
  }

  place(r, c) {
    if (this.isOver) throw new Error('이미 끝난 게임입니다.');
    if (!this.inside(r, c)) throw new Error('판 밖입니다.');
    if (this.at(r, c) !== EMPTY) throw new Error('이미 돌이 있습니다.');
    if (this.placedThisTurn >= this.config.maxPlacesPerTurn) throw new Error('한 턴에 세 개까지만 둘 수 있습니다.');
    if (this.stones[this.current] <= 0) throw new Error('보유한 돌이 없습니다.');

    const color = this.current;
    this.board[r * this.size + c] = color;
    this.stones[color] -= 1;
    this.placedThisTurn += 1;
    this.stonesOnBoard += 1;
    this.lastPlaced.push([r, c]);
    this.history.push([color, [r, c]]);

    const line = this.winningLineAt(r, c, color);
    if (line) {
      this.winner = color;
      this.winningLine = line;
    } else if (this.stonesOnBoard > this.drawThreshold) {
      this.isDraw = true;
    }
  }

  endTurn() {
    if (this.isOver) return;
    const color = this.current;
    this.stones[color] += this.config.stonesGainedPerTurn;
    this.history.push([color, END_TURN]);
    this.current = other(color);
    this.placedThisTurn = 0;
    this.lastPlaced = [];
  }

  play(move) {
    if (move === END_TURN) this.endTurn();
    else this.place(move[0], move[1]);
  }

  /** (r, c)에 color가 놓였을 때 완성되는 승리 라인. 없으면 null. */
  winningLineAt(r, c, color) {
    const need = this.config.winLength;
    for (const [dr, dc] of DIRECTIONS) {
      const cells = [[r, c]];
      for (const sign of [1, -1]) {
        let rr = r + dr * sign, cc = c + dc * sign;
        while (this.inside(rr, cc) && this.at(rr, cc) === color) {
          cells.push([rr, cc]);
          rr += dr * sign;
          cc += dc * sign;
        }
      }
      const run = cells.length;
      if (run === need || (this.config.allowOverline && run > need)) {
        cells.sort((a, b) => (a[0] - b[0]) || (a[1] - b[1]));
        return cells;
      }
    }
    return null;
  }
}

/* ------------------------------------------------------------------ 신경망 입력

   v5.10+ encoding.py 와 동일한 13평면 current-player-relative 규격이다.
   절대 색상 평면은 사용하지 않는다. */

const N_PLANES = 13;
const STONE_SCALE = 6;
const STONE_LINEAR_LIMIT = 12;
const STONE_TAIL_SCALE = 48;

function encodeStoneCount(stones) {
  if (stones <= STONE_LINEAR_LIMIT) return stones / STONE_SCALE;
  return 2 + (stones - STONE_LINEAR_LIMIT) / STONE_TAIL_SCALE;
}

/** 각 자리가 color 에게 얼마나 위험한/유망한 자리인지.

    연속 개수가 아니라 '연속된 여섯 칸(창)' 을 본다.
    창 안에 상대 돌이 하나라도 있으면 그 창은 죽은 창이라 세지 않는다.

    best[i]   i 를 품은 살아 있는 창 중 color 돌이 가장 많은 개수
    forks4[i] 그런 창의 돌이 need-2 개 이상인 방향의 수
    forks3[i] need-3 개 이상인 방향의 수 */
function threatGrids(game, color) {
  const n = game.size;
  const plane = n * n;
  const need = game.config.winLength;
  const best = new Int16Array(plane).fill(-1);
  const forks4 = new Int16Array(plane);
  const forks3 = new Int16Array(plane);
  const dirBest = new Int16Array(plane);

  for (const [dr, dc] of DIRECTIONS) {
    dirBest.fill(-1);
    for (let r = 0; r < n; r++) {
      for (let c = 0; c < n; c++) {
        const er = r + dr * (need - 1), ec = c + dc * (need - 1);
        if (er < 0 || er >= n || ec < 0 || ec >= n) continue;

        let mine = 0, blocked = false;
        for (let k = 0; k < need; k++) {
          const v = game.board[(r + dr * k) * n + (c + dc * k)];
          if (v === color) mine += 1;
          else if (v !== EMPTY) { blocked = true; break; }
        }
        if (blocked) continue;

        for (let k = 0; k < need; k++) {
          const i = (r + dr * k) * n + (c + dc * k);
          if (dirBest[i] < mine) dirBest[i] = mine;
        }
      }
    }
    for (let i = 0; i < plane; i++) {
      if (dirBest[i] > best[i]) best[i] = dirBest[i];
      if (dirBest[i] >= need - 2) forks4[i] += 1;
      if (dirBest[i] >= need - 3) forks3[i] += 1;
    }
  }
  return { best, forks4, forks3 };
}

/** 지금 두면 곧바로 6목이 되는 자리인가. */
function isImmediateWin(game, index, color) {
  const { best } = threatGrids(game, color);
  return best[index] >= game.config.winLength - 1;
}

/** 현재 차례인 쪽 관점으로 (13, n, n) 입력을 만든다. */
function encodeState(game) {
  const n = game.size;
  const plane = n * n;
  const me = game.current;
  const foe = other(me);
  const need = game.config.winLength;
  const data = new Float32Array(N_PLANES * plane);

  for (let i = 0; i < plane; i++) {
    const v = game.board[i];
    if (v === me) data[i] = 1;
    else if (v === foe) data[plane + i] = 1;
  }

  data.fill(encodeStoneCount(game.stones[me]), 2 * plane, 3 * plane);
  data.fill(game.placedThisTurn / game.config.maxPlacesPerTurn, 3 * plane, 4 * plane);
  data.fill(encodeStoneCount(game.stones[foe]), 4 * plane, 5 * plane);

  for (const [offset, color] of [[5, me], [9, foe]]) {
    const { best, forks3 } = threatGrids(game, color);
    for (let i = 0; i < plane; i++) {
      if (game.board[i] !== EMPTY) continue;
      if (best[i] >= need - 1) data[(offset + 0) * plane + i] = 1;
      if (best[i] >= need - 2) data[(offset + 1) * plane + i] = 1;
      if (best[i] >= need - 3) data[(offset + 2) * plane + i] = 1;
      if (forks3[i] >= 2) data[(offset + 3) * plane + i] = 1;
    }
  }
  return data;
}

/* ------------------------------------------------------------------ 규칙 기반 AI */

/* 창(여섯 칸) 안의 내 돌 개수별 가중치. 창에 상대 돌이 있으면 죽은 창이다.
   한 자리는 최대 24개 창에 걸리므로, 3등급 이하를 다 합쳐도(24 x 120 = 2880)
   4등급 하나(3000)를 넘지 못한다. 그래야 '한 수면 6목' 일 때만 급한 상황이 된다. */
const RUN_SCORE = { 0: 1, 1: 3, 2: 20, 3: 120, 4: 3000, 5: 200000 };
const WIN_SCORE = 10000000;

/** 각 자리의 가치. 그 자리를 품은 살아 있는 창을 모두 합산한다. */
function windowScores(game, color) {
  const n = game.size;
  const plane = n * n;
  const need = game.config.winLength;
  const total = new Float64Array(plane);

  for (const [dr, dc] of DIRECTIONS) {
    for (let r = 0; r < n; r++) {
      for (let c = 0; c < n; c++) {
        const er = r + dr * (need - 1), ec = c + dc * (need - 1);
        if (er < 0 || er >= n || ec < 0 || ec >= n) continue;

        let mine = 0, dead = false;
        for (let k = 0; k < need; k++) {
          const v = game.board[(r + dr * k) * n + (c + dc * k)];
          if (v === color) mine += 1;
          else if (v !== EMPTY) { dead = true; break; }
        }
        if (dead) continue;

        const value = mine >= need - 1 ? WIN_SCORE : (RUN_SCORE[mine] || 0);
        for (let k = 0; k < need; k++) {
          total[(r + dr * k) * n + (c + dc * k)] += value;
        }
      }
    }
  }
  return total;
}

function candidates(game, radius = 2) {
  const n = game.size;
  if (game.stonesOnBoard === 0) return [[Math.floor(n / 2), Math.floor(n / 2)]];
  const seen = new Set();
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      if (game.at(r, c) === EMPTY) continue;
      for (let dr = -radius; dr <= radius; dr++) {
        for (let dc = -radius; dc <= radius; dc++) {
          const rr = r + dr, cc = c + dc;
          if (game.inside(rr, cc) && game.at(rr, cc) === EMPTY) seen.add(rr * n + cc);
        }
      }
    }
  }
  return [...seen].map((i) => [Math.floor(i / n), i % n]);
}

/* 난이도.
   규칙 기반 fallback의 mistake/defense/hoard 값은 기존 동작을 유지한다.
   학습 모델은 모든 합법 후보를 같은 1-ply P×V로 평가하고 temperature와 가중치 컷오프로 난이도를 나눈다. */
const AI_LEVELS = {
  easy:   { label: '쉬움',   mistake: 0.35, defense: 0.40, hoard: false, temperature: 1.2, cutoffRel: 0.02 },
  normal: { label: '보통',   mistake: 0.12, defense: 0.75, hoard: true,  temperature: 0.6, cutoffRel: 0.05 },
  hard:   { label: '어려움', mistake: 0.00, defense: 1.00, hoard: true,  temperature: 0.3, cutoffRel: 0.10 },
};

function bestPoint(game, color, defenseWeight = 1.0) {
  const attack = windowScores(game, color);
  const defense = windowScores(game, other(color));
  const n = game.size;
  let best = [], bestScore = -1;
  for (const [r, c] of candidates(game)) {
    const i = r * n + c;
    const score = attack[i] + defenseWeight * defense[i];
    if (score > bestScore) { best = [[r, c]]; bestScore = score; }
    else if (score === bestScore) best.push([r, c]);
  }
  if (!best.length) return { point: null, score: -1 };
  return { point: best[Math.floor(Math.random() * best.length)], score: bestScore };
}

/** 한 턴 분량의 행동 목록을 만든다. 게임 상태는 바꾸지 않는다. */
function planTurn(game, options = {}) {
  const { hoard = true, mistake = 0, defense = 1.0 } = options;
  const sim = game.clone();
  const moves = [];
  while (sim.canPlace()) {
    let { point, score } = bestPoint(sim, sim.current, defense);
    if (!point) break;

    if (mistake > 0 && score < WIN_SCORE && Math.random() < mistake) {
      const pool = candidates(sim);
      if (pool.length) {
        point = pool[Math.floor(Math.random() * pool.length)];
        score = 0;
      }
    }

    const urgent = score >= RUN_SCORE[4];
    if (hoard && !urgent && sim.stones[sim.current] < sim.config.maxPlacesPerTurn && sim.placedThisTurn === 0) {
      break;
    }
    sim.place(point[0], point[1]);
    moves.push(point);
    if (sim.isOver) break;
  }
  return moves;
}

if (typeof module !== 'undefined') {
  module.exports = {
    SixMok, planTurn, bestPoint, windowScores, encodeState, encodeStoneCount,
    threatGrids, isImmediateWin, AI_LEVELS, N_PLANES,
    EMPTY, BLACK, WHITE, END_TURN, other,
  };
}