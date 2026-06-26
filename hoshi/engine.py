"""HOSHI engine — pure, deterministic, dependency-free.

Coordinate system: points are (r, c), 0-indexed, on an N x N grid of
intersections. A piece is (color, kind) with color in {0, 1} and
kind in {'T' (trooper), 'D' (drone)}.

KEY RULE DECISIONS (see CLAUDE.md for rationale):
  * Capture is GROUP-based (Go liberties): connected same-color stones
    share liberties and die together when the group has none.
  * "Trooper protection" is emergent: a drone adjacent to a trooper joins
    its group and shares liberties. No special-case rule. Toggle a stricter
    variant with config.hard_trooper_protection if you want it back.
  * Suicide is illegal unless the move captures.
  * Optional positional superko prevents board-state repetition.
  * Range uses Chebyshev distance (king steps); capture uses orthogonal
    liberties only. Influence spreads diagonally; sieges are cardinal.
"""
from __future__ import annotations
from dataclasses import dataclass, field, replace
from typing import Optional, Iterable
import random

Point = tuple[int, int]
Piece = tuple[int, str]          # (color, 'T'|'D')
Move = tuple                     # ('drop_t', pt) | ('drone', pt) | ('pass',)


@dataclass(frozen=True)
class RuleConfig:
    n: int = 13                  # board is n x n
    drone_range: int = 4         # Chebyshev radius a trooper can deploy within
    troopers: int = 5            # troopers per player (reserve + home)
    capture_to_win: int = 3      # lose this many troopers and you lose
    setup_window: bool = True    # a trooper can't deploy the turn it lands
    superko: bool = True         # forbid repeating a whole board position
    hard_trooper_protection: bool = False  # NOTE: inert under group capture — an
    #   adjacent friendly drone is already in the trooper's group, so it dies with
    #   it and the protection check never fires. Kept as a hook; for a REAL trooper
    #   toughness lever see CLAUDE.md ("trooper armor"). Use troopers / capture_to_win
    #   / drone_range / board size as the working strength dials.
    drone_supply: int = 9999     # effectively unlimited; cap if you want scarcity
    max_plies: int = 600         # safety cap -> territory score if reached
    komi: float = 0.0            # added to player 1's territory score (tie-break / balance dial)


def neighbors(pt: Point, n: int) -> Iterable[Point]:
    r, c = pt
    if r > 0:     yield (r - 1, c)
    if r < n - 1: yield (r + 1, c)
    if c > 0:     yield (r, c - 1)
    if c < n - 1: yield (r, c + 1)


def cheb(a: Point, b: Point) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


@dataclass
class State:
    cfg: RuleConfig
    board: dict[Point, Piece] = field(default_factory=dict)
    reserve: list[int] = field(default_factory=lambda: [0, 0])      # troopers still in hand
    drones_used: list[int] = field(default_factory=lambda: [0, 0])
    lost_troopers: list[int] = field(default_factory=lambda: [0, 0])  # troopers of player i removed so far
    placed_ply: dict[Point, int] = field(default_factory=dict)       # trooper -> ply it landed
    to_move: int = 0
    ply: int = 0
    passes: int = 0
    history: set = field(default_factory=set)                        # positional superko hashes
    winner: Optional[int] = None                                    # set when game ends; -1 == draw
    over: bool = False

    # ---- introspection -------------------------------------------------
    def copy(self) -> "State":
        return State(
            cfg=self.cfg, board=dict(self.board), reserve=list(self.reserve),
            drones_used=list(self.drones_used), lost_troopers=list(self.lost_troopers),
            placed_ply=dict(self.placed_ply), to_move=self.to_move, ply=self.ply,
            passes=self.passes, history=set(self.history), winner=self.winner, over=self.over,
        )

    def trooper_ready(self, pt: Point) -> bool:
        if not self.cfg.setup_window:
            return True
        # ready on the owner's *next* turn: same parity, at least 2 plies later
        return self.ply >= self.placed_ply.get(pt, -10) + 2

    def my_ready_troopers(self, color: int) -> list[Point]:
        return [p for p, (col, k) in self.board.items()
                if col == color and k == 'T' and self.trooper_ready(p)]


# ---------------------------------------------------------------------------
# Group / liberty mechanics
# ---------------------------------------------------------------------------
def _group_of(board: dict[Point, Piece], start: Point, n: int) -> tuple[set[Point], set[Point]]:
    color = board[start][0]
    seen = {start}
    stack = [start]
    libs: set[Point] = set()
    while stack:
        p = stack.pop()
        for q in neighbors(p, n):
            if q in board:
                if board[q][0] == color and q not in seen:
                    seen.add(q); stack.append(q)
            else:
                libs.add(q)
    return seen, libs


def _dead_enemy_groups(board, just_played: Point, mover: int, n: int) -> set[Point]:
    dead: set[Point] = set()
    for q in neighbors(just_played, n):
        if q in board and board[q][0] != mover and q not in dead:
            grp, libs = _group_of(board, q, n)
            if not libs:
                dead |= grp
    return dead


# ---------------------------------------------------------------------------
# Move generation
# ---------------------------------------------------------------------------
def legal_moves(s: State) -> list[Move]:
    if s.over:
        return []
    moves: list[Move] = [('pass',)]
    cfg = s.cfg
    empties = [(r, c) for r in range(cfg.n) for c in range(cfg.n) if (r, c) not in s.board]

    # drop a trooper anywhere empty (legality of suicide checked in apply)
    if s.reserve[s.to_move] > 0:
        for pt in empties:
            if _move_ok(s, ('drop_t', pt)):
                moves.append(('drop_t', pt))

    # deploy a drone within range of a ready trooper
    if s.drones_used[s.to_move] < cfg.drone_supply:
        ready = s.my_ready_troopers(s.to_move)
        if ready:
            for pt in empties:
                if any(cheb(pt, tp) <= cfg.drone_range for tp in ready) and _move_ok(s, ('drone', pt)):
                    moves.append(('drone', pt))
    return moves


def _move_ok(s: State, move: Move) -> bool:
    """Cheap legality probe: simulate placement + captures, reject suicide / ko."""
    return _try_apply(s, move) is not None


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------
def _try_apply(s: State, move: Move) -> Optional[State]:
    cfg = s.cfg
    if move[0] == 'pass':
        ns = s.copy()
        ns.passes += 1
        ns.to_move ^= 1
        ns.ply += 1
        if ns.passes >= 2:
            _finish_territory(ns)
        return ns

    kind = 'T' if move[0] == 'drop_t' else 'D'
    pt = move[1]
    if pt in s.board:
        return None
    if kind == 'T' and s.reserve[s.to_move] <= 0:
        return None
    if kind == 'D':
        ready = s.my_ready_troopers(s.to_move)
        if not any(cheb(pt, tp) <= cfg.drone_range for tp in ready):
            return None

    ns = s.copy()
    mover = s.to_move
    ns.board[pt] = (mover, kind)

    # 1) remove dead enemy groups (protection-aware)
    dead = _dead_enemy_groups_protected(ns, pt, mover, cfg)
    for d in dead:
        if ns.board[d][1] == 'T':
            ns.lost_troopers[mover ^ 1] += 1
            ns.placed_ply.pop(d, None)
        del ns.board[d]

    # 2) suicide check on the moving group
    _, libs = _group_of(ns.board, pt, cfg.n)
    if not libs and not dead:
        return None

    # optional hard protection: a trooper with an adjacent friendly drone cannot be self-captured
    # (already alive here, so nothing to do; protection matters in enemy-capture step below)

    # 3) superko
    if cfg.superko:
        h = _hash(ns)
        if h in ns.history:
            return None

    # bookkeeping
    if kind == 'T':
        ns.reserve[mover] -= 1
        ns.placed_ply[pt] = ns.ply
    else:
        ns.drones_used[mover] += 1
    ns.passes = 0
    ns.to_move ^= 1
    ns.ply += 1
    if cfg.superko:
        ns.history.add(_hash_after(ns, s))

    # win checks
    if ns.lost_troopers[mover ^ 1] >= cfg.capture_to_win:
        ns.over = True; ns.winner = mover
    elif ns.ply >= cfg.max_plies:
        _finish_territory(ns)
    return ns


def apply_move(s: State, move: Move) -> State:
    ns = _try_apply(s, move)
    if ns is None:
        raise ValueError(f"illegal move {move}")
    return ns


# hard_trooper_protection variant: filter dead troopers that have a friendly adjacent drone
def _dead_enemy_groups_protected(ns, pt, mover, cfg):
    dead = _dead_enemy_groups(ns.board, pt, mover, cfg.n)
    if not cfg.hard_trooper_protection:
        return dead
    keep = set()
    for d in dead:
        if ns.board[d][1] == 'T':
            col = ns.board[d][0]
            if any(q in ns.board and ns.board[q] == (col, 'D') and q not in dead
                   for q in neighbors(d, cfg.n)):
                keep.add(d)
    return dead - keep


def _hash(s: State) -> int:
    return hash((frozenset(s.board.items()), s.to_move ^ 1))


def _hash_after(ns: State, prev: State) -> int:
    return hash((frozenset(ns.board.items()), ns.to_move))


# ---------------------------------------------------------------------------
# Territory (area) scoring
# ---------------------------------------------------------------------------
def area_score(s: State) -> tuple[float, float]:
    cfg = s.cfg
    stones = [0, 0]
    for (col, _k) in s.board.values():
        stones[col] += 1
    # flood empty regions; a region scores for a color iff it borders only that color
    seen: set[Point] = set()
    terr = [0, 0]
    for r in range(cfg.n):
        for c in range(cfg.n):
            p = (r, c)
            if p in s.board or p in seen:
                continue
            region = {p}; stack = [p]; borders: set[int] = set()
            seen.add(p)
            while stack:
                x = stack.pop()
                for q in neighbors(x, cfg.n):
                    if q in s.board:
                        borders.add(s.board[q][0])
                    elif q not in seen:
                        seen.add(q); region.add(q); stack.append(q)
            if len(borders) == 1:
                terr[borders.pop()] += len(region)
    a = stones[0] + terr[0]
    b = stones[1] + terr[1] + cfg.komi
    return a, b


def _finish_territory(s: State) -> None:
    a, b = area_score(s)
    s.over = True
    s.winner = 0 if a > b else (1 if b > a else -1)


# ---------------------------------------------------------------------------
# Setup variants -> initial State
# ---------------------------------------------------------------------------
def home_rows(color: int, n: int, k: int, rng: random.Random) -> list[Point]:
    """Pick k spread-out points on a player's back two rows."""
    rows = [0, 1] if color == 0 else [n - 1, n - 2]
    cells = [(r, c) for r in rows for c in range(n)]
    rng.shuffle(cells)
    # spread along columns
    step = max(1, n // max(1, k))
    chosen, used_cols = [], set()
    for (r, c) in sorted(cells, key=lambda x: x[1]):
        if len(chosen) >= k:
            break
        if all(abs(c - uc) >= step for uc in used_cols):
            chosen.append((r, c)); used_cols.add(c)
    while len(chosen) < k and cells:
        p = cells.pop()
        if p not in chosen:
            chosen.append(p)
    return chosen[:k]


SETUPS = {  # name -> number placed at home (rest go to reserve)
    "beachhead": 0, "vanguard": 1, "garrison": 2, "bastion": 3, "fog": -1,
}


def make_initial_state(cfg: RuleConfig, setup0="garrison", setup1="garrison",
                       seed: int = 0) -> State:
    rng = random.Random(seed)
    s = State(cfg=cfg, reserve=[cfg.troopers, cfg.troopers])
    for color, name in ((0, setup0), (1, setup1)):
        home = SETUPS[name]
        if home < 0:  # fog: random anywhere
            pts = []
            while len(pts) < cfg.troopers:
                p = (rng.randrange(cfg.n), rng.randrange(cfg.n))
                if p not in s.board and p not in pts:
                    pts.append(p)
        else:
            pts = home_rows(color, cfg.n, home, rng)
        for p in pts:
            s.board[p] = (color, 'T')
            s.reserve[color] -= 1
            s.placed_ply[p] = -10   # already "awake" at game start
    if cfg.superko:
        s.history.add(_hash(s) ^ 1)  # seed; exact value irrelevant
    return s
