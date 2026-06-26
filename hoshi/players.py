"""Baseline players. The greedy bot doubles as an MCTS rollout policy later."""
from __future__ import annotations
import random
from .engine import State, Move, legal_moves, apply_move, area_score


class RandomPlayer:
    name = "random"
    def __init__(self, seed=0):
        self.rng = random.Random(seed)
    def choose(self, s: State) -> Move:
        moves = [m for m in legal_moves(s) if m[0] != 'pass']
        return self.rng.choice(moves) if moves else ('pass',)


def _evaluate(s: State, me: int) -> float:
    """Cheap static eval from `me`'s perspective. Not strong — a baseline."""
    if s.over:
        if s.winner == me:   return 1e6
        if s.winner == 1-me: return -1e6
        return 0.0
    # captured-trooper pressure dominates; then board presence; then trooper safety
    cap = 30 * (s.lost_troopers[1-me] - s.lost_troopers[me])
    stones = [0, 0]
    atari = 0
    from .engine import _group_of
    seen = set()
    for p, (col, k) in s.board.items():
        stones[col] += 1
    # crude atari count for my troopers (troopers with a single liberty are in danger)
    for p, (col, k) in s.board.items():
        if col == me and k == 'T' and p not in seen:
            grp, libs = _group_of(s.board, p, s.cfg.n)
            seen |= grp
            if len(libs) <= 1:
                atari += 1
    return cap + (stones[me] - stones[1-me]) - 4 * atari


class GreedyPlayer:
    name = "greedy"
    def __init__(self, seed=0, pass_when_full=0.92):
        self.rng = random.Random(seed)
        self.pass_when_full = pass_when_full
    def choose(self, s: State) -> Move:
        me = s.to_move
        moves = legal_moves(s)
        # consider passing only when the board is nearly full (lets territory resolve)
        filled = len(s.board) / (s.cfg.n * s.cfg.n)
        scored = []
        for m in moves:
            if m[0] == 'pass' and filled < self.pass_when_full:
                continue
            try:
                ns = apply_move(s, m)
            except ValueError:
                continue
            scored.append((_evaluate(ns, me) + self.rng.random() * 0.01, m))
        if not scored:
            return ('pass',)
        scored.sort(reverse=True)
        return scored[0][1]
