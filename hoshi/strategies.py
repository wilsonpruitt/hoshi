"""Playstyle policies.

A strategy here is a *way of playing*, not an opening. Each StyleBot is a 1-ply
greedy over a weighted feature eval; the weight vector defines the style. Swap in
the MCTS wrapper (mcts.py, TODO) for trustworthy diversity numbers — heuristic
bots measure the heuristics' blind spots as much as the game's structure.

Features are all CHEAP (no flood fill of territory per node): capture pressure,
own-trooper safety (atari), board presence, attack proximity, and range coverage
(a territory proxy). Coverage stands in for area so eval stays fast enough to run
thousands of games per sweep cell.
"""
from __future__ import annotations
import random
from .engine import State, Move, legal_moves, apply_move, _group_of, neighbors, cheb


def _features(s: State, me: int) -> dict:
    opp = 1 - me
    stones = [0, 0]
    for (col, _k) in s.board.values():
        stones[col] += 1
    # own trooper atari (groups with <=1 liberty are in danger)
    atari = 0
    seen = set()
    my_troopers, opp_troopers = [], []
    for p, (col, k) in s.board.items():
        if k == 'T':
            (my_troopers if col == me else opp_troopers).append(p)
        if col == me and k == 'T' and p not in seen:
            grp, libs = _group_of(s.board, p, s.cfg.n)
            seen |= grp
            if len(libs) <= 1:
                atari += 1
    # attack proximity: how close my pieces sit to enemy troopers (closing for the kill)
    attack = 0.0
    if opp_troopers:
        my_pieces = [p for p, (col, _k) in s.board.items() if col == me]
        for et in opp_troopers:
            if my_pieces:
                d = min(cheb(p, et) for p in my_pieces)
                attack += max(0, s.cfg.n - d)
    # coverage: empty points within range of a ready trooper (territory proxy)
    ready = s.my_ready_troopers(me)
    cov = 0
    if ready:
        for r in range(s.cfg.n):
            for c in range(s.cfg.n):
                if (r, c) not in s.board and any(cheb((r, c), t) <= s.cfg.drone_range for t in ready):
                    cov += 1
    return {
        "cap": s.lost_troopers[opp] - s.lost_troopers[me],
        "stones": stones[me] - stones[opp],
        "atari": atari,
        "attack": attack,
        "cover": cov,
    }


class StyleBot:
    def __init__(self, name, weights, seed=0, drop_bias="neutral"):
        self.name = name
        self.w = weights
        self.rng = random.Random(seed)
        self.drop_bias = drop_bias        # 'deep' | 'home' | 'neutral'

    def _eval(self, s: State, me: int) -> float:
        if s.over:
            return 1e6 if s.winner == me else (-1e6 if s.winner == 1 - me else 0.0)
        f = _features(s, me)
        return (self.w["cap"] * f["cap"] + self.w["stones"] * f["stones"]
                - self.w["atari"] * f["atari"] + self.w["attack"] * f["attack"]
                + self.w["cover"] * f["cover"])

    def _drop_score(self, s: State, pt) -> float:
        if self.drop_bias == "neutral":
            return 0.0
        home_row = 0 if s.to_move == 0 else s.cfg.n - 1
        depth = abs(pt[0] - home_row) / (s.cfg.n - 1)   # 0 = home, 1 = far side
        return 1.5 * depth if self.drop_bias == "deep" else 1.5 * (1 - depth)

    def choose(self, s: State) -> Move:
        me = s.to_move
        filled = len(s.board) / (s.cfg.n * s.cfg.n)
        best, best_m = None, ('pass',)
        for m in legal_moves(s):
            if m[0] == 'pass' and filled < 0.9:
                continue
            try:
                ns = apply_move(s, m)
            except ValueError:
                continue
            v = self._eval(ns, me) + self.rng.random() * 0.01
            if m[0] == 'drop_t':
                v += self._drop_score(s, m[1])
            if best is None or v > best:
                best, best_m = v, m
        return best_m


# Weight vectors. Tuned to be *different*, not perfectly balanced — that's the point.
STYLE_WEIGHTS = {
    "aggressive":   dict(cap=40, stones=0.5, atari=2, attack=0.8, cover=0.1),
    "territorial":  dict(cap=12, stones=1.0, atari=6, attack=0.0, cover=0.7),
    "balanced":     dict(cap=25, stones=1.0, atari=4, attack=0.3, cover=0.4),
    "expansionist": dict(cap=10, stones=0.6, atari=3, attack=0.1, cover=1.0),
}
STYLE_DROP_BIAS = {"aggressive": "deep", "territorial": "home",
                   "balanced": "neutral", "expansionist": "neutral"}


def make_styles(seed=0):
    return [StyleBot(name, STYLE_WEIGHTS[name], seed, STYLE_DROP_BIAS[name])
            for name in STYLE_WEIGHTS]


class StyleEval:
    """Picklable (state, me) -> float: a StyleBot's weighted-feature eval, usable
    as an `MCTSPlayer.eval_fn` so the search itself adopts a playstyle. A plain
    top-level class (not a closure) so it survives multiprocessing pickling."""
    def __init__(self, weights: dict):
        self.w = weights

    def __call__(self, s: State, me: int) -> float:
        if s.over:
            return 1e6 if s.winner == me else (-1e6 if s.winner == 1 - me else 0.0)
        f = _features(s, me)
        w = self.w
        return (w["cap"] * f["cap"] + w["stones"] * f["stones"] - w["atari"] * f["atari"]
                + w["attack"] * f["attack"] + w["cover"] * f["cover"])


def make_style_mcts(seed=0, sims=80, action_width=8, rollout_depth=40):
    """The StyleBots, but each one is an MCTS search toward its style's eval.
    Use these (not the 1-ply StyleBots) for trustworthy diversity numbers."""
    from .mcts import MCTSPlayer
    return [MCTSPlayer(sims=sims, seed=seed, action_width=action_width,
                       rollout_depth=rollout_depth, eval_fn=StyleEval(STYLE_WEIGHTS[name]),
                       name=name)
            for name in STYLE_WEIGHTS]
