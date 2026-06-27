"""Strategic runs: the eval is the strategy.

A bot's playstyle is a vector of weights over a small set of board FEATURES. Vary
the weights → vary the strategy (decap-hunter / territorial / expansionist /
squeeze). This module:

  1. extracts features and builds an eval from a weight vector (`make_eval`),
  2. wraps it in a fast 1-ply greedy player (`WeightGreedy`) — 1-ply expresses the
     eval cleanly and runs fast enough for big tournaments,
  3. seeds a few hand-built styles and scores them with diversity.py.

Two strategic loops build on this (see roadmap):
  • DIVERSITY  — coevolve a population scored on intransitivity (distinct styles
    that cycle-beat each other → varied watch / distinct bot personalities).
  • STRENGTH   — optimize one vector for win-rate → the Hard / Expert bot's eval.

Features (all from `me`'s view, opponent-relative):
  cap   decap progress      lost_troopers[opp] - lost_troopers[me]
  area  board + territory   area[me] - area[opp]
  cov   unique deploy reach coverage(me) - coverage(opp)   (ALL troopers)
  spc   trooper spacing     clump(opp) - clump(me)         (avoid contiguous troopers)
  prs   liberty pressure    enemy squeezed + / mine -      (trooper groups 8x)
"""
from __future__ import annotations
import random
from .engine import (RuleConfig, legal_moves, apply_move, area_score,
                     neighbors, cheb, _group_of)

FEATURE_NAMES = ["cap", "area", "cov", "spc", "prs"]


def coverage(s, color):
    n, R = s.cfg.n, s.cfg.drone_range
    troopers = [p for p, (c, k) in s.board.items() if c == color and k == 'T']
    if not troopers:
        return 0
    cells = set()
    for (tr, tc) in troopers:
        for r in range(max(0, tr - R), min(n - 1, tr + R) + 1):
            for c in range(max(0, tc - R), min(n - 1, tc + R) + 1):
                if (r, c) not in s.board:
                    cells.add((r, c))
    return len(cells)


def clump(s, color):
    """Orthogonally adjacent same-color trooper pairs (counted once)."""
    pairs = 0
    for p, (col, k) in s.board.items():
        if col == color and k == 'T':
            for q in neighbors(p, s.cfg.n):
                qq = s.board.get(q)
                if qq and qq[1] == 'T' and qq[0] == color and q > p:
                    pairs += 1
    return pairs


def pressure(s, me):
    opp = 1 - me
    seen, pr = set(), 0.0
    for p in list(s.board):
        if p in seen:
            continue
        col, _k = s.board[p]
        grp, libs = _group_of(s.board, p, s.cfg.n)
        has_t = any(s.board[g][1] == 'T' for g in grp)
        seen |= grp
        L = len(libs)
        danger = 1.0 if L <= 1 else 0.45 if L == 2 else 0.15 if L == 3 else 0.0
        if danger:
            w = 8 if has_t else 2
            pr += (w if col == opp else -w) * danger
    return pr


def features(s, me):
    opp = 1 - me
    a = area_score(s)
    return (
        (s.lost_troopers[opp] - s.lost_troopers[me]),
        (a[me] - a[opp]),
        (coverage(s, me) - coverage(s, opp)),
        (clump(s, opp) - clump(s, me)),
        pressure(s, me),
    )


def make_eval(weights):
    def ev(s, me):
        if s.over:
            return 1e6 if s.winner == me else (-1e6 if s.winner == 1 - me else 0.0)
        f = features(s, me)
        return sum(w * x for w, x in zip(weights, f))
    return ev


class WeightGreedy:
    """1-ply greedy over a weighted-feature eval. The weight vector IS the style."""
    def __init__(self, weights, seed=0, name="ws", pass_when_full=0.92):
        self.weights = tuple(weights)
        self.ev = make_eval(self.weights)
        self.rng = random.Random(seed)
        self.name = name
        self.pwf = pass_when_full

    def choose(self, s):
        me = s.to_move
        filled = len(s.board) / (s.cfg.n * s.cfg.n)
        best, bestm = None, ('pass',)
        for m in legal_moves(s):
            if m[0] == 'pass' and filled < self.pwf:
                continue
            ns = apply_move(s, m)
            v = self.ev(ns, me) + self.rng.random() * 1e-6
            if best is None or v > best:
                best, bestm = v, m
        return bestm


# weight order: [cap, area, cov, spc, prs]
SEED_STYLES = {
    "aggro":        (42, 1.0, 0.5, 3.0, 2.2),   # decap-hunter
    "territorial":  (18, 3.2, 0.6, 3.0, 0.5),   # hold ground
    "expansionist": (24, 1.2, 2.2, 3.0, 0.8),   # spread for reach
    "tactical":     (28, 1.0, 0.5, 3.0, 3.2),   # squeeze liberties
    "balanced":     (30, 1.5, 0.8, 3.0, 1.0),   # the current-ish hand eval
}


def seed_players(seed=0):
    return [WeightGreedy(w, seed=seed + i, name=n)
            for i, (n, w) in enumerate(SEED_STYLES.items())]


# A strategy is only meaningful against ONE fixed ruleset — encircle/mobile/range
# are different GAMES, so every strategic run pins one of these and holds it. Each
# distinct ruleset we ship/feature gets its OWN diversity + strength run.
CONFIGS = {
    # the base shipped game (both optional dials OFF) — what most players get
    "base-r4": RuleConfig(n=9, drone_range=4, troopers=5, capture_to_win=3),
    "base-r3": RuleConfig(n=9, drone_range=3, troopers=5, capture_to_win=3),
    # the distinctive "its own game" config — only if we feature it
    "mobile":  RuleConfig(n=9, drone_range=3, troopers=5, capture_to_win=3,
                          mobile_drones=True, drone_move_mode='step'),
}


def main():
    import sys
    from .diversity import diversity_report, diversity_objective
    which = sys.argv[1] if len(sys.argv) > 1 else "base-r4"
    gpp = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    cfg = CONFIGS[which]
    players = seed_players()
    names = list(SEED_STYLES)
    print(f"# PoC tournament  ruleset={which} (encircle={cfg.trooper_encircle} "
          f"mobile={cfg.mobile_drones} n={cfg.n} R{cfg.drone_range} cap{cfg.capture_to_win}) "
          f"games/pair={gpp} styles={names}", flush=True)
    rep = diversity_report(cfg, players, setups=["beachhead"] * len(players),
                           games_per_pair=gpp)
    obj = diversity_objective(rep)
    print("payoff W[i][j] = win-rate of row vs col:")
    print("        " + "  ".join(f"{nm[:5]:>5}" for nm in names))
    for i, row in enumerate(rep["W"]):
        print(f"  {names[i][:6]:>6} " + "  ".join(f"{v:5.2f}" for v in row))
    print(f"\nintransitivity = {rep['intransitivity']}   (push UP — cyclic/RPS share)")
    print(f"dominance      = {rep['dominance']}   (best win-rate any style gets vs field; push DOWN)")
    print(f"nash support   = {rep.get('nash_support')}   entropy = {rep.get('nash_entropy')}")
    print(f"diversity score= {obj['score']}")


if __name__ == "__main__":
    main()
