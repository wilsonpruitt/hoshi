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
import random, math
from multiprocessing import Pool
import numpy as np
from .engine import (RuleConfig, legal_moves, apply_move, area_score,
                     neighbors, cheb, _group_of)

FEATURE_NAMES = ["cap", "area", "cov", "spc", "prs", "res"]


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


def fast_area_diff(s, me):
    """Cheap stand-in for area_score during the greedy search: stone count minus the
    lost-trooper penalty, skipping the O(N^2) territory flood-fill (territory only
    resolves late-game). ~5-10x faster per candidate move, which makes the GA viable."""
    opp = 1 - me
    sm = so = 0
    for (c, _k) in s.board.values():
        if c == me:
            sm += 1
        else:
            so += 1
    pen = s.cfg.trooper_loss_penalty
    return (sm - pen * s.lost_troopers[me]) - (so - pen * s.lost_troopers[opp])


def features(s, me):
    opp = 1 - me
    # reserve: value holding a SMALL reserve (up to 2) to replace fallen troopers
    # later — capped so the bot still deploys most of them for board presence.
    res = min(s.reserve[me], 2) - min(s.reserve[opp], 2)
    return (
        (s.lost_troopers[opp] - s.lost_troopers[me]),
        fast_area_diff(s, me),
        (coverage(s, me) - coverage(s, opp)),
        (clump(s, opp) - clump(s, me)),
        pressure(s, me),
        res,
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


# weight order: [cap, area, cov, spc, prs, res]
SEED_STYLES = {
    "aggro":        (42, 1.0, 0.5, 3.0, 2.2, 1.0),   # decap-hunter, dumps troopers
    "territorial":  (18, 3.2, 0.6, 3.0, 0.5, 3.0),   # hold ground, keep reserve
    "expansionist": (24, 1.2, 2.2, 3.0, 0.8, 2.0),   # spread for reach
    "tactical":     (28, 1.0, 0.5, 3.0, 3.2, 1.5),   # squeeze liberties
    "balanced":     (30, 1.5, 0.8, 3.0, 1.0, 2.0),   # the current-ish hand eval
}


def seed_players(seed=0):
    return [WeightGreedy(w, seed=seed + i, name=n)
            for i, (n, w) in enumerate(SEED_STYLES.items())]


# ---------------------------------------------------------------------------
# Coevolution: search the weight-space for a DIVERSE population (cycle-beat,
# no single dominant) — measured by intransitivity / dominance. Greedy players
# (fast) for the search; MCTS-validate the survivors afterwards.
# ---------------------------------------------------------------------------
# weight order:        [cap,  area, cov,  spc, prs, res]
WMIN = np.array([8.0,  0.3, 0.2, 0.0, 0.3, 0.0])
WMAX = np.array([50.0, 4.0, 2.6, 5.0, 4.0, 6.0])


def random_weights(rng):
    return [round(rng.uniform(lo, hi), 3) for lo, hi in zip(WMIN, WMAX)]


def mutate(w, rng, rate=0.5, scale=0.25):
    out = list(w)
    for k in range(len(out)):
        if rng.random() < rate:
            out[k] *= math.exp(rng.gauss(0, scale))          # multiplicative jiggle
        out[k] = float(min(WMAX[k], max(WMIN[k], out[k])))
    return [round(x, 3) for x in out]


def crossover(a, b, rng):
    return [a[k] if rng.random() < 0.5 else b[k] for k in range(len(a))]


def _match(arg):
    """One game between two weight vectors; returns the winner (0/1/-1)."""
    wi, wj, cfg, su_i, su_j, seed = arg
    from .harness import play_game
    r = play_game(cfg, WeightGreedy(wi, seed=seed), WeightGreedy(wj, seed=seed + 1),
                  su_i, su_j, seed=seed)
    return r.winner


def payoff_matrix_par(cfg, pop, setups, gpp, seed=0, workers=3):
    n = len(pop)
    jobs, meta = [], []
    for i in range(n):
        for j in range(n):
            for g in range(gpp):
                jobs.append((pop[i], pop[j], cfg, setups[i], setups[j], seed + g)); meta.append((i, j, 0))
                jobs.append((pop[j], pop[i], cfg, setups[j], setups[i], seed + 500 + g)); meta.append((i, j, 1))
    with Pool(workers) as pool:
        res = pool.map(_match, jobs)
    W = np.zeros((n, n)); cnt = np.zeros((n, n))
    for (i, j, swap), winner in zip(meta, res):
        win = (winner == 0) if swap == 0 else (winner == 1)
        W[i][j] += 1.0 if win else (0.5 if winner == -1 else 0.0); cnt[i][j] += 1
    return W / np.maximum(cnt, 1)


def diversity_metrics(W):
    from .diversity import hodge, alpha_rank, dominance, nash_zero_sum, entropy
    P = (W + 1 - W.T) / 2                 # symmetrize (per diversity.py: W[i][j],W[j][i] are different games)
    h = hodge(P)
    dom, _ = dominance(W)
    nash = nash_zero_sum(W - W.T)
    pi = alpha_rank(W)
    return {"intransitivity": round(h["intransitivity"], 3),
            "dominance": round(dom, 3),
            "nash_support": int((nash > 0.02).sum()),
            "alpha_entropy": round(entropy(pi), 3),
            "alpha": pi}


def pop_objective(m):
    # push intransitivity UP, punish a dominant style (>0.5 win-rate vs field)
    return round(m["intransitivity"] - 0.6 * max(0.0, m["dominance"] - 0.5), 4)


def individual_fitness(W):
    """Reward distinct (novel) AND cyclic (neither dominant nor dominated) members."""
    n = len(W)
    avg = W.mean(axis=1)                                   # avg win-rate vs field
    fit = []
    for i in range(n):
        novelty = np.mean([np.abs(W[i] - W[j]).mean() for j in range(n) if j != i]) if n > 1 else 0.0
        cyclic = min(avg[i], 1 - avg[i])                   # ~0.5 = beats some, loses to some
        fit.append(novelty + cyclic)
    return fit


def coevolve(which="base-r4", pop_size=8, gens=5, gpp=3, workers=3, seed=0, max_plies=120):
    import time
    from dataclasses import replace
    cfg = replace(CONFIGS[which], max_plies=max_plies)   # cap game length: greedy is slow,
    rng = random.Random(seed)                            # and capped territory scoring is fine for ranking
    pop = [list(w) for w in SEED_STYLES.values()]
    while len(pop) < pop_size:
        pop.append(random_weights(rng))
    pop = pop[:pop_size]
    setups = ["beachhead"] * pop_size
    best = None
    print(f"# coevolve ruleset={which} pop={pop_size} gens={gens} games/pair={gpp} workers={workers}", flush=True)
    for gen in range(gens):
        t0 = time.time()
        W = payoff_matrix_par(cfg, [tuple(w) for w in pop], setups, gpp, seed=seed + gen * 1000, workers=workers)
        m = diversity_metrics(W); score = pop_objective(m)
        if best is None or score > best["score"]:
            best = {"score": score, "pop": [list(w) for w in pop], "metrics": m, "W": W, "gen": gen}
        print(f"  gen {gen}: intransitivity={m['intransitivity']} dominance={m['dominance']} "
              f"nash_support={m['nash_support']} obj={score}  [{time.time()-t0:.0f}s]", flush=True)
        # selection + breeding
        fit = individual_fitness(W)
        order = sorted(range(pop_size), key=lambda i: -fit[i])
        keep = [pop[i] for i in order[:max(2, pop_size // 2)]]
        newpop = [list(w) for w in keep]
        while len(newpop) < pop_size:
            if rng.random() < 0.25:
                newpop.append(random_weights(rng))                 # immigrant (exploration)
            else:
                a, b = rng.sample(keep, 2)
                newpop.append(mutate(crossover(a, b, rng), rng))
        pop = newpop
    return best


# A strategy is only meaningful against ONE fixed ruleset — encircle/mobile/range
# are different GAMES, so every strategic run pins one of these and holds it. Each
# distinct ruleset we ship/feature gets its OWN diversity + strength run.
# LOCKED canonical ruleset for strategic runs: base-r4 = the live shipped default
# (9x9, range 4, both optional dials off), so learned styles/eval transfer straight
# into the game. mobile/encircle, if featured, get their own runs.
CANON = "base-r4"
CONFIGS = {
    # the base shipped game (both optional dials OFF) — what most players get
    "base-r4": RuleConfig(n=9, drone_range=4, troopers=5, capture_to_win=3, trooper_loss_penalty=4),
    "base-r3": RuleConfig(n=9, drone_range=3, troopers=5, capture_to_win=3, trooper_loss_penalty=4),
    # the distinctive "its own game" config — only if we feature it
    "mobile":  RuleConfig(n=9, drone_range=3, troopers=5, capture_to_win=3,
                          mobile_drones=True, drone_move_mode='step'),
}


def _winrate_vs_panel(arg):
    """win-rate of weight vector w vs every panel member, color-swapped."""
    w, panel, cfg, gpp, seed = arg
    from .harness import play_game
    wins = games = 0
    for p in panel:
        for g in range(gpp):
            r = play_game(cfg, WeightGreedy(w, seed=seed + g), WeightGreedy(p, seed=seed + 100 + g),
                          "beachhead", "beachhead", seed=seed + g)
            wins += 1 if r.winner == 0 else (0.5 if r.winner == -1 else 0); games += 1
            r2 = play_game(cfg, WeightGreedy(p, seed=seed + 200 + g), WeightGreedy(w, seed=seed + 300 + g),
                           "beachhead", "beachhead", seed=seed + 500 + g)
            wins += 1 if r2.winner == 1 else (0.5 if r2.winner == -1 else 0); games += 1
    return wins / games


def strength(which="base-r4", iters=18, lam=6, gpp=4, workers=3, seed=0, max_plies=120):
    """(1+lambda) hill-climb: evolve ONE weight vector to maximize win-rate vs a
    fixed seed panel (a stable yardstick). Output = a strong, FAST 1-ply bot."""
    import time
    from dataclasses import replace
    cfg = replace(CONFIGS[which], max_plies=max_plies); rng = random.Random(seed)
    panel = [tuple(w) for w in SEED_STYLES.values()]               # fixed strength yardstick
    champ = list(SEED_STYLES["balanced"])
    print(f"# strength ruleset={which} iters={iters} lambda={lam} games/opp={gpp}", flush=True)

    def eval_many(cands, it):
        jobs = [(tuple(c), panel, cfg, gpp, seed + it * 9000 + k * 13) for k, c in enumerate(cands)]
        with Pool(workers) as pool:
            return pool.map(_winrate_vs_panel, jobs)

    champ_wr = eval_many([champ], 0)[0]
    print(f"  start: balanced wr={champ_wr:.3f}", flush=True)
    for it in range(iters):
        t0 = time.time()
        mutants = [mutate(champ, rng, rate=0.6, scale=0.3) for _ in range(lam)]
        wrs = eval_many(mutants, it + 1)
        bi = max(range(lam), key=lambda k: wrs[k])
        improved = wrs[bi] > champ_wr
        if improved:
            champ, champ_wr = mutants[bi], wrs[bi]
        print(f"  iter {it}: best mutant wr={wrs[bi]:.3f} champ wr={champ_wr:.3f} "
              f"{'*' if improved else ' '}  [{time.time()-t0:.0f}s]", flush=True)
    return {"weights": [round(x, 3) for x in champ], "winrate": round(champ_wr, 3)}


def describe(w):
    """Name a style by which feature it leans on, relative to the balanced baseline."""
    base = SEED_STYLES["balanced"]
    ratios = [w[k] / base[k] if base[k] else 0 for k in range(len(w))]
    lead = max(range(len(w)), key=lambda k: ratios[k])
    tag = {0: "decap-hunter", 1: "territorial", 2: "expansionist", 3: "spacing",
           4: "squeeze", 5: "reserve-keeper"}[lead]
    return tag


def main():
    import sys
    from .diversity import diversity_report, diversity_objective
    mode = sys.argv[1] if len(sys.argv) > 1 else "base-r4"

    if mode == "strength":
        which = sys.argv[2] if len(sys.argv) > 2 else CANON
        iters = int(sys.argv[3]) if len(sys.argv) > 3 else 18
        res = strength(which=which, iters=iters)
        w = res["weights"]
        print(f"\nSTRONGEST 1-ply weights ({describe(w)}):")
        print(f"  [cap {w[0]:.1f}, area {w[1]:.2f}, cov {w[2]:.2f}, spc {w[3]:.2f}, prs {w[4]:.2f}, res {w[5]:.2f}]")
        print(f"  win-rate vs seed panel = {res['winrate']}  (a FAST bot — 1-ply, no MCTS lag)")
        return

    if mode == "evolve":
        from dataclasses import replace
        which = sys.argv[2] if len(sys.argv) > 2 else CANON
        gens = int(sys.argv[3]) if len(sys.argv) > 3 else 6
        gpp = int(sys.argv[4]) if len(sys.argv) > 4 else 3
        best = coevolve(which=which, pop_size=6, gens=gens, gpp=gpp, max_plies=80, workers=2)
        # confirmation of the winning population at higher fidelity (per-gen metrics
        # use few games/pair and are noisy).
        cfg = replace(CONFIGS[which], max_plies=80)
        pop = [tuple(w) for w in best["pop"]]
        Wc = payoff_matrix_par(cfg, pop, ["beachhead"] * len(pop), gpp=8, workers=2, seed=777)
        mc = diversity_metrics(Wc)
        print(f"\nBEST population (gen {best['gen']}) — CONFIRMED at gpp=12: "
              f"intransitivity={mc['intransitivity']}  dominance={mc['dominance']}  "
              f"nash_support={mc['nash_support']}")
        pi = mc["alpha"]
        avg = Wc.mean(axis=1)
        for i in sorted(range(len(pop)), key=lambda i: -pi[i]):
            w = best["pop"][i]
            mark = "*" if pi[i] > 0.05 else " "
            print(f" {mark} alpha={pi[i]:.3f} wr={avg[i]:.2f} {describe(w):>14}  "
                  f"[cap {w[0]:.1f}, area {w[1]:.2f}, cov {w[2]:.2f}, spc {w[3]:.2f}, prs {w[4]:.2f}, res {w[5]:.2f}]")
        print("(* = carries equilibrium mass; wr = win-rate vs the population — want a "
              "spread of distinct styles all near 0.5, not one runaway)")
        return

    which = mode
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
