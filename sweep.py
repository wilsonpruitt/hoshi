"""MCTS-backed sweeps — the trustworthy version of the demos.

The demos measure with weak 1-ply bots; CLAUDE.md's trust gate says re-run with
MCTS before believing any ordering. This does that, in two parts:

  A. BALANCE SWEEP — MCTS self-play across rule dials, ranked by
     `balance_objective` (lower = fairer/healthier game).
  B. DIVERSITY RE-RUN — the same config measured twice, greedy StyleBots vs
     style-MCTS, to see whether the "one playstyle dominates" collapse from the
     greedy demo survives strong play, or was just the bots' blind spot.

Parallelized with multiprocessing (engine state is plain data → picklable).
Workers capped low — this targets an 8 GB machine. Boards/sims kept small so it
finishes in minutes; raise N_WORKERS, sims, games, and n for sharper numbers.

    python3 sweep.py
"""
from __future__ import annotations
import os
import time
import multiprocessing as mp

import numpy as np

from hoshi.engine import RuleConfig
from hoshi.players import GreedyPlayer
from hoshi.mcts import MCTSPlayer
from hoshi.strategies import make_styles, make_style_mcts
from hoshi.harness import play_game, summarize, balance_objective, GameRecord
from hoshi.diversity import (nash_zero_sum, alpha_rank, hodge, entropy,
                             dominance, diversity_objective)

N_WORKERS = min(4, max(1, (os.cpu_count() or 2) - 2))


# ---------------------------------------------------------------------------
# Parallel game runner. Each task is fully picklable (cfg + player objects).
# ---------------------------------------------------------------------------
def _run_task(task):
    cfg, p0, p1, s0, s1, seed, tag = task
    rec = play_game(cfg, p0, p1, s0, s1, seed=seed)
    return tag, rec.winner, rec.win_kind, rec.plies


def _run_all(tasks):
    if N_WORKERS == 1:
        return [_run_task(t) for t in tasks]
    with mp.Pool(N_WORKERS) as pool:
        return pool.map(_run_task, tasks, chunksize=1)


# ---------------------------------------------------------------------------
# A. Balance sweep — MCTS self-play
# ---------------------------------------------------------------------------
def balance_sweep(grid, n_games=16, sims=80, n=7, troopers=4, max_plies=160,
                  drone_supply=9999):
    print("=" * 72)
    print(f"A. BALANCE SWEEP — MCTS self-play (n={n}, sims={sims}, "
          f"supply={drone_supply}, {n_games} games/cell, {N_WORKERS} workers)")
    print("   lower balance_objective = fairer & both win-paths alive")
    print("=" * 72)
    tasks, meta = [], []
    for (dr, ctw) in grid:
        cfg = RuleConfig(n=n, drone_range=dr, troopers=troopers,
                         capture_to_win=ctw, max_plies=max_plies,
                         drone_supply=drone_supply)
        cell = len(meta)
        for g in range(n_games):
            # fresh seeds per game; swap which seed leads to balance first-player
            p0 = MCTSPlayer(sims=sims, seed=g)
            p1 = MCTSPlayer(sims=sims, seed=1000 + g)
            tasks.append((cfg, p0, p1, "garrison", "garrison", g, cell))
        meta.append((dr, ctw, cfg))

    results = _run_all(tasks)
    by_cell: dict[int, list] = {i: [] for i in range(len(meta))}
    for tag, winner, kind, plies in results:
        by_cell[tag].append(GameRecord(winner, kind, plies, (0, 0)))

    print(f"\n{'range':>5} {'capWin':>6} {'p0win':>6} {'decap':>6} "
          f"{'avgPly':>6} {'object':>7}")
    rows = []
    for i, (dr, ctw, _cfg) in enumerate(meta):
        s = summarize(by_cell[i])
        obj = balance_objective(s)
        rows.append((obj, dr, ctw, s))
        print(f"{dr:>5} {ctw:>6} {s['p0_winrate']:>6.2f} "
              f"{s['decap_share']:>6.2f} {s['avg_plies']:>6.1f} {obj:>7.3f}")
    best = min(rows, key=lambda r: r[0])
    print(f"\n  -> healthiest: range={best[1]} capture_to_win={best[2]} "
          f"(objective {best[0]:.3f})")
    return rows


# ---------------------------------------------------------------------------
# B. Diversity re-run — greedy StyleBots vs style-MCTS
# ---------------------------------------------------------------------------
def _payoff_parallel(cfg, policies, games_per_pair, seed):
    """W[i][j] = winrate of policy i (p0) vs j (p1), color-swap averaged. Same
    contract as diversity.build_payoff_matrix, but games run in the pool."""
    n = len(policies)
    tasks = []
    for i in range(n):
        for j in range(n):
            for g in range(games_per_pair):
                tasks.append((cfg, policies[i], policies[j], "garrison", "garrison",
                              seed + g, (i, j, 0)))
                tasks.append((cfg, policies[j], policies[i], "garrison", "garrison",
                              seed + 500 + g, (i, j, 1)))
    results = _run_all(tasks)
    wins = np.zeros((n, n)); games = np.zeros((n, n))
    for (i, j, which), winner, _kind, _plies in results:
        # which==0: i is p0, credit i on winner==0. which==1: i is p1, credit on winner==1.
        good = 0 if which == 0 else 1
        wins[i][j] += 1.0 if winner == good else (0.5 if winner == -1 else 0.0)
        games[i][j] += 1
    return wins / games


def _report_from_W(names, W):
    P = (W + 1.0 - W.T) / 2.0
    np.fill_diagonal(P, 0.5)
    A = 2 * P - 1
    nash = nash_zero_sum(A)
    arank = alpha_rank(P)
    h = hodge(A)
    dom, dom_i = dominance(P)
    n = len(names); logn = np.log(n)
    return dict(
        names=names, W=W,
        nash=nash, nash_support=int((nash > 0.05).sum()),
        nash_entropy=round(entropy(nash) / logn, 3),
        arank_entropy=round(entropy(arank) / logn, 3),
        intransitivity=round(h["intransitivity"], 3),
        transitivity=round(h["transitivity"], 3),
        dominance=round(dom, 3), dominant=names[dom_i],
    )


def _show(tag, rep):
    obj = diversity_objective(rep)
    print(f"\n  [{tag}]")
    print("    payoff W (row beats col):  " + " ".join(f"{n[:6]:>7}" for n in rep["names"]))
    for i, nm in enumerate(rep["names"]):
        print(f"    {nm[:12]:12} " + " ".join(f"{rep['W'][i][j]:>7.2f}"
                                              for j in range(len(rep["names"]))))
    print(f"    Nash support/entropy   {rep['nash_support']} / {rep['nash_entropy']}"
          f"     alpha-Rank entropy {rep['arank_entropy']}")
    print(f"    intransitivity(cyclic) {rep['intransitivity']}"
          f"     dominance {rep['dominance']} -> {rep['dominant']}")
    print(f"    DIVERSITY OBJECTIVE    {obj['score']}   {obj}")


def diversity_rerun(cfg, games_per_pair=3, sims=80, seed=1):
    print("\n" + "=" * 72)
    print(f"B. DIVERSITY RE-RUN — same config, weak vs strong bots "
          f"({games_per_pair} games/pair, {N_WORKERS} workers)")
    print(f"   cfg: n={cfg.n} range={cfg.drone_range} troopers={cfg.troopers} "
          f"capture_to_win={cfg.capture_to_win}")
    print("   Question: does the greedy demo's strategy collapse survive MCTS?")
    print("=" * 72)

    greedy = make_styles(seed=0)
    names = [p.name for p in greedy]
    Wg = _payoff_parallel(cfg, greedy, games_per_pair, seed)
    _show("greedy StyleBots (the demo's bots)", _report_from_W(names, Wg))

    strong = make_style_mcts(seed=0, sims=sims)
    Wm = _payoff_parallel(cfg, strong, games_per_pair, seed)
    _show(f"style-MCTS (sims={sims})", _report_from_W(names, Wm))


def supply_sweep(base_cfg, supplies, games_per_pair=3, sims=80, seed=1):
    """Does tightening drone scarcity push the game from a flat ladder toward a
    rock-paper-scissors cycle? Strong (style-MCTS) bots only; report cyclicity
    (intransitivity) + how many styles stay viable, per drone_supply value."""
    print("=" * 72)
    print(f"DRONE-SUPPLY SWEEP — style-MCTS (n={base_cfg.n} range={base_cfg.drone_range} "
          f"capWin={base_cfg.capture_to_win}, sims={sims}, {games_per_pair} games/pair)")
    print("   hypothesis: scarce drones force commitment -> cyclicity (cyclic) rises")
    print("=" * 72)
    from dataclasses import replace
    print(f"\n{'supply':>7} {'cyclic':>6} {'nashSup':>7} {'nashEnt':>7} "
          f"{'domin':>6} {'object':>7}  dominant")
    rows = []
    for sup in supplies:
        cfg = replace(base_cfg, drone_supply=sup)
        strong = make_style_mcts(seed=0, sims=sims)
        names = [p.name for p in strong]
        W = _payoff_parallel(cfg, strong, games_per_pair, seed)
        rep = _report_from_W(names, W)
        obj = diversity_objective(rep)
        label = "inf" if sup >= 9999 else str(sup)
        rows.append((label, rep))
        print(f"{label:>7} {rep['intransitivity']:>6.2f} {rep['nash_support']:>7} "
              f"{rep['nash_entropy']:>7.2f} {rep['dominance']:>6.2f} "
              f"{obj['score']:>7.3f}  {rep['dominant']}")
    print("\n  rising 'cyclic' as supply drops = scarcity is creating the cycle.")
    return rows


def setup_diversity(cfg, setups=None, games_per_pair=4, sims=80, seed=1):
    """Do the *designed* setup variants cycle? Payoff matrix over the setups
    using one neutral strong bot (same eval both sides) so the only thing that
    varies is the structural commitment each setup encodes. Free diversity if
    they rock-paper-scissors; a ladder means the setups need rebalancing."""
    setups = setups or ["beachhead", "vanguard", "garrison", "bastion", "fog"]
    print("=" * 72)
    print(f"SETUP-VARIANT DIVERSITY — neutral strong bot (n={cfg.n} range={cfg.drone_range} "
          f"capWin={cfg.capture_to_win} supply={cfg.drone_supply}, sims={sims}, "
          f"{games_per_pair} games/pair)")
    print("   varying only the setup; do the designed variants counter each other?")
    print("=" * 72)
    n = len(setups)
    tasks = []
    for i in range(n):
        for j in range(n):
            for g in range(games_per_pair):
                tasks.append((cfg, MCTSPlayer(sims=sims, seed=g),
                              MCTSPlayer(sims=sims, seed=1000 + g),
                              setups[i], setups[j], seed + g, (i, j, 0)))
                tasks.append((cfg, MCTSPlayer(sims=sims, seed=2000 + g),
                              MCTSPlayer(sims=sims, seed=3000 + g),
                              setups[j], setups[i], seed + 500 + g, (i, j, 1)))
    results = _run_all(tasks)
    wins = np.zeros((n, n)); games = np.zeros((n, n))
    for (i, j, which), winner, _k, _p in results:
        good = 0 if which == 0 else 1
        wins[i][j] += 1.0 if winner == good else (0.5 if winner == -1 else 0.0)
        games[i][j] += 1
    W = wins / games
    rep = _report_from_W(setups, W)
    _show("setup variants (neutral strong bot)", rep)
    print()
    return rep


def combined_diversity(cfg, styles, setups, games_per_pair=3, sims=80, seed=1):
    """Diversity of the WHOLE game: every (playstyle x setup) combo is one
    strategy. Does crossing the two axes give more viable distinct ways to play
    than either alone? Strong style-MCTS bots; setup is the game's init."""
    from hoshi.strategies import STYLE_WEIGHTS, StyleEval
    strat = [(st, su) for st in styles for su in setups]
    labels = [f"{st[:3]}+{su[:3]}" for (st, su) in strat]
    n = len(strat)
    print("=" * 72)
    print(f"COMBINED STRATEGY SPACE — {len(styles)} styles x {len(setups)} setups "
          f"= {n} strategies (n={cfg.n} range={cfg.drone_range} supply={cfg.drone_supply}, "
          f"sims={sims}, {games_per_pair} games/pair)")
    print("=" * 72)
    tasks = []
    for i, (si, sui) in enumerate(strat):
        for j, (sj, suj) in enumerate(strat):
            for g in range(games_per_pair):
                tasks.append((cfg, MCTSPlayer(sims=sims, seed=g, eval_fn=StyleEval(STYLE_WEIGHTS[si])),
                              MCTSPlayer(sims=sims, seed=1000 + g, eval_fn=StyleEval(STYLE_WEIGHTS[sj])),
                              sui, suj, seed + g, (i, j, 0)))
                tasks.append((cfg, MCTSPlayer(sims=sims, seed=2000 + g, eval_fn=StyleEval(STYLE_WEIGHTS[sj])),
                              MCTSPlayer(sims=sims, seed=3000 + g, eval_fn=StyleEval(STYLE_WEIGHTS[si])),
                              suj, sui, seed + 500 + g, (i, j, 1)))
    results = _run_all(tasks)
    wins = np.zeros((n, n)); games = np.zeros((n, n))
    for (i, j, which), winner, _k, _p in results:
        good = 0 if which == 0 else 1
        wins[i][j] += 1.0 if winner == good else (0.5 if winner == -1 else 0.0)
        games[i][j] += 1
    W = wins / games
    rep = _report_from_W(labels, W)
    obj = diversity_objective(rep)

    P = (W + 1.0 - W.T) / 2.0; np.fill_diagonal(P, 0.5)
    h = hodge(2 * P - 1)
    order = np.argsort(-h["ratings"])
    nash = rep["nash"]

    print("\n  Skill spine (HodgeRank) + Nash weight (equilibrium share):")
    for i in order:
        bar = "#" * int(round(nash[i] * 40))
        print(f"    {labels[i]:8} rating {h['ratings'][i]:+.3f}  nash {nash[i]:.2f} {bar}")
    print(f"\n  intransitivity(cyclic) {rep['intransitivity']}   "
          f"Nash support {rep['nash_support']} / entropy {rep['nash_entropy']}   "
          f"alpha-Rank entropy {rep['arank_entropy']}")
    print(f"  dominance {rep['dominance']} -> {rep['dominant']}    "
          f"DIVERSITY OBJECTIVE {obj['score']}")
    return rep


def rule_audit(base_cfg, n_games=20, sims=80):
    """Elegance audit: does each VISIBLE rule earn its keep? Toggle one rule off
    at a time, re-run MCTS self-play at the healthy config, compare balance. A
    rule whose removal barely moves the metrics is a candidate to CUT (then
    re-check diversity before deleting). Group-liberty capture and the two win
    paths are structural/correctness — not toggleable here.

    Read it as: which printed-rulebook lines are load-bearing vs ceremony?"""
    from dataclasses import replace
    variants = {
        "baseline (all on)":    base_cfg,
        "no setup_window":      replace(base_cfg, setup_window=False),
        "no superko":           replace(base_cfg, superko=False),
        "hard_protection ON":   replace(base_cfg, hard_trooper_protection=True),
    }
    print("=" * 72)
    print(f"RULE-ELEGANCE AUDIT — toggle each visible rule off (n={base_cfg.n} "
          f"range={base_cfg.drone_range} supply={base_cfg.drone_supply}, sims={sims}, "
          f"{n_games} games)")
    print("   a rule whose removal barely moves balance = candidate to cut")
    print("=" * 72)
    print(f"\n{'variant':22} {'p0win':>6} {'decap':>6} {'draw':>6} {'avgPly':>7} {'object':>7}")
    base_line = None
    for name, cfg in variants.items():
        tasks = [(cfg, MCTSPlayer(sims=sims, seed=g),
                  MCTSPlayer(sims=sims, seed=1000 + g), "garrison", "garrison", g, g)
                 for g in range(n_games)]
        results = _run_all(tasks)
        recs = [GameRecord(w, k, p, (0, 0)) for (_t, w, k, p) in results]
        s = summarize(recs); obj = balance_objective(s)
        flag = ""
        if name.startswith("baseline"):
            base_line = (s["p0_winrate"], s["decap_share"], s["avg_plies"], obj)
        elif base_line:
            d_obj = abs(obj - base_line[3])
            d_decap = abs(s["decap_share"] - base_line[1])
            flag = "  <- ~unchanged: CUT candidate" if (d_obj < 0.15 and d_decap < 0.15) else "  <- matters: keep"
        print(f"{name:22} {s['p0_winrate']:>6.2f} {s['decap_share']:>6.2f} "
              f"{s['draw_rate']:>6.2f} {s['avg_plies']:>7.1f} {obj:>7.3f}{flag}")
    print("\n  (balance only — re-check setup-variant diversity before deleting any rule.)")


def main():
    t0 = time.time()
    balance_sweep(grid=[(2, 2), (2, 3), (3, 2), (3, 3)], n_games=16, sims=80)
    diversity_rerun(RuleConfig(n=7, drone_range=3, troopers=4, capture_to_win=2,
                               max_plies=160), games_per_pair=3, sims=80)
    print(f"\n(done in {time.time() - t0:.0f}s with {N_WORKERS} workers; "
          f"raise sims/games/n for sharper numbers.)")


if __name__ == "__main__":
    main()
