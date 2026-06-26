"""python3 diversity_demo.py

(1) Full diversity report for one rule config, including the payoff matrix.
(2) A sweep over piece-strength dials, ranked by diversity_objective — i.e.
    'which rule settings make the most ways-to-play viable?'

Small/fast settings so it runs in this environment. For trustworthy numbers:
raise n, games_per_pair, and swap StyleBots for an MCTS wrapper (see CLAUDE.md).
"""
import numpy as np
from hoshi import RuleConfig, make_styles, diversity_report, diversity_objective

np.set_printoptions(precision=2, suppress=True)


def show_report(cfg, games_per_pair=6, seed=1):
    pol = make_styles(seed=0)
    rep = diversity_report(cfg, pol, games_per_pair=games_per_pair, seed=seed)
    print("payoff matrix W  (row beats column, win-rate):")
    print("            " + "".join(f"{n[:7]:>9}" for n in rep["names"]))
    for i, n in enumerate(rep["names"]):
        print(f"  {n[:10]:10}" + "".join(f"{rep['W'][i][j]:>9.2f}" for j in range(len(rep['names']))))
    print()
    print("  HodgeRank (skill spine):", rep["ratings"])
    print(f"  intransitivity (cyclic share)   {rep['intransitivity']}   (1.0 = pure rock-paper-scissors)")
    print(f"  transitivity   (ladder share)   {rep['transitivity']}")
    print(f"  Nash mixture                    {dict(zip(rep['names'], np.round(rep['nash'],2)))}")
    print(f"  Nash support / entropy          {rep['nash_support']} / {rep['nash_entropy']}")
    print(f"  alpha-Rank entropy              {rep['arank_entropy']}")
    print(f"  dominance (best vs field)       {rep['dominance']}  -> {rep['dominant']}")
    obj = diversity_objective(rep)
    print(f"  DIVERSITY OBJECTIVE             {obj['score']}   {obj}")
    return rep


def main():
    print("=" * 70)
    print("DIVERSITY REPORT — baseline config (n=7 for speed)")
    print("=" * 70)
    show_report(RuleConfig(n=7, drone_range=3, troopers=4, capture_to_win=2, max_plies=200))

    print("\n" + "=" * 70)
    print("PIECE-STRENGTH SWEEP — find the most varied region")
    print("  dials: drone_range (drone power) x capture_to_win (trooper value)")
    print("         x troopers (how many anchors)")
    print("=" * 70)
    rows = []
    for dr in (2, 3):
        for ctw in (2, 3):
            for ntr in (3, 5):
                cfg = RuleConfig(n=7, drone_range=dr, troopers=ntr, capture_to_win=ctw,
                                 max_plies=200)
                pol = make_styles(seed=0)
                rep = diversity_report(cfg, pol, games_per_pair=4, seed=3)
                obj = diversity_objective(rep)
                rows.append((obj["score"], dr, ctw, ntr, rep["intransitivity"],
                             rep["arank_entropy"], rep["dominance"], rep["dominant"]))
    rows.sort(reverse=True)
    print(f"\n{'score':>6} {'range':>5} {'capWin':>6} {'troops':>6} "
          f"{'cyclic':>6} {'aR-ent':>6} {'domin':>6}  dominant")
    for sc, dr, ctw, ntr, cyc, ar, dom, who in rows:
        print(f"{sc:>6.3f} {dr:>5} {ctw:>6} {ntr:>6} {cyc:>6.2f} {ar:>6.2f} {dom:>6.2f}  {who}")
    print("\nTop row = most strategic variety under this (weak, heuristic) bot set.")
    print("Re-run with MCTS + more games before trusting any specific ordering.")


if __name__ == "__main__":
    main()
