"""Run me: python3 selfplay_demo.py

Proves the pipeline end to end:
  1. a metrics run on the default config
  2. a setup-variant round-robin (which mode beats which?)
This is the skeleton an optimization agent drives — swap GreedyPlayer for
an MCTS bot and widen the sweeps for real signal.
"""
from hoshi import RuleConfig, GreedyPlayer, play_game, summarize, balance_objective
from hoshi.engine import SETUPS

def run_config(cfg, n_games=40, setup="garrison"):
    recs = []
    for g in range(n_games):
        # alternate colors and vary seeds so first-player bias is measurable
        p0, p1 = GreedyPlayer(seed=g), GreedyPlayer(seed=1000 + g)
        recs.append(play_game(cfg, p0, p1, setup, setup, seed=g))
    return summarize(recs)

def main():
    print("=" * 64)
    print("HOSHI self-play — default rules (9x9 for speed; bump n=13 for real)")
    print("=" * 64)
    cfg = RuleConfig(n=9, drone_range=3, troopers=5, capture_to_win=3, max_plies=300)
    s = run_config(cfg, n_games=30)
    for k, v in s.items():
        print(f"  {k:18} {v}")
    print(f"  {'balance_objective':18} {balance_objective(s)}  (lower = better)")

    print("\n" + "=" * 64)
    print("Setup-variant round-robin (mirror matches, 12 games each)")
    print("  win% = player-0 (the row variant) vs player-1 (the column variant)")
    print("=" * 64)
    variants = ["beachhead", "vanguard", "garrison", "bastion"]
    cfg = RuleConfig(n=9, drone_range=3, troopers=5, capture_to_win=3, max_plies=300)
    header = "          " + "".join(f"{v[:8]:>10}" for v in variants)
    print(header)
    for a in variants:
        row = f"{a[:9]:9} "
        for b in variants:
            wins = 0
            for g in range(12):
                rec = play_game(cfg, GreedyPlayer(g), GreedyPlayer(99 - g), a, b, seed=g)
                if rec.winner == 0:
                    wins += 1
            row += f"{wins/12:>10.2f}"
        print(row)
    print("\n(Greedy baseline only — directional, not authoritative. MCTS will sharpen it.)")

if __name__ == "__main__":
    main()
