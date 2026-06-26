"""MCTS strength + sanity tests.

The load-bearing claim (CLAUDE.md build order): the MCTS bot reliably beats
GreedyPlayer >70%. Until that holds, every rule/diversity number is measured
with weak bots and can't be trusted. Keep this green before tuning anything.

Deterministic: all seeds are fixed, so a green run stays green. The strength
test runs ~3 min single-threaded at the default budget; set HOSHI_FAST=1 for a
quick (weaker, ~30s) smoke check during iteration. Strength rises monotonically
with sims (≈0.67 @80, 0.75 @120, ≈0.8 @160) — the signature of real lookahead,
since MCTS and Greedy share the same leaf eval.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from hoshi.engine import RuleConfig, make_initial_state, legal_moves, apply_move
from hoshi.players import GreedyPlayer
from hoshi.mcts import MCTSPlayer
from hoshi.harness import play_game


def test_choose_returns_legal_move():
    cfg = RuleConfig(n=7, drone_range=3, troopers=4, capture_to_win=2, max_plies=200)
    s = make_initial_state(cfg, "garrison", "garrison", seed=0)
    bot = MCTSPlayer(sims=40, seed=1)
    m = bot.choose(s)
    assert m in legal_moves(s), "MCTS must return a legal move"
    # and it must apply without raising
    apply_move(s, m)


def test_mcts_terminates_game():
    cfg = RuleConfig(n=7, drone_range=3, troopers=3, capture_to_win=2, max_plies=200)
    rec = play_game(cfg, MCTSPlayer(sims=30, seed=1), GreedyPlayer(2), seed=5)
    assert rec.winner in (0, 1, -1)
    assert rec.plies <= cfg.max_plies


def test_mcts_beats_greedy():
    """MCTS vs Greedy, color-swapped so first-player bias can't carry it."""
    fast = os.environ.get("HOSHI_FAST")
    sims, pairs, thresh = (90, 4, 0.60) if fast else (160, 8, 0.70)
    cfg = RuleConfig(n=7, drone_range=3, troopers=4, capture_to_win=2, max_plies=160)
    games, wins = 0, 0.0
    for g in range(pairs):
        # MCTS as player 0
        rec = play_game(cfg, MCTSPlayer(sims=sims, seed=g), GreedyPlayer(100 + g), seed=g)
        wins += 1.0 if rec.winner == 0 else (0.5 if rec.winner == -1 else 0.0)
        games += 1
        # MCTS as player 1 (swap colors)
        rec = play_game(cfg, GreedyPlayer(200 + g), MCTSPlayer(sims=sims, seed=500 + g), seed=300 + g)
        wins += 1.0 if rec.winner == 1 else (0.5 if rec.winner == -1 else 0.0)
        games += 1
    rate = wins / games
    assert rate > thresh, f"MCTS only scored {rate:.2f} vs Greedy (want >{thresh})"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        fn(); print(f"  ok  {fn.__name__}"); passed += 1
    print(f"\n{passed}/{len(fns)} MCTS tests passed")
