"""Correctness tests. Agents amplify engine bugs — keep these green."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from hoshi.engine import (RuleConfig, State, apply_move, legal_moves,
                          _group_of, area_score, make_initial_state, cheb)


def mk(n=5, troopers=3, **kw):
    return RuleConfig(n=n, drone_range=2, troopers=troopers, capture_to_win=2,
                      superko=True, **kw)


def test_orthogonal_capture():
    cfg = mk()
    s = State(cfg=cfg, reserve=[3, 3])
    # white (1) drone at center, black (0) surrounds orthogonally
    s.board = {(2, 2): (1, 'D'), (1, 2): (0, 'D'), (3, 2): (0, 'D'), (2, 1): (0, 'D')}
    s.to_move = 0
    # place the surrounding stone directly to isolate capture logic
    s.board[(2, 3)] = (0, 'D')
    grp, libs = _group_of(s.board, (2, 2), cfg.n)
    assert libs == set(), "fully surrounded drone should have no liberties"


def test_group_shares_liberties():
    cfg = mk()
    s = State(cfg=cfg)
    s.board = {(2, 2): (1, 'T'), (2, 3): (1, 'D')}  # connected pair
    grp, libs = _group_of(s.board, (2, 2), cfg.n)
    assert grp == {(2, 2), (2, 3)}
    assert len(libs) == 6, f"two-stone group on open board has 6 liberties, got {len(libs)}"


def test_protection_is_emergent():
    # a trooper with an adjacent friendly drone survives a partial surround
    cfg = mk()
    s = State(cfg=cfg)
    # trooper at (2,2), friendly drone at (2,3). enemy on the trooper's other 3 sides.
    s.board = {(2, 2): (1, 'T'), (2, 3): (1, 'D'),
               (1, 2): (0, 'D'), (3, 2): (0, 'D'), (2, 1): (0, 'D')}
    grp, libs = _group_of(s.board, (2, 2), cfg.n)
    assert len(libs) > 0, "group still breathes through the drone's liberties"


def test_suicide_illegal():
    cfg = mk(troopers=1)
    s = State(cfg=cfg, reserve=[1, 1])
    s.to_move = 1
    # surround an empty point with player-0 stones, then player 1 tries to play into it
    s.board = {(1, 2): (0, 'D'), (3, 2): (0, 'D'), (2, 1): (0, 'D'), (2, 3): (0, 'D')}
    # give player 1 a ready trooper in range so the drone is otherwise legal
    s.board[(0, 0)] = (1, 'T'); s.placed_ply[(0, 0)] = -10
    cfg2 = RuleConfig(n=5, drone_range=9, troopers=1, capture_to_win=2)
    s.cfg = cfg2
    ms = legal_moves(s)
    assert ('drone', (2, 2)) not in ms, "playing into a fully enclosed point is suicide -> illegal"


def test_capture_beats_suicide():
    # the same enclosed point IS legal if playing there captures an enemy first
    cfg = RuleConfig(n=5, drone_range=9, troopers=1, capture_to_win=2)
    s = State(cfg=cfg, reserve=[1, 1], to_move=1)
    # enemy drone at (2,2) with only one liberty at (2,3); player1 ring + trooper
    s.board = {(2, 2): (0, 'D'),
               (1, 2): (1, 'D'), (3, 2): (1, 'D'), (2, 1): (1, 'D'),
               (0, 0): (1, 'T')}
    s.placed_ply[(0, 0)] = -10
    ms = legal_moves(s)
    assert ('drone', (2, 3)) in ms
    s2 = apply_move(s, ('drone', (2, 3)))
    assert (2, 2) not in s2.board, "the enemy drone should be captured"


def test_setup_window():
    cfg = RuleConfig(n=7, drone_range=3, troopers=2, capture_to_win=2, setup_window=True)
    s = State(cfg=cfg, reserve=[2, 2])
    s2 = apply_move(s, ('drop_t', (3, 3)))   # player 0 drops a trooper
    # player 1 moves
    s3 = apply_move(s2, ('pass',))
    # back to player 0: the trooper landed at ply 0, now ply==2 -> ready
    assert s3.trooper_ready((3, 3)), "trooper should be ready on owner's next turn"
    # and immediately after landing it should NOT be ready
    assert not s2.trooper_ready((3, 3))


def test_decapitation_win():
    cfg = RuleConfig(n=5, drone_range=9, troopers=2, capture_to_win=1)
    s = State(cfg=cfg, reserve=[0, 0], to_move=0)
    # enemy trooper at (2,2) in atari; player0 trooper to deploy + ring
    s.board = {(2, 2): (1, 'T'),
               (1, 2): (0, 'D'), (3, 2): (0, 'D'), (2, 1): (0, 'D'),
               (4, 4): (0, 'T')}
    s.placed_ply[(4, 4)] = -10
    s2 = apply_move(s, ('drone', (2, 3)))
    assert s2.over and s2.winner == 0, "capturing the only trooper needed should win"


def test_full_game_terminates():
    cfg = RuleConfig(n=7, drone_range=3, troopers=3, capture_to_win=2, max_plies=400)
    import random
    from hoshi.players import GreedyPlayer
    from hoshi.harness import play_game
    rec = play_game(cfg, GreedyPlayer(1), GreedyPlayer(2), seed=7)
    assert rec.winner in (0, 1, -1)
    assert rec.plies <= cfg.max_plies


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        fn(); print(f"  ok  {fn.__name__}"); passed += 1
    print(f"\n{passed}/{len(fns)} tests passed")
