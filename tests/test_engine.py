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


def _enc(n=5):
    # encircle on; capture_to_win high so a capture doesn't end the game mid-assert
    return RuleConfig(n=n, drone_range=2, troopers=3, capture_to_win=3,
                      superko=False, trooper_encircle=True)


def test_encircle_trooper_immune_to_group_capture():
    # A trooper joined to a friendly drone, group reduced to one liberty. The enemy
    # fills it: under encircle the DRONE dies but the trooper SURVIVES (it can't be
    # captured like a drone via the group's shared liberty).
    s = State(cfg=_enc(), reserve=[3, 3], to_move=0)
    s.board = {(2, 2): (1, 'T'), (2, 3): (1, 'D'),
               (1, 2): (0, 'D'), (3, 2): (0, 'D'), (2, 1): (0, 'D'),
               (1, 3): (0, 'D'), (3, 3): (0, 'D')}      # last liberty is (2,4)
    s.placed_ply[(2, 2)] = -10
    grp, libs = _group_of(s.board, (2, 2), 5)
    assert libs == {(2, 4)}, f"setup should leave one liberty, got {libs}"
    s2 = apply_move(s, ('drop_t', (2, 4)))              # fill the last group liberty
    assert (2, 2) in s2.board and s2.board[(2, 2)] == (1, 'T'), "trooper must survive"
    assert (2, 3) not in s2.board, "the drone in the dead group must be removed"
    assert s2.lost_troopers[1] == 0, "no trooper captured by group-liberty fill"


def test_encircle_trooper_dies_when_walled_in():
    # A lone trooper with three enemy neighbors; the enemy plays the fourth. Every
    # on-board neighbor is now an enemy stone -> the trooper is removed.
    s = State(cfg=_enc(), reserve=[3, 3], to_move=0)
    s.board = {(2, 2): (1, 'T'), (1, 2): (0, 'D'), (3, 2): (0, 'D'), (2, 1): (0, 'D')}
    s.placed_ply[(2, 2)] = -10
    s2 = apply_move(s, ('drop_t', (2, 3)))              # complete the wall
    assert (2, 2) not in s2.board, "fully walled-in trooper must be captured"
    assert s2.lost_troopers[1] == 1


def test_encircle_corner_trooper_uses_onboard_neighbors():
    # A corner trooper has only two on-board neighbors; filling both walls it in.
    s = State(cfg=_enc(), reserve=[3, 3], to_move=0)
    s.board = {(0, 0): (1, 'T'), (1, 0): (0, 'D')}
    s.placed_ply[(0, 0)] = -10
    s2 = apply_move(s, ('drop_t', (0, 1)))
    assert (0, 0) not in s2.board, "corner trooper walled by its 2 on-board neighbors"
    assert s2.lost_troopers[1] == 1


def test_encircle_off_keeps_go_capture():
    # Same liberty-fill as the immune test, but with encircle OFF: classic group
    # capture removes the whole group, trooper included.
    cfg = RuleConfig(n=5, drone_range=2, troopers=3, capture_to_win=3, superko=False)
    s = State(cfg=cfg, reserve=[3, 3], to_move=0)
    s.board = {(2, 2): (1, 'T'), (2, 3): (1, 'D'),
               (1, 2): (0, 'D'), (3, 2): (0, 'D'), (2, 1): (0, 'D'),
               (1, 3): (0, 'D'), (3, 3): (0, 'D')}
    s.placed_ply[(2, 2)] = -10
    s2 = apply_move(s, ('drop_t', (2, 4)))
    assert (2, 2) not in s2.board, "default group capture should remove the trooper"
    assert s2.lost_troopers[1] == 1


def _mob(n=5):
    return RuleConfig(n=n, drone_range=2, troopers=3, capture_to_win=3,
                      superko=False, mobile_drones=True)


def test_mobile_drone_relocates_within_range():
    s = State(cfg=_mob(), reserve=[3, 3], to_move=0)
    s.board = {(2, 2): (0, 'T'), (2, 3): (0, 'D')}      # ready trooper + a drone
    s.placed_ply[(2, 2)] = -10
    assert ('move', (2, 3), (1, 2)) in legal_moves(s), "in-range relocate should be legal"
    s2 = apply_move(s, ('move', (2, 3), (1, 2)))
    assert (2, 3) not in s2.board, "drone left its old point"
    assert s2.board.get((1, 2)) == (0, 'D'), "drone arrived at the new point"


def test_mobile_off_has_no_move_actions():
    cfg = RuleConfig(n=5, drone_range=2, troopers=3, capture_to_win=3, superko=False)
    s = State(cfg=cfg, reserve=[3, 3], to_move=0)
    s.board = {(2, 2): (0, 'T'), (2, 3): (0, 'D')}
    s.placed_ply[(2, 2)] = -10
    assert all(m[0] != 'move' for m in legal_moves(s)), "no move actions when mobile off"


def test_mobile_drone_out_of_range_illegal():
    s = State(cfg=_mob(), reserve=[3, 3], to_move=0)
    s.board = {(0, 0): (0, 'T'), (1, 1): (0, 'D')}      # trooper reaches only cheb<=2 of (0,0)
    s.placed_ply[(0, 0)] = -10
    assert ('move', (1, 1), (4, 4)) not in legal_moves(s), "out-of-reach target is illegal"
    try:
        apply_move(s, ('move', (1, 1), (4, 4))); assert False, "should have raised"
    except ValueError:
        pass


def test_mobile_only_own_drones_move():
    s = State(cfg=_mob(), reserve=[3, 3], to_move=0)
    s.board = {(2, 2): (0, 'T'), (1, 1): (1, 'D')}      # enemy drone, my trooper
    s.placed_ply[(2, 2)] = -10
    try:
        apply_move(s, ('move', (1, 1), (2, 1))); assert False, "can't move an enemy drone"
    except ValueError:
        pass
    try:
        apply_move(s, ('move', (2, 2), (2, 1))); assert False, "can't 'move' a trooper"
    except ValueError:
        pass


def test_mobile_move_can_capture():
    s = State(cfg=_mob(), reserve=[3, 3], to_move=0)
    # enemy drone at (2,2) in atari (last liberty (2,3)); my drone relocates onto it.
    s.board = {(2, 2): (1, 'D'), (1, 2): (0, 'D'), (3, 2): (0, 'D'), (2, 1): (0, 'D'),
               (4, 4): (0, 'T'), (4, 3): (0, 'D')}
    s.placed_ply[(4, 4)] = -10
    s2 = apply_move(s, ('move', (4, 3), (2, 3)))        # (2,3) is within range of (4,4)
    assert (2, 2) not in s2.board, "relocating to the last liberty captures the enemy drone"
    assert (4, 3) not in s2.board, "the moved drone vacated its old point"


def _step(n=5):
    return RuleConfig(n=n, drone_range=3, troopers=3, capture_to_win=3,
                      superko=False, mobile_drones=True, drone_move_mode='step')


def test_step_mode_only_orthogonal():
    s = State(cfg=_step(), reserve=[3, 3], to_move=0)
    s.board = {(2, 2): (0, 'T'), (2, 3): (0, 'D')}     # R=3 trooper shadows the area
    s.placed_ply[(2, 2)] = -10
    legal = legal_moves(s)
    assert ('move', (2, 3), (2, 4)) in legal, "orthogonal step is legal"
    assert ('move', (2, 3), (1, 4)) not in legal, "diagonal is not a step"
    assert ('move', (2, 3), (0, 3)) not in legal, "a two-cell jump is not a step"


def test_step_mode_stays_in_shadow():
    # a step whose destination leaves every ready trooper's range is illegal
    s = State(cfg=_step(), reserve=[3, 3], to_move=0)
    s.board = {(0, 0): (0, 'T'), (0, 3): (0, 'D')}     # trooper at corner, R=3
    s.placed_ply[(0, 0)] = -10
    # (0,3) is in range (cheb 3); stepping to (0,4) is cheb 4 -> out of shadow
    assert ('move', (0, 3), (0, 4)) not in legal_moves(s), "step out of shadow illegal"
    assert ('move', (0, 3), (0, 2)) in legal_moves(s), "step staying in shadow is legal"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        fn(); print(f"  ok  {fn.__name__}"); passed += 1
    print(f"\n{passed}/{len(fns)} tests passed")
