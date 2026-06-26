"""Run games and compute the balance metrics that 'rule optimization' targets."""
from __future__ import annotations
from dataclasses import dataclass, field
from .engine import RuleConfig, State, make_initial_state, apply_move, area_score


@dataclass
class GameRecord:
    winner: int            # 0, 1, or -1 (draw)
    win_kind: str          # 'decapitation' | 'territory'
    plies: int
    score: tuple           # area score at end
    moves: list = field(default_factory=list)


def play_game(cfg: RuleConfig, p0, p1, setup0="garrison", setup1="garrison",
              seed=0, record_moves=False) -> GameRecord:
    s = make_initial_state(cfg, setup0, setup1, seed=seed)
    players = [p0, p1]
    moves = []
    # detect win kind by watching trooper losses cross the threshold
    while not s.over:
        mover = s.to_move
        m = players[mover].choose(s)
        s = apply_move(s, m)
        if record_moves:
            moves.append((mover, m))
    if s.winner != -1 and s.lost_troopers[1 - s.winner] >= cfg.capture_to_win and \
       max(s.lost_troopers) >= cfg.capture_to_win:
        kind = "decapitation"
    else:
        kind = "territory"
    return GameRecord(s.winner, kind, s.ply, area_score(s), moves)


def summarize(records: list[GameRecord]) -> dict:
    n = len(records)
    p0 = sum(1 for r in records if r.winner == 0)
    p1 = sum(1 for r in records if r.winner == 1)
    draw = sum(1 for r in records if r.winner == -1)
    decap = sum(1 for r in records if r.win_kind == "decapitation")
    terr = n - decap
    avg_len = sum(r.plies for r in records) / n if n else 0
    return {
        "games": n,
        "p0_winrate": round(p0 / n, 3) if n else 0,
        "p1_winrate": round(p1 / n, 3) if n else 0,
        "draw_rate": round(draw / n, 3) if n else 0,
        "decap_share": round(decap / n, 3) if n else 0,   # want this NOT ~0 and NOT ~1
        "territory_share": round(terr / n, 3) if n else 0,
        "avg_plies": round(avg_len, 1),
    }


def balance_objective(summary: dict) -> float:
    """Lower is better. A single number an optimizer/agent can minimize.

    Penalizes: first-player imbalance, a dead win-path, degenerate length.
    Tune the weights to encode what *you* think 'a good game' means.
    """
    import math
    fp_imbalance = abs(summary["p0_winrate"] - 0.5)
    # both win paths should stay alive; punish shares near 0 or 1
    path = summary["decap_share"]
    path_dead = max(0.0, 0.12 - min(path, 1 - path))   # 0 if path in [.12,.88]
    too_short = max(0.0, (40 - summary["avg_plies"]) / 100)
    return round(2.0 * fp_imbalance + 3.0 * path_dead + too_short, 4)
