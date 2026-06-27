"""Strategy-analyst probe: does trooper PLACEMENT diversify as troopers get more
valuable? Measures variance-of-play, not endgame outcome.

Per config, run MCTS self-play (color-mirrored) and record where each side puts
its troopers (home setup + every 'drop_t'). Then report:

  within_disp   mean pairwise distance among ONE side's own troopers, normalized
                0..1. High = troopers spread across the board (a supply/logistics
                spread); low = clustered (one corner, Go-blob).
  open_entropy  Shannon entropy of WHERE troopers land, pooled over all games,
                normalized 0..1. High = different placements game-to-game (rich,
                varied openings); low = same star points every game (solved / Go).
  win_minus_los winner's within_disp minus loser's. Far from 0 = placement is a
                STRATEGIC lever (winners place differently); ~0 = placement noise.

Constraints (sanity, not the objective): decap_share, p0_winrate, avg_plies.

Usage:  python3.11 trooper_diversity.py [pilot|full]
"""
from __future__ import annotations
import math, sys, time
from itertools import combinations
from multiprocessing import Pool
from hoshi.engine import RuleConfig, make_initial_state, apply_move, area_score
from hoshi.mcts import MCTSPlayer

WORKERS = 3          # 4-core / 8 GB box: leave one core for the system
ROLLOUT_DEPTH = 30   # trimmed from the 50 default for speed; enough for a signal


def trooper_points(board, color):
    return [pt for pt, (c, k) in board.items() if c == color and k == 'T']


def play_and_log(cfg, sims, setup, seed):
    """One self-play game. Returns (winner, kind, plies, placements[2], disp[2])."""
    s = make_initial_state(cfg, setup, setup, seed=seed)
    p0 = MCTSPlayer(sims=sims, seed=seed * 2, rollout_depth=ROLLOUT_DEPTH)
    p1 = MCTSPlayer(sims=sims, seed=seed * 2 + 1, rollout_depth=ROLLOUT_DEPTH)
    players = [p0, p1]
    # placement points per color = home troopers at start + every drop_t played
    placed = [list(trooper_points(s.board, 0)), list(trooper_points(s.board, 1))]
    while not s.over:
        mover = s.to_move
        m = players[mover].choose(s)
        if m[0] == 'drop_t':
            placed[mover].append(m[1])
        s = apply_move(s, m)
    if s.winner != -1 and s.lost_troopers[1 - s.winner] >= cfg.capture_to_win and \
       max(s.lost_troopers) >= cfg.capture_to_win:
        kind = "decapitation"
    else:
        kind = "territory"
    disp = [pairwise_disp(placed[0], cfg.n), pairwise_disp(placed[1], cfg.n)]
    return s.winner, kind, s.ply, placed, disp


def pairwise_disp(pts, n):
    """Mean pairwise Chebyshev distance among a side's troopers, normalized 0..1."""
    if len(pts) < 2:
        return 0.0
    tot = 0.0
    for (r0, c0), (r1, c1) in combinations(pts, 2):
        tot += max(abs(r0 - r1), abs(c0 - c1))
    return (tot / math.comb(len(pts), 2)) / (n - 1)


def spatial_entropy(counts, n):
    """Normalized Shannon entropy of a board-cell count histogram, 0..1."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    h = -sum((v / total) * math.log(v / total) for v in counts.values() if v)
    return h / math.log(n * n)


def ring(pt, n):
    """Distance to the nearest edge: 0 = on the rim, max at center."""
    r, c = pt
    return min(r, c, n - 1 - r, n - 1 - c)


def zone_profile(pts, n):
    """Color-symmetric summary of WHERE winners place troopers."""
    if not pts:
        return {"edge": 0.0, "center": 0.0, "centrality": 0.0}
    maxring = (n - 1) // 2
    edge = sum(1 for p in pts if ring(p, n) == 0) / len(pts)
    center = sum(1 for p in pts if ring(p, n) == maxring) / len(pts)
    centrality = sum(ring(p, n) for p in pts) / (len(pts) * maxring)
    return {"edge": round(edge, 2), "center": round(center, 2),
            "centrality": round(centrality, 2)}


def fold_heatmap(pts, n):
    """Fold every placement into one quadrant (color/side-agnostic) and render an
    ASCII intensity map: rows top=rim .. bottom=center, so the shape of 'where
    winners anchor' is readable at a glance."""
    h = (n + 1) // 2
    grid = [[0] * h for _ in range(h)]
    for r, c in pts:
        fr, fc = min(r, n - 1 - r), min(c, n - 1 - c)
        grid[fr][fc] += 1
    mx = max((max(row) for row in grid), default=0) or 1
    ramp = " .:-=+*#%@"
    lines = []
    for row in grid:
        lines.append("".join(ramp[min(len(ramp) - 1, round(v / mx * (len(ramp) - 1)))]
                             for v in row))
    return lines


def _game_worker(arg):
    cfg, sims, setup, seed = arg
    return play_and_log(cfg, sims, setup, seed)


def run_cell(label, n, R, troopers, capture_to_win, games, sims, setup="garrison",
             encircle=False, mobile=False, max_plies=600, move_mode='redeploy',
             komi=0.0):
    cfg = RuleConfig(n=n, drone_range=R, troopers=troopers,
                     capture_to_win=capture_to_win, trooper_encircle=encircle,
                     mobile_drones=mobile, max_plies=max_plies, drone_move_mode=move_mode,
                     komi=komi)
    win = [0, 0, 0]            # p0, p1, draw
    decap = 0
    capped = 0                 # games that ran into the ply cap (the "drag" signal)
    plies_tot = 0
    cell_counts = {}           # board-cell -> placement count (pooled, both sides)
    win_disp, los_disp = [], []
    within_all = []
    winner_pts = []            # every trooper placement made by the winning side
    jobs = [(cfg, sims, setup, g) for g in range(games)]
    with Pool(WORKERS) as pool:
        results = pool.map(_game_worker, jobs)
    for winner, kind, plies, placed, disp in results:
        win[winner if winner != -1 else 2] += 1
        if winner != -1:
            winner_pts.extend(placed[winner])
        if kind == "decapitation":
            decap += 1
        if plies >= cfg.max_plies:
            capped += 1
        plies_tot += plies
        for color in (0, 1):
            within_all.append(disp[color])
            for pt in placed[color]:
                cell_counts[pt] = cell_counts.get(pt, 0) + 1
        if winner != -1:
            win_disp.append(disp[winner])
            los_disp.append(disp[1 - winner])
    n_games = games
    out = {
        "label": label,
        "n": n, "R": R, "troopers": troopers, "cap": capture_to_win,
        "reach_note": f"ratio={n/R:.1f}",
        "within_disp": round(sum(within_all) / len(within_all), 3),
        "open_entropy": round(spatial_entropy(cell_counts, n), 3),
        "win_minus_los": round(
            (sum(win_disp) / len(win_disp) - sum(los_disp) / len(los_disp)), 3)
            if win_disp and los_disp else None,
        "decap_share": round(decap / n_games, 2),
        "capped_share": round(capped / n_games, 2),
        "p0_winrate": round(win[0] / n_games, 2),
        "avg_plies": round(plies_tot / n_games, 1),
        "zones": zone_profile(winner_pts, n),
        "heatmap": fold_heatmap(winner_pts, n),
    }
    return out


CELLS_FULL = [
    # R3 band: hold range moderate (decap stays viable) and make troopers valuable
    # via count & capture_to_win instead of by strangling reach.
    # label,           n, R, troopers, cap
    ("baseline R4",     9, 4, 5, 3),   # reference: current web default
    ("R3 t5 cap3",      9, 3, 5, 3),   # range binds a little
    ("R3 t4 cap3",      9, 3, 4, 3),
    ("R3 t3 cap3",      9, 3, 3, 3),   # scarce troopers
    ("R3 t5 cap2",      9, 3, 5, 2),   # precious (lose only 2)
    ("R3 t4 cap2",      9, 3, 4, 2),
    ("R3 t3 cap2",      9, 3, 3, 2),   # scarce AND precious, range still allows the hunt
]
CELLS_PILOT = [
    ("baseline (loose)", 9, 4, 5, 3),
    ("tight+precious",   9, 2, 3, 2),
]


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "pilot"
    # mobile mode: isolate the mobile-drone lever (redeploy within trooper range).
    # Watch capped_share + avg_plies (does fluid play drag games to the cap?) and
    # decap (do captures still happen?) and edge/centrality (does play leave the
    # corners?). Lean: few games, low sims, a ply cap — mobility explodes branching.
    if mode == "mobile":
        MP = 220
        cells = [("R3 t5 cap2 static",   9, 3, 5, 2, False, False),
                 ("R3 t5 cap2 mobile",   9, 3, 5, 2, False, True),
                 ("R3 t5 enc2 mobile",   9, 3, 5, 2, True,  True)]
        games, sims = 14, 50
        hdr = f"{'label':20} {'within':>6} {'open':>5} {'wl':>6} {'decap':>5} " \
              f"{'capped':>6} {'p0wr':>5} {'plies':>6}"
        print(f"# mode=mobile games/cell={games} sims={sims} max_plies={MP} workers={WORKERS}\n{hdr}",
              flush=True)
        print("-" * len(hdr), flush=True)
        for lbl, n, R, t, cap, enc, mob in cells:
            t0 = time.time()
            r = run_cell(lbl, n, R, t, cap, games=games, sims=sims,
                         encircle=enc, mobile=mob, max_plies=MP)
            wml = "n/a" if r["win_minus_los"] is None else f"{r['win_minus_los']:+.3f}"
            print(f"{r['label']:20} {r['within_disp']:>6.3f} {r['open_entropy']:>5.2f} "
                  f"{wml:>6} {r['decap_share']:>5.2f} {r['capped_share']:>6.2f} "
                  f"{r['p0_winrate']:>5.2f} {r['avg_plies']:>6.1f}  [{time.time()-t0:.0f}s]",
                  flush=True)
            z = r["zones"]
            print(f"    winner placement  edge={z['edge']} center={z['center']} "
                  f"centrality={z['centrality']}:", flush=True)
            for line in r["heatmap"]:
                print(f"      {line}", flush=True)
        return
    # tune mode: step-mobility is the keeper; close the two warts. Sweep capture_to_win
    # (path balance: cap3 revives territory so komi has games to act on) x komi
    # (first-player fairness). Target: p0wr ~0.5 AND decap in a healthy [.15,.85] band.
    if mode == "tune":
        MP = 220
        caps, komis = [2, 3], [0]   # komi paused — isolate the capture_to_win effect first
        games, sims = 14, 50
        hdr = f"{'label':18} {'decap':>5} {'p0wr':>5} {'plies':>6} {'capped':>6} " \
              f"{'edge':>5} {'centr':>5}"
        print(f"# mode=tune step-mobility R3 t5  games/cell={games} sims={sims} "
              f"max_plies={MP} workers={WORKERS}\n{hdr}", flush=True)
        print("-" * len(hdr), flush=True)
        for cap in caps:
            for komi in komis:
                t0 = time.time()
                r = run_cell(f"cap{cap} komi{komi}", 9, 3, 5, cap, games=games, sims=sims,
                             mobile=True, move_mode='step', max_plies=MP, komi=komi)
                z = r["zones"]
                print(f"cap{cap} komi{komi:<10} {r['decap_share']:>5.2f} "
                      f"{r['p0_winrate']:>5.2f} {r['avg_plies']:>6.1f} "
                      f"{r['capped_share']:>6.2f} {z['edge']:>5.2f} {z['centrality']:>5.2f} "
                      f"  [{time.time()-t0:.0f}s]", flush=True)
        return
    # step mode: orthogonal-within-shadow drones (wriggle, don't teleport). The Plan-B
    # to redeploy: a drone can step one cell but never leave trooper range, so an
    # attacker can still run it down — captures should survive where redeploy killed
    # them. Compares static vs step vs redeploy at the balanced R3 t5 cap2 cell.
    if mode == "step":
        MP = 220
        cells = [("R3 t5 cap2 static",   False, 'redeploy'),
                 ("R3 t5 cap2 step",     True,  'step'),
                 ("R3 t5 cap2 redeploy", True,  'redeploy')]
        games, sims = 14, 50
        hdr = f"{'label':22} {'within':>6} {'open':>5} {'wl':>6} {'decap':>5} " \
              f"{'capped':>6} {'p0wr':>5} {'plies':>6}"
        print(f"# mode=step games/cell={games} sims={sims} max_plies={MP} workers={WORKERS}\n{hdr}",
              flush=True)
        print("-" * len(hdr), flush=True)
        for lbl, mob, mm in cells:
            t0 = time.time()
            r = run_cell(lbl, 9, 3, 5, 2, games=games, sims=sims,
                         mobile=mob, max_plies=MP, move_mode=mm)
            wml = "n/a" if r["win_minus_los"] is None else f"{r['win_minus_los']:+.3f}"
            print(f"{r['label']:22} {r['within_disp']:>6.3f} {r['open_entropy']:>5.2f} "
                  f"{wml:>6} {r['decap_share']:>5.2f} {r['capped_share']:>6.2f} "
                  f"{r['p0_winrate']:>5.2f} {r['avg_plies']:>6.1f}  [{time.time()-t0:.0f}s]",
                  flush=True)
            z = r["zones"]
            print(f"    winner placement  edge={z['edge']} center={z['center']} "
                  f"centrality={z['centrality']}:", flush=True)
            for line in r["heatmap"]:
                print(f"      {line}", flush=True)
        return
    # rescue mode: encircle ON, lower capture_to_win so the rare wall-in capture is
    # DECISIVE. Tests whether durable-but-precious troopers revive the decap path
    # and pull placement off the corners. (cap3 [enc] numbers come from `encircle`.)
    if mode == "rescue":
        cells = [("R3 t5 enc cap2", 9, 3, 5, 2), ("R3 t5 enc cap1", 9, 3, 5, 1),
                 ("R3 t4 enc cap2", 9, 3, 4, 2), ("R3 t4 enc cap1", 9, 3, 4, 1)]
        games, sims = 24, 100
        hdr = f"{'label':18} {'dials':16} {'within':>6} {'opening':>7} {'win-los':>7} " \
              f"{'decap':>5} {'p0wr':>5} {'plies':>6}"
        print(f"# mode=rescue (encircle ON) games/cell={games} sims={sims} workers={WORKERS}\n{hdr}",
              flush=True)
        print("-" * len(hdr), flush=True)
        for lbl, n, R, t, cap in cells:
            t0 = time.time()
            r = run_cell(lbl, n, R, t, cap, games=games, sims=sims, encircle=True)
            wml = "n/a" if r["win_minus_los"] is None else f"{r['win_minus_los']:+.3f}"
            print(f"{r['label']:18} n{r['n']} R{r['R']} t{r['troopers']} cap{r['cap']:<4} "
                  f"{r['within_disp']:>6.3f} {r['open_entropy']:>7.3f} {wml:>7} "
                  f"{r['decap_share']:>5.2f} {r['p0_winrate']:>5.2f} {r['avg_plies']:>6.1f} "
                  f"  [{time.time()-t0:.0f}s]", flush=True)
            z = r["zones"]
            print(f"    winner placement  edge={z['edge']} center={z['center']} "
                  f"centrality={z['centrality']}:", flush=True)
            for line in r["heatmap"]:
                print(f"      {line}", flush=True)
        return
    # encircle mode: the same R3 cells run twice (Go-capture vs local-encirclement)
    # to isolate the rule's effect on placement-strategy and the decap win-path.
    if mode == "encircle":
        base = [("R3 t5 cap3", 9, 3, 5, 3), ("R3 t4 cap3", 9, 3, 4, 3)]
        cells = ([(lbl + " [go]", *r, False) for lbl, *r in base] +
                 [(lbl + " [enc]", *r, True) for lbl, *r in base])
        games, sims = 24, 100
        hdr = f"{'label':18} {'dials':16} {'within':>6} {'opening':>7} {'win-los':>7} " \
              f"{'decap':>5} {'p0wr':>5} {'plies':>6}"
        print(f"# mode=encircle games/cell={games} sims={sims} workers={WORKERS}\n{hdr}",
              flush=True)
        print("-" * len(hdr), flush=True)
        for lbl, n, R, t, cap, enc in cells:
            t0 = time.time()
            r = run_cell(lbl, n, R, t, cap, games=games, sims=sims, encircle=enc)
            wml = "n/a" if r["win_minus_los"] is None else f"{r['win_minus_los']:+.3f}"
            print(f"{r['label']:18} n{r['n']} R{r['R']} t{r['troopers']} cap{r['cap']:<4} "
                  f"{r['within_disp']:>6.3f} {r['open_entropy']:>7.3f} {wml:>7} "
                  f"{r['decap_share']:>5.2f} {r['p0_winrate']:>5.2f} {r['avg_plies']:>6.1f} "
                  f"  [{time.time()-t0:.0f}s]", flush=True)
            z = r["zones"]
            print(f"    winner placement  edge={z['edge']} center={z['center']} "
                  f"centrality={z['centrality']}:", flush=True)
            for line in r["heatmap"]:
                print(f"      {line}", flush=True)
        return
    if mode == "pilot":
        cells, games, sims = CELLS_PILOT, 9, 60
    else:
        cells, games, sims = CELLS_FULL, 24, 100
    hdr = f"{'label':18} {'dials':16} {'within':>6} {'opening':>7} {'win-los':>7} " \
          f"{'decap':>5} {'p0wr':>5} {'plies':>6}"
    print(f"# mode={mode} games/cell={games} sims={sims} workers={WORKERS}\n{hdr}", flush=True)
    print("-" * len(hdr), flush=True)
    for c in cells:
        t0 = time.time()
        r = run_cell(*c, games=games, sims=sims)
        wml = "n/a" if r["win_minus_los"] is None else f"{r['win_minus_los']:+.3f}"
        print(f"{r['label']:18} "
              f"n{r['n']} R{r['R']} t{r['troopers']} cap{r['cap']:<4} "
              f"{r['within_disp']:>6.3f} {r['open_entropy']:>7.3f} {wml:>7} "
              f"{r['decap_share']:>5.2f} {r['p0_winrate']:>5.2f} {r['avg_plies']:>6.1f} "
              f"  [{time.time()-t0:.0f}s]", flush=True)
        z = r["zones"]
        print(f"    winner placement  edge={z['edge']} center={z['center']} "
              f"centrality={z['centrality']}  (folded heatmap, rim->center):", flush=True)
        for line in r["heatmap"]:
            print(f"      {line}", flush=True)


if __name__ == "__main__":
    main()
