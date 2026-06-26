# HOSHI — engine + agent harness (handoff)

A Go-derived board game. Troopers (anchors) drop anywhere; drones deploy within
range of a *ready* trooper and convert influence into territory. Win by
**decapitation** (capture N enemy troopers) or **territory** (area score on two
passes). This repo is the substrate for two jobs: **tuning the rules** and
**discovering strategy**. Keep them separate — they want different machinery.

## Status
- `hoshi/engine.py` — rules, move gen, group capture, scoring, setup variants. **Tested.**
- `hoshi/players.py` — `RandomPlayer`, `GreedyPlayer` (1-ply, also usable as a rollout policy).
- `hoshi/mcts.py` — `MCTSPlayer` (UCT + 1-ply shortlist + fast rollouts). **Beats Greedy >70%.** The workhorse: use it for any number you intend to trust.
- `hoshi/harness.py` — `play_game`, `summarize`, `balance_objective`.
- `tests/test_engine.py` — 8 correctness tests. **Run before and after every change.**
- `tests/test_mcts.py` — strength gate (>70% vs Greedy). ~3 min; `HOSHI_FAST=1` for a quick smoke check.
- `selfplay_demo.py` — end-to-end metrics + variant round-robin.

```
python3 tests/test_engine.py     # must stay green
python3 tests/test_mcts.py       # strength gate (HOSHI_FAST=1 to hurry)
python3 selfplay_demo.py         # end-to-end sanity
```

## The prime directive
**Never tune rules or trust strategy claims while a test is red.** An agent
optimizing on a broken engine produces confident garbage. Engine correctness is
the foundation; everything else amplifies it. If you change a rule, add a test
that pins the new behavior first.

## Invariants vs. dials (do not blur these)
The whole design philosophy is a fixed core with a menu of variations bolted on.

**Invariant core (don't "optimize" these away):**
- Drones deploy only from a living, *ready* trooper; drones never spawn drones.
- Setup window: a trooper can't deploy the turn it lands.
- Capture is **group liberties** (Go). Suicide illegal unless it captures.
- Two win paths, raced in parallel.

**Dials (`RuleConfig`) — this is the search space:**
`n` (board), `drone_range`, `troopers`, `capture_to_win`, `setup_window`,
`superko`, `hard_trooper_protection`, `komi`, `drone_supply`.
Plus the categorical **setup variant** per player
(`beachhead/vanguard/garrison/bastion/fog`).

## One rule decision you should know about
The cardboard sheet says "a piece dies with no empty adjacent point." Literally,
that self-captures friendly blobs. The engine uses **group liberties** instead,
which (a) fixes that and (b) makes "trooper protection" *emergent*: a drone next
to a trooper joins its group and shares liberties. So there is no special
protection rule by default. If you want troopers categorically tougher, flip
`hard_trooper_protection=True` (a trooper immune while an orthogonally adjacent
friendly drone exists) and add tests for it.

## What "optimize the rules" means here
Maximize game health, measured by `harness.summarize`:
- `p0_winrate` near 0.5 (first-player fairness; `komi` is the lever).
- `decap_share` *not* near 0 or 1 — both win paths must stay viable.
- `avg_plies` in a sane band (not degenerate-short, not a grind).
`balance_objective(summary)` rolls these into one number to minimize. **Edit its
weights to encode your taste** — that function literally defines "a good game."

## Build order (suggested)
1. **MCTS bot** (`hoshi/mcts.py`): ✅ DONE. UCT with a 1-ply greedy *shortlist*
   (prune ~40 near-duplicate drops to top-`action_width`, else sims are pure
   noise at the root), fast lazy-sampled rollouts, eval fallback at the depth
   cap. Beats Greedy >70% and strength rises monotonically with `sims`. Swap it
   in for `GreedyPlayer`/`StyleBot` anywhere you want trustworthy numbers.
2. **Sweep tool** (`sweep.py`): grid or CMA-ES over `RuleConfig`, N games per
   cell, dump a CSV/JSON of summaries + `balance_objective`. Parallelize with
   `multiprocessing` (engine state is plain data — easy to pickle).
3. **Instrumentation**: per-game logs of trooper-drop locations, opening drone
   shapes, lead-over-time. Feed aggregates (heatmaps, not raw games) to analysis.
4. **Perf** only if sweeps are too slow: profile `_group_of` first; consider a
   numpy/bitboard rewrite or a Rust core behind the same `legal_moves/apply` API.

## The two agent loops (this is the "agents" part)
**A. Rule-tuner (closed loop).** Tool surface: run a config → get `summarize` +
`balance_objective`. The agent proposes a config, runs ~200 MCTS-vs-MCTS games,
reads metrics, hypothesizes ("territory never fires; capture too cheap at
range 4 → try range 3 or capture_to_win=4"), edits, re-runs. Stop when the
objective plateaus. Keep a log of (config, metrics, rationale).

**B. Strategy-analyst (open loop).** Freeze a config. Run MCTS self-play, collect
aggregate stats per setup variant, and have the agent *write up* the emergent
strategy: where winners drop the first trooper, how deep-drop win-rate compares
to safe-drop, which variant dominates the round-robin and why. Code computes the
numbers; the agent narrates the pattern. Output: a short strategy memo per mode.

Do **not** run balance sweeps as LLM-agent games — too slow/expensive for zero
extra signal. LLM agents drive the outer loop; MCTS bots supply the volume.

## API contract (stable; build against this)
```python
from hoshi import RuleConfig, make_initial_state, legal_moves, apply_move, area_score
s  = make_initial_state(cfg, setup0="garrison", setup1="beachhead", seed=0)
ms = legal_moves(s)                # list of ('drop_t',pt) | ('drone',pt) | ('pass',)
s2 = apply_move(s, ms[0])          # returns a NEW State; raises on illegal
s2.over, s2.winner                 # winner: 0 | 1 | -1 (draw)
```
A `Player` is anything with `.choose(state) -> move`.

---

# Diversity layer (the real objective)

The goal is **strategic diversity**, not balance: how many distinct ways to play
stay viable when everyone plays well, and whether they beat each other in cycles
rather than a straight ladder. Balance is a *constraint* (no single dominant
strategy), not the target.

Files: `hoshi/strategies.py` (playstyle policies) · `hoshi/diversity.py`
(measurement) · `tests/test_diversity.py` (math validated on RPS / transitive
ladders) · `diversity_demo.py` (report + piece-strength sweep).

```
python3 tests/test_diversity.py   # 9 tests; keep green
python3 diversity_demo.py
```

## What it measures
- **Payoff matrix** over playstyle policies (aggressive / territorial / balanced
  / expansionist), color-swapped so first-player bias cancels.
- **Nash** (maximin mixed strategy) — support size + entropy = how many ways to
  play survive optimal counter-play.
- **alpha-Rank** — evolutionary stationary distribution; robust, no
  equilibrium-selection ambiguity.
- **Hodge split** — decomposes the advantage flow into a transitive skill spine
  + a cyclic (rock-paper-scissors) part. `intransitivity` (cyclic squared-norm
  share, in [0,1]) is the number to push UP. This is the "spinning top" measure.
- **dominance** — best win-rate any policy gets vs the field; the no-dominance
  constraint.
- `diversity_objective` rolls these into one score to MAXIMIZE. Its weights
  encode your taste — edit them deliberately.

## Two findings from building this (don't relearn the hard way)
1. **Symmetrize before decomposing.** `W[i][j]` and `W[j][i]` come from different
   games, so the raw matrix isn't antisymmetric and Hodge ratios blow past 1.
   `diversity_report` builds `P = (W + 1 - W.T)/2` first. Keep that.
2. **`hard_trooper_protection` is inert under group capture.** A drone adjacent to
   a trooper is already in its group, so it dies with it — the protection check
   never fires. For a REAL trooper-toughness dial, implement **trooper armor**:
   capturing a trooper requires filling its group's liberties AND it survives the
   first such event (a one-time "must be surrounded twice"), tracked as state on
   the trooper. Add tests, then it becomes a genuine strength lever.

## Working strength dials (the sweep space)
`drone_range` (drone power), `capture_to_win` (trooper value), `troopers` (anchor
count), board `n` and the **n/range ratio** (the biggest structural lever — too
little zone overlap and the game decomposes into two solitaires, which reads as
low variance but is really *no interaction*). Future engine levers: **mobile
drones** (a `move_drone` action) and **trooper armor** above.

## Trust gate (load-bearing)
Every number here is computed with weak 1-ply heuristic bots, so it partly
measures the bots' blind spots. **Build the MCTS bot first** and re-run before
trusting any ordering. Diversity measured with a weak bot is the bot's noise in
a costume. The math is validated independently (test_diversity.py); the *bots*
are the weak link, not the metrics.
