# HOSHI

A Go-derived board game of paratroopers, drones, and territory. Troopers
(anchors) drop anywhere; drones deploy within range of a *ready* trooper and
convert influence into territory. Win by **decapitation** (capture N enemy
troopers) or **territory** (area score after two passes).

This repo has two halves:

## 1. The playable game — `web/hoshi.html`
**Live at https://hoshi.wrootlabs.com** (deployed on Vercel from `web/`).

A single, self-contained HTML file. **Open it in a browser to play** — vs the
computer, hot-seat two players, or watch Computer vs Computer. Place troopers,
deploy drones in the lit zone (tinted to the active player), capture by
surrounding groups, race the two win paths. Pick a board size and a setup
variant from the controls; games auto-save and a single "End & score" finishes.

```
node web/test_rules.js     # checks the game's rules engine (no browser needed)
```

## 2. The research engine — `hoshi/` (Python)
A pure, dependency-free engine plus a self-play harness for **tuning the rules**
and **measuring strategic diversity**. This is where the game's balance and
variety were validated with MCTS bots. Read **CLAUDE.md** for the full design
notes and findings.

```
python3 tests/test_engine.py      # 8 correctness tests
python3 tests/test_mcts.py        # MCTS strength gate (HOSHI_FAST=1 to hurry)
python3 tests/test_diversity.py   # diversity-math tests
python3 selfplay_demo.py          # metrics + setup round-robin
python3 sweep.py                  # MCTS-backed balance / diversity sweeps
```

Layout:
```
web/hoshi.html       playable game (self-contained)
web/test_rules.js    Node test for the game's rules engine
hoshi/engine.py      rules, move-gen, capture, scoring, setups
hoshi/players.py     Random / Greedy bots
hoshi/mcts.py        MCTS bot (the trustworthy one)
hoshi/strategies.py  playstyle policies + style-MCTS
hoshi/diversity.py   payoff matrix, Nash, alpha-Rank, Hodge split
hoshi/harness.py     play games + balance metrics
tests/               Python test suites
sweep.py             parallel sweeps (balance, diversity, supply, setups)
CLAUDE.md            design notes, invariants vs dials, findings
```

## Key findings (from the research engine)
- **Board ÷ drone-range ≈ 2.3** keeps both win paths alive; too sparse and the
  game decomposes into two solitaires. (9×9 → range 4, 13×13 → range 6.)
- The **setup variants** are the main source of strategic variety (rock-paper-
  scissors between commitment levels) and cost nothing in rules to learn.
- Diversity numbers are only trustworthy when measured with the **MCTS bot** —
  weak 1-ply bots overstate variety.
