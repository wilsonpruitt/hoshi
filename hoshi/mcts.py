"""MCTS bot — the workhorse for rule-tuning and strategy discovery.

Plain UCT (upper-confidence-trees) over the engine's pure State:
  * tree policy: descend by UCB1, expand one untried legal move per visit
  * default policy: a fast random rollout (lazy move sampling, no full
    `legal_moves` per ply) capped at `rollout_depth`; if the cap is hit before
    the game ends, fall back to the sign of `players._evaluate` as the result
  * backup: win/draw/loss credited to the player who *made* each edge's move

Why a custom rollout sampler instead of `legal_moves`? `legal_moves` validates
every empty point with a full board copy — fine at tree nodes (visited rarely),
ruinous inside rollouts. `_sample_rollout_move` tries a handful of shuffled
candidates and returns the first legal one, so a rollout is cheap.

State is plain data and every op returns a NEW State, so the search never
mutates the caller's position and the whole thing is trivially picklable for
`multiprocessing` sweeps later.

Strength target (CLAUDE.md): reliably beats GreedyPlayer >70%.
"""
from __future__ import annotations
import math
import random
from typing import Optional

from typing import Callable
from .engine import State, Move, legal_moves, apply_move, _try_apply, cheb
from .players import _evaluate


def _result(winner: int, player: int) -> float:
    """Reward in [0,1] for `player` given the terminal/estimated `winner`."""
    if winner == player:
        return 1.0
    if winner == -1:
        return 0.5
    return 0.0


class _Node:
    __slots__ = ("state", "parent", "move", "mover", "children", "untried", "N", "W")

    def __init__(self, state: State, parent: "Optional[_Node]", move: Optional[Move]):
        self.state = state
        self.parent = parent
        self.move = move                      # edge from parent that produced this state
        self.mover = parent.state.to_move if parent is not None else None
        self.children: dict[Move, _Node] = {}
        self.untried: Optional[list[Move]] = None   # filled lazily on first expansion
        self.N = 0
        self.W = 0.0                          # total reward from `mover`'s perspective


class MCTSPlayer:
    name = "mcts"

    def __init__(self, sims: int = 200, c: float = 1.4, rollout_depth: int = 50,
                 seed: int = 0, max_candidates: int = 40, action_width: int = 8,
                 eval_fn: "Optional[Callable[[State, int], float]]" = None,
                 name: "Optional[str]" = None):
        self.sims = sims
        self.c = c
        self.rollout_depth = rollout_depth
        self.rng = random.Random(seed)
        self.max_candidates = max_candidates
        self.action_width = action_width      # keep only the top-K eval'd moves per node
        # leaf/shortlist eval; default = the greedy baseline. Swap in a style eval
        # to make an MCTS bot that searches toward a particular *playstyle*.
        self.eval_fn = eval_fn or _evaluate
        if name:
            self.name = name

    # -- public API --------------------------------------------------------
    def choose(self, s: State) -> Move:
        moves = legal_moves(s)
        if not moves:
            return ("pass",)
        if len(moves) == 1:
            return moves[0]

        root = _Node(s.copy(), None, None)
        for _ in range(self.sims):
            leaf = self._select(root)
            winner = self._simulate(leaf.state)
            self._backprop(leaf, winner)

        # robust choice: most-visited child
        best = max(root.children.values(), key=lambda nd: nd.N)
        return best.move

    # -- tree policy -------------------------------------------------------
    def _select(self, node: _Node) -> _Node:
        while True:
            if node.state.over:
                return node
            if node.untried is None:
                node.untried = self._shortlist(node.state)
            if node.untried:
                m = node.untried.pop()
                child = _Node(apply_move(node.state, m), node, m)
                node.children[m] = child
                return child
            node = self._uct_child(node)

    def _shortlist(self, state: State) -> list[Move]:
        """Rank legal moves by 1-ply greedy eval, keep the top `action_width`.

        Returned in *ascending* score so `untried.pop()` expands the best move
        first (first-play urgency). Most drop/drone points are near-duplicates;
        pruning to a shortlist concentrates the sim budget where it matters and
        is what lifts MCTS clearly above the greedy baseline.
        """
        me = state.to_move
        scored = []
        for m in legal_moves(state):
            try:
                ns = apply_move(state, m)
            except ValueError:
                continue
            scored.append((self.eval_fn(ns, me) + self.rng.random() * 1e-6, m))
        if not scored:
            return [("pass",)]
        scored.sort()                          # ascending; pop() takes the best
        k = self.action_width
        return [m for _v, m in scored[-k:]]

    def _uct_child(self, node: _Node) -> _Node:
        # parent visit count drives exploration; children store reward for the
        # player who moved *into* them (== node.state.to_move).
        logN = math.log(node.N) if node.N > 0 else 0.0
        best, best_val = None, -1.0
        for child in node.children.values():
            exploit = child.W / child.N if child.N else 0.0
            explore = self.c * math.sqrt(logN / child.N) if child.N else float("inf")
            val = exploit + explore
            if val > best_val:
                best, best_val = child, val
        return best

    # -- default policy (rollout) -----------------------------------------
    def _simulate(self, state: State) -> int:
        s = state
        depth = 0
        while not s.over and depth < self.rollout_depth:
            s = self._sample_rollout_move(s)
            depth += 1
        if s.over:
            return s.winner
        # depth cap hit: estimate the leaf from player 0's perspective
        v = self.eval_fn(s, 0)
        return 0 if v > 0 else (1 if v < 0 else -1)

    def _sample_rollout_move(self, s: State) -> State:
        """Apply a cheap random legal move and return the resulting state.

        Tries shuffled candidate points, preferring a drone when the point is in
        range of a ready trooper, else a trooper drop. Falls back to pass. Avoids
        the full validate-every-point cost of `legal_moves`.
        """
        cfg = s.cfg
        me = s.to_move
        ready = s.my_ready_troopers(me)
        can_drop = s.reserve[me] > 0
        if not ready and not can_drop:
            return apply_move(s, ("pass",))

        empties = [(r, c) for r in range(cfg.n) for c in range(cfg.n) if (r, c) not in s.board]
        self.rng.shuffle(empties)
        tried = 0
        for pt in empties:
            in_range = ready and any(cheb(pt, tp) <= cfg.drone_range for tp in ready)
            if in_range:
                ns = _try_apply(s, ("drone", pt))
                if ns is not None:
                    return ns
            if can_drop:
                ns = _try_apply(s, ("drop_t", pt))
                if ns is not None:
                    return ns
            tried += 1
            if tried >= self.max_candidates:
                break
        return apply_move(s, ("pass",))

    # -- backup ------------------------------------------------------------
    def _backprop(self, node: _Node, winner: int) -> None:
        while node is not None:
            node.N += 1
            if node.mover is not None:
                node.W += _result(winner, node.mover)
            node = node.parent
