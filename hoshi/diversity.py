"""Measure strategic diversity of a RuleConfig over a strategy set.

The pipeline:
  1. build_payoff_matrix  -> W[i][j] = win-rate of policy i vs j
  2. antisymmetric advantage A = 2W - 1   (A = -A^T)
  3. nash_zero_sum(A)     -> maximin mixed strategy; its support/entropy = how
                             many ways to play survive when everyone plays well
  4. alpha_rank(W)        -> evolutionary stationary distribution (robust; no
                             equilibrium-selection ambiguity)
  5. hodge(A)             -> split A into transitive (skill ranking) + cyclic
                             (rock-paper-scissors) parts; intransitivity = the
                             squared-norm fraction in the cyclic part
  6. diversity_report / diversity_objective: combine into one number to MAXIMIZE,
     penalizing dominance (any policy > ~0.65 vs the field).

References this operationalizes: Balduzzi et al. "Open-ended Learning in
Symmetric Zero-sum Games" (Nash + Schur/Hodge decomposition); Omidshafiei et al.
"alpha-Rank"; Czarnecki et al. "Real World Games Look Like Spinning Tops"
(transitive spine + intransitive layer, widest at intermediate skill).
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import linprog
from .engine import RuleConfig
from .harness import play_game


# ---------------------------------------------------------------------------
def build_payoff_matrix(cfg: RuleConfig, policies, setups=None,
                        games_per_pair=8, seed=0):
    """W[i][j] = win-rate of policy i (as player 0) vs policy j (as player 1),
    averaged over color-swapped pairings so first-player bias cancels."""
    n = len(policies)
    setups = setups or ["garrison"] * n
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            wins = 0.0; games = 0
            for g in range(games_per_pair):
                # i as p0 vs j as p1
                r = play_game(cfg, policies[i], policies[j], setups[i], setups[j], seed=seed + g)
                wins += 1.0 if r.winner == 0 else (0.5 if r.winner == -1 else 0.0)
                games += 1
                # swap colors: j as p0 vs i as p1, credit i's result
                r2 = play_game(cfg, policies[j], policies[i], setups[j], setups[i], seed=seed + 500 + g)
                wins += 1.0 if r2.winner == 1 else (0.5 if r2.winner == -1 else 0.0)
                games += 1
            W[i][j] = wins / games
    return W


# ---------------------------------------------------------------------------
def nash_zero_sum(A: np.ndarray):
    """Maximin mixed strategy of the symmetric zero-sum game with antisymmetric
    payoff A (value is 0). LP: max v s.t. A^T x >= v, sum x = 1, x >= 0."""
    n = A.shape[0]
    # vars y = [x_0..x_{n-1}, v]; minimize -v
    c = np.zeros(n + 1); c[-1] = -1.0
    # for each column j: v - sum_i A[i][j] x_i <= 0
    A_ub = np.zeros((n, n + 1))
    for j in range(n):
        A_ub[j, :n] = -A[:, j]
        A_ub[j, n] = 1.0
    b_ub = np.zeros(n)
    A_eq = np.zeros((1, n + 1)); A_eq[0, :n] = 1.0
    b_eq = np.array([1.0])
    bounds = [(0, 1)] * n + [(None, None)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        return np.ones(n) / n
    x = np.clip(res.x[:n], 0, None)
    return x / x.sum() if x.sum() > 0 else np.ones(n) / n


# ---------------------------------------------------------------------------
def alpha_rank(W: np.ndarray, alpha=6.0, m=30):
    """Single-population pairwise alpha-Rank stationary distribution.
    Payoff difference between mutant t and resident r is A = 2W-1.
     rho(d) = (1-e^{-a d}) / (1-e^{-a m d}); rho(0)=1/m."""
    n = W.shape[0]
    A = 2 * W - 1
    C = np.zeros((n, n))
    for r in range(n):
        for t in range(n):
            if r == t:
                continue
            d = A[t, r]                      # advantage of mutant t over resident r
            if abs(d) < 1e-12:
                rho = 1.0 / m
            else:
                num = 1 - np.exp(-alpha * d)
                den = 1 - np.exp(-alpha * m * d)
                rho = num / den if abs(den) > 1e-15 else 1.0 / m
            C[r, t] = rho / (n - 1)
        C[r, r] = 1 - C[r].sum()
    # stationary distribution: left eigenvector for eigenvalue 1
    vals, vecs = np.linalg.eig(C.T)
    k = np.argmin(np.abs(vals - 1.0))
    pi = np.real(vecs[:, k])
    pi = np.clip(pi, 0, None)
    return pi / pi.sum() if pi.sum() > 0 else np.ones(n) / n


# ---------------------------------------------------------------------------
def hodge(A: np.ndarray):
    """Combinatorial Hodge decomposition of the advantage flow on the complete
    graph (uniform weights). r_i = mean_j A[i][j]; transitive T[i][j]=r_i-r_j;
    cyclic C = A - T (divergence-free, orthogonal to T)."""
    n = A.shape[0]
    r = A.mean(axis=1)                      # least-squares HodgeRank ratings
    T = r[:, None] - r[None, :]
    Cy = A - T
    aF2 = float((A ** 2).sum()) + 1e-12
    transitivity = float((T ** 2).sum()) / aF2
    intransitivity = float((Cy ** 2).sum()) / aF2   # == 1 - transitivity
    return dict(ratings=r, transitive=T, cyclic=Cy,
                transitivity=transitivity, intransitivity=intransitivity)


def entropy(p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1)
    return float(-(p * np.log(p)).sum())


def dominance(W: np.ndarray):
    """Best mean win-rate any single policy gets vs the rest of the field."""
    n = W.shape[0]
    mask = ~np.eye(n, dtype=bool)
    field = np.array([W[i][mask[i]].mean() for i in range(n)])
    i = int(np.argmax(field))
    return float(field[i]), i


# ---------------------------------------------------------------------------
def diversity_report(cfg: RuleConfig, policies, setups=None,
                     games_per_pair=8, seed=0):
    names = [p.name for p in policies]
    W = build_payoff_matrix(cfg, policies, setups, games_per_pair, seed)
    # W[i][j] and W[j][i] are estimated from different games, so W is only
    # approximately antisymmetric. Symmetrize into a consistent P with
    # P_ij + P_ji = 1 exactly, or the Hodge decomposition is invalid.
    P = (W + 1.0 - W.T) / 2.0
    np.fill_diagonal(P, 0.5)
    A = 2 * P - 1                       # antisymmetric by construction
    nash = nash_zero_sum(A)
    arank = alpha_rank(P)
    h = hodge(A)
    dom, dom_i = dominance(P)
    n = len(policies)
    logn = np.log(n)
    return dict(
        names=names, W=W, P=P,
        nash=nash, nash_support=int((nash > 0.05).sum()),
        nash_entropy=round(entropy(nash) / logn, 3),
        alpha_rank=arank, arank_entropy=round(entropy(arank) / logn, 3),
        intransitivity=round(h["intransitivity"], 3),
        transitivity=round(h["transitivity"], 3),
        ratings={names[i]: round(float(h["ratings"][i]), 3) for i in range(n)},
        dominance=round(dom, 3), dominant=names[dom_i],
    )


def diversity_objective(report) -> dict:
    """Higher = more diverse. Combines equilibrium spread + intransitivity,
    penalizing any dominant strategy. Edit the weights to encode your taste —
    this function literally defines 'a more varied game'."""
    spread = 0.5 * report["arank_entropy"] + 0.5 * report["nash_entropy"]
    cyclic = report["intransitivity"]
    dom_pen = max(0.0, report["dominance"] - 0.65) * 2.0
    score = 1.0 * spread + 1.0 * cyclic - dom_pen
    return dict(score=round(score, 4), spread=round(spread, 3),
                cyclic=cyclic, dominance_penalty=round(dom_pen, 3))
