"""Validate the diversity math on matrices with known structure.
If these pass, the metrics mean what we claim regardless of bot quality."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from hoshi.diversity import (nash_zero_sum, alpha_rank, hodge, entropy,
                             dominance)


def test_rps_is_maximally_cyclic():
    A = np.array([[0., 1, -1], [-1, 0, 1], [1, -1, 0]])
    h = hodge(A)
    assert h["intransitivity"] > 0.95, h["intransitivity"]
    assert h["transitivity"] < 0.05


def test_rps_nash_uniform():
    A = np.array([[0., 1, -1], [-1, 0, 1], [1, -1, 0]])
    x = nash_zero_sum(A)
    assert np.allclose(x, np.ones(3) / 3, atol=0.05), x


def test_transitive_ladder_has_no_cycle():
    r = np.array([1.0, 0.0, -1.0])
    A = r[:, None] - r[None, :]            # pure gradient flow
    h = hodge(A)
    assert h["intransitivity"] < 0.02, h["intransitivity"]


def test_transitive_nash_concentrates():
    # strategy 0 weakly dominates -> Nash puts ~all mass on it
    W = np.array([[0.5, 0.9, 0.95],
                  [0.1, 0.5, 0.9],
                  [0.05, 0.1, 0.5]])
    A = 2 * W - 1
    x = nash_zero_sum(A)
    assert x[0] > 0.9, x


def test_dominance_detected():
    W = np.array([[0.5, 0.8, 0.85],
                  [0.2, 0.5, 0.6],
                  [0.15, 0.4, 0.5]])
    dom, i = dominance(W)
    assert i == 0 and dom > 0.65


def test_entropy_bounds():
    assert abs(entropy(np.ones(4) / 4) - np.log(4)) < 1e-9
    assert entropy(np.array([1.0, 0, 0, 0])) < 1e-6


def test_alpha_rank_picks_winner_in_transitive():
    W = np.array([[0.5, 0.95, 0.98],
                  [0.05, 0.5, 0.95],
                  [0.02, 0.05, 0.5]])
    pi = alpha_rank(W)
    assert np.argmax(pi) == 0, pi


def test_alpha_rank_spreads_in_rps():
    W = np.array([[0.5, 1.0, 0.0], [0.0, 0.5, 1.0], [1.0, 0.0, 0.5]])
    pi = alpha_rank(W)
    assert entropy(pi) / np.log(3) > 0.9, pi


def test_intransitivity_bounded_after_symmetrize():
    rng = np.random.default_rng(0)
    for _ in range(20):
        W = rng.random((4, 4))            # noisy, NOT antisymmetric
        P = (W + 1 - W.T) / 2
        np.fill_diagonal(P, 0.5)
        A = 2 * P - 1
        h = hodge(A)
        assert -1e-9 <= h["intransitivity"] <= 1 + 1e-9, h["intransitivity"]
        assert abs(h["intransitivity"] + h["transitivity"] - 1) < 1e-6


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} diversity tests passed")
