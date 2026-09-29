"""Batch acquisition maths on hand-built draws matrices: rows are surfaces, columns candidates."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from malt.engine.acquisition import (
    central_composite,
    continuous_greedy_q_nei,
    greedy_q_nei,
    greedy_q_ucb,
    q_expected_improvement,
    thompson_batch,
)
from malt.engine.factors import Factor, candidate_grid
from malt.engine.search import space_filling

from benchmarks.polish_tradeoff import compare


def marginal_ei(draws: np.ndarray, incumbent: np.ndarray) -> np.ndarray:
    return np.maximum(draws - incumbent[:, None], 0.0).mean(axis=0)


# q-NEI


def test_greedy_spreads_the_batch_over_rival_surfaces():
    # A is best on average. C looks second best, but only where A already wins;
    # B is the one that wins on the surface A loses. Top-2 by marginal EI picks
    # {A, C}; q-EI picks {A, B}.
    draws = np.array([[10.0, 0.0, 9.0], [7.0, 10.0, 7.0]])
    incumbent = np.array([5.0, 5.0])
    assert list(np.argsort(marginal_ei(draws, incumbent))[::-1][:2]) == [0, 2]
    assert list(greedy_q_nei(draws, incumbent, 2)) == [0, 1]


def test_the_bar_is_per_surface_not_the_mean():
    # Candidate 0 has the higher mean but never beats its surface's bar;
    # candidate 1 beats it on one surface.
    draws = np.array([[9.0, 11.0], [9.0, 0.0]])
    incumbent = np.array([10.0, 10.0])
    assert list(greedy_q_nei(draws, incumbent, 1)) == [1]


def test_single_pick_is_the_argmax_of_expected_improvement():
    rng = np.random.default_rng(0)
    draws = rng.normal(size=(200, 30))
    incumbent = rng.normal(size=200)
    assert greedy_q_nei(draws, incumbent, 1)[0] == np.argmax(marginal_ei(draws, incumbent))


@pytest.mark.parametrize("seed", range(5))
def test_greedy_batch_is_within_the_submodular_bound(seed):
    rng = np.random.default_rng(seed)
    draws = rng.normal(size=(50, 8))
    incumbent = rng.normal(size=50) + 0.5
    greedy = q_expected_improvement(draws[:, greedy_q_nei(draws, incumbent, 3)], incumbent)
    best = max(q_expected_improvement(draws[:, list(c)], incumbent) for c in itertools.combinations(range(8), 3))
    assert greedy >= (1 - 1 / math.e) * best


def test_single_draw_model_falls_back_to_the_mean():
    # One surface: after its argmax nothing improves, so the rest go by mu.
    assert list(greedy_q_nei(np.array([[3.0, 1.0, 2.0]]), np.array([0.0]), 3)) == [0, 2, 1]


def test_greedy_refuses_more_picks_than_candidates():
    with pytest.raises(ValueError, match="cannot choose 4 of 3"):
        greedy_q_nei(np.zeros((2, 3)), np.zeros(2), 4)


# q-UCB


def q_ucb(batch_draws: np.ndarray, mean: np.ndarray, beta: float) -> float:
    return float((mean + np.sqrt(beta * np.pi / 2) * np.abs(batch_draws - mean)).max(axis=1).mean())


def test_ucb_beta_trades_a_sure_thing_for_a_long_shot():
    # A: mean 1, sd 0.1. B: mean 0, sd 1. UCB is m + sqrt(beta) sd.
    rng = np.random.default_rng(0)
    draws = np.column_stack([rng.normal(1.0, 0.1, 20_000), rng.normal(0.0, 1.0, 20_000)])
    assert greedy_q_ucb(draws, 1, beta=0.0)[0] == 0
    assert greedy_q_ucb(draws, 1, beta=4.0)[0] == 1  # 0 + 2 > 1 + 0.2


def test_ucb_with_zero_beta_is_top_q_by_mean():
    draws = np.random.default_rng(1).normal(size=(100, 10))
    np.testing.assert_array_equal(greedy_q_ucb(draws, 4, beta=0.0), np.argsort(draws.mean(axis=0))[::-1][:4])


@pytest.mark.parametrize("seed", range(5))
def test_ucb_greedy_batch_is_within_the_submodular_bound(seed):
    draws = np.random.default_rng(seed).normal(size=(50, 8))
    mean = draws.mean(axis=0)
    picked = greedy_q_ucb(draws, 3, beta=2.0)
    greedy = q_ucb(draws[:, picked], mean[picked], 2.0)
    best = max(q_ucb(draws[:, list(c)], mean[list(c)], 2.0) for c in itertools.combinations(range(8), 3))
    assert greedy >= (1 - 1 / math.e) * best


def test_ucb_refuses_negative_beta():
    with pytest.raises(ValueError, match="beta"):
        greedy_q_ucb(np.zeros((2, 3)), 1, beta=-1.0)


# Thompson sampling


def test_thompson_is_reproducible_from_rng():
    draws = np.random.default_rng(0).normal(size=(100, 20))
    a = thompson_batch(draws, 5, np.random.default_rng(1))
    b = thompson_batch(draws, 5, np.random.default_rng(1))
    np.testing.assert_array_equal(a, b)
    # Every pick is the argmax of some surface.
    assert set(a) <= set(draws.argmax(axis=1))


def test_thompson_on_a_single_draw_repeats_its_argmax():
    assert list(thompson_batch(np.array([[1.0, 3.0, 2.0]]), 4, np.random.default_rng(0))) == [1, 1, 1, 1]


# Central composite design


def test_face_centred_design_has_corners_axials_and_centre_runs():
    design = central_composite(np.zeros(2), radius=0.5, n_center=3)
    assert design.shape == (4 + 4 + 3, 2)
    np.testing.assert_allclose(np.abs(design[:4]), 0.5)
    np.testing.assert_allclose(np.sort(np.abs(design[4:8]), axis=1), [[0.0, 0.5]] * 4)
    np.testing.assert_allclose(design[8:], 0.0)


def test_design_at_the_boundary_is_moved_inward_whole():
    design = central_composite(np.array([1.0, -0.9]), radius=0.5, n_center=1)
    np.testing.assert_allclose(design[-1], [0.5, -0.5])
    assert np.abs(design).max() <= 1.0
    assert len(np.unique(design, axis=0)) == len(design)  # nothing clipped onto a face


def test_rotatable_axials_reach_further_than_the_corners():
    alpha = 2 ** (3 / 4)
    design = central_composite(np.zeros(3), radius=0.4, alpha=alpha)
    np.testing.assert_allclose(np.abs(design[8:]).max(axis=1), 0.4 * alpha)
    np.testing.assert_allclose(design.mean(axis=0), 0.0, atol=1e-12)


@pytest.mark.parametrize(("radius", "alpha"), [(0.0, 1.0), (0.6, 2.0), (1.2, 0.5)])
def test_design_that_cannot_fit_is_refused(radius, alpha):
    with pytest.raises(ValueError, match="radius"):
        central_composite(np.zeros(2), radius, alpha)


# Candidate grid


def test_candidate_grid_is_full_factorial_and_geometric_on_log_scale():
    factors = (Factor("a", 0.0, 10.0), Factor("b", 0.1, 10.0, scale="log"))
    grid = candidate_grid(factors, 3)
    assert list(grid.columns) == ["a", "b"] and len(grid) == 9
    np.testing.assert_allclose(sorted(set(grid["a"])), [0.0, 5.0, 10.0])
    np.testing.assert_allclose(sorted(set(grid["b"])), [0.1, 1.0, 10.0])


def test_candidate_grid_needs_two_levels():
    with pytest.raises(ValueError, match="at least 2 levels"):
        candidate_grid((Factor("a", 0.0, 1.0),), 1)


# Continuous q-NEI: the same greedy rule, but picks are not confined to a pool


def bumps(peaks: np.ndarray, heights: np.ndarray):
    """`S` Gaussian surfaces with closed-form derivatives: `mu_s(z) = h_s exp(-||z - p_s||^2)`.

    A stand-in for a posterior over surfaces, so the acquisition can be tested
    against exact values and exact gradients without fitting anything.
    """

    def mu(z: np.ndarray) -> np.ndarray:
        offset = np.atleast_2d(z)[None, :, :] - peaks[:, None, :]
        return heights[:, None] * np.exp(-(offset**2).sum(axis=2))

    def d_mu(z: np.ndarray) -> np.ndarray:
        offset = np.atleast_2d(z)[None, :, :] - peaks[:, None, :]
        return mu(z)[:, :, None] * (-2.0 * offset)

    return mu, d_mu


def three_surfaces(k: int = 2, s: int = 24, spread: float = 0.45, seed: int = 0):
    rng = np.random.default_rng(seed)
    peaks = rng.normal(0.25, spread, (s, k))
    heights = rng.uniform(1.0, 3.0, s)
    return bumps(peaks, heights)


def test_the_objective_is_q_expected_improvement_per_candidate():
    """The docstring's claim, asserted numerically rather than shared textually.

    `continuous_greedy_q_nei` scores the scan with a vectorized
    `max(0, mu - bar)` mean; `q_expected_improvement` maxes over a batch first.
    For a one-point batch they are the same formula, and this is what stops the
    two drifting apart.

    Not bitwise equal: numpy's pairwise summation adds the draws in a different
    order for a contiguous column than for a 2-D reduction, so the two differ by
    an ulp. A few ulps is the whole budget — there is no smoothing or
    approximation in either, which is what makes this tighter than comparing
    against a smoothed log-EI implementation would be.
    """
    mu, _ = three_surfaces()
    pool = space_filling(2, 64)
    values = mu(pool)
    bar = np.full(values.shape[0], 0.8)

    vectorized = np.maximum(values - bar[:, None], 0.0).mean(axis=0)
    per_column = np.array([q_expected_improvement(values[:, [j]], bar) for j in range(values.shape[1])])
    np.testing.assert_allclose(vectorized, per_column, rtol=8 * np.finfo(float).eps, atol=0.0)


@pytest.mark.parametrize("q", [1, 3, 6])
def test_continuous_picks_beat_the_best_the_pool_can_offer(q):
    """The reason for the change: a discrete pick is capped by the pool's resolution."""
    mu, d_mu = three_surfaces()
    bar = np.full(24, 0.8)
    rng = np.random.default_rng(1)

    pool = space_filling(2, 4096, np.random.default_rng(1))
    discrete = pool[greedy_q_nei(mu(pool), bar, q)]
    polished = continuous_greedy_q_nei(mu, d_mu, pool, bar, q)

    # Exact, not approximate: each pick starts from the discrete rule's own
    # choice, and L-BFGS-B only moves off a start to improve on it.
    assert q_expected_improvement(mu(polished), bar) >= q_expected_improvement(mu(discrete), bar)


def test_a_single_pick_beats_a_dense_brute_force_grid():
    mu, d_mu = three_surfaces()
    bar = np.full(24, 0.8)
    grid = np.array(list(itertools.product(np.linspace(-1.0, 1.0, 201), repeat=2)))
    best_on_grid = np.maximum(mu(grid) - bar[:, None], 0.0).mean(axis=0).max()

    picked = continuous_greedy_q_nei(mu, d_mu, space_filling(2, 2048, np.random.default_rng(0)), bar, 1)
    assert q_expected_improvement(mu(picked), bar) >= best_on_grid - 1e-9


def test_the_gradient_is_finite_nonzero_and_matches_central_differences():
    mu, d_mu = three_surfaces()
    bar = np.full(24, 0.8)
    z = np.array([[0.15, -0.05]])
    assert np.maximum(mu(z) - bar[:, None], 0.0).mean() > 0.0, "need a point where q-NEI is positive"

    def acquisition(zz):
        return np.maximum(mu(zz) - bar[:, None], 0.0).mean(axis=0)

    gradient = ((mu(z) > bar[:, None])[:, :, None] * d_mu(z)).mean(axis=0)
    assert np.isfinite(gradient).all() and np.abs(gradient).min() > 0.0

    h = 1e-6
    expected = np.column_stack(
        [(acquisition(z + h * np.eye(2)[d]) - acquisition(z - h * np.eye(2)[d])) / (2 * h) for d in range(2)]
    )
    np.testing.assert_allclose(gradient, expected, rtol=1e-6)


@pytest.mark.parametrize("offset", [0.25, 3.0], ids=["interior", "outside-the-box"])
def test_picks_stay_inside_the_coded_box(offset):
    rng = np.random.default_rng(2)
    mu, d_mu = bumps(rng.normal(offset, 0.3, (16, 3)), rng.uniform(1.0, 3.0, 16))
    picks = continuous_greedy_q_nei(mu, d_mu, space_filling(3, 2048, rng), np.full(16, 0.5), 5)
    assert picks.shape == (5, 3)
    assert np.abs(picks).max() <= 1.0


def test_a_batch_does_not_stack_on_one_point():
    """The bar rising after each pick is what spreads the batch.

    q-NEI may legitimately replicate where the posterior is confident, so this
    uses surfaces that genuinely disagree about where the peak is — there, a
    batch collapsing to one point would mean the bar was not being raised.
    """
    mu, d_mu = three_surfaces(s=32, spread=0.6, seed=4)
    picks = continuous_greedy_q_nei(
        mu, d_mu, space_filling(2, 2048, np.random.default_rng(5)), np.full(32, 0.8), 4
    )
    distances = [
        np.linalg.norm(picks[i] - picks[j]) for i, j in itertools.combinations(range(len(picks)), 2)
    ]
    assert min(distances) > 1e-3


def test_a_single_draw_model_still_returns_distinct_points():
    """A point-estimate surrogate has nothing left to improve on after one pick.

    `_greedy_batch` fills the rest by highest mean; the continuous rule falls
    back to the same ordering over its space-filling pool rather than repeating
    one point `q` times or erroring.
    """
    mu, d_mu = bumps(np.array([[0.2, -0.3]]), np.array([2.0]))
    picks = continuous_greedy_q_nei(
        mu, d_mu, space_filling(2, 2048, np.random.default_rng(6)), np.array([0.5]), 4
    )
    assert len({tuple(p) for p in picks}) == 4
    assert np.abs(picks).max() <= 1.0


@pytest.mark.parametrize(
    ("q", "incumbent", "match"),
    [
        (0, np.zeros(24), "cannot choose 0"),
        (99, np.zeros(24), "cannot choose 99 of 64"),
        (2, np.zeros((24, 1)), "one value per draw"),
    ],
)
def test_continuous_q_nei_rejects_bad_arguments(q, incumbent, match):
    mu, d_mu = three_surfaces()
    with pytest.raises(ValueError, match=match):
        continuous_greedy_q_nei(mu, d_mu, space_filling(2, 64), incumbent, q)


def test_polishing_matters_more_as_the_candidate_pool_thins_out():
    """The reason `continuous_greedy_q_nei` exists, guarded so the claim can't rot.

    A pool of `n` Sobol points gives `n ** (1/k)` levels per axis, so the
    discrete pick's accuracy degrades with the number of factors while the
    polished one does not. At 3 factors the pool is dense enough that polishing
    is worth almost nothing; at 7 it is not. `benchmarks/polish_tradeoff.py`
    is the same measurement across more factor counts, and documents the two
    ways to measure it wrong.
    """
    low = compare(3, q=4, draws=128, n_candidates=1024, seed=0)
    high = compare(7, q=4, draws=128, n_candidates=1024, seed=0)

    assert low["levels_per_axis"] > high["levels_per_axis"]
    assert low["gain"] < 0.02, "a dense pool leaves almost nothing to polish"
    assert high["gain"] > 4 * max(low["gain"], 1e-3), "a thin pool should leave a lot"
    # Interior batches, so this is search accuracy and not corner-seeking.
    assert max(low["mean_abs_z"], high["mean_abs_z"]) < 0.6
