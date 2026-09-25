"""Batch acquisition maths on hand-built draws matrices: rows are surfaces, columns candidates."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from malt.engine.acquisition import (
    central_composite,
    greedy_q_nei,
    greedy_q_ucb,
    q_expected_improvement,
    thompson_batch,
)
from malt.engine.factors import Factor, candidate_grid


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
