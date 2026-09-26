"""Continuous search over the coded box, without a grid."""

from __future__ import annotations

import numpy as np
import pytest

from malt.engine.search import maximize, space_filling


def bowl(peak: np.ndarray):
    return lambda z: -((z - peak) ** 2).sum(axis=1)


def test_space_filling_stays_in_the_box_and_is_reproducible():
    a = space_filling(3, 256, np.random.default_rng(0))
    assert a.shape == (256, 3) and np.abs(a).max() <= 1.0
    np.testing.assert_array_equal(a, space_filling(3, 256, np.random.default_rng(0)))
    assert not np.array_equal(a, space_filling(3, 256, np.random.default_rng(1)))
    np.testing.assert_array_equal(space_filling(3, 256), space_filling(3, 256))


def test_space_filling_needs_a_power_of_two():
    with pytest.raises(ValueError, match="power of 2"):
        space_filling(2, 100)


def test_maximize_finds_an_interior_optimum_off_any_lattice():
    peak = np.array([0.3137, -0.7071, 0.1234])
    x, value = maximize(bowl(peak), 3)
    np.testing.assert_allclose(x, peak, atol=1e-4)
    assert value == pytest.approx(0.0, abs=1e-8)


def test_maximize_stops_at_the_boundary():
    x, _ = maximize(bowl(np.array([1.5, -0.2])), 2)
    np.testing.assert_allclose(x, [1.0, -0.2], atol=1e-4)


def test_maximize_is_never_worse_than_its_raw_points():
    # A spike narrower than the gap between raw points, next to a broad hill:
    # whatever the polish does, the answer is at least the best point sampled.
    def f(z):
        return np.exp(-((z - 0.9) ** 2).sum(axis=1) / 1e-6) * 5 + np.exp(-(z**2).sum(axis=1))

    raw = space_filling(2, 2048)
    _, value = maximize(f, 2)
    assert value >= f(raw).max()


def test_maximize_is_deterministic_without_an_rng():
    f = bowl(np.array([0.2, 0.4]))
    assert maximize(f, 2)[1] == maximize(f, 2)[1]
    np.testing.assert_array_equal(maximize(f, 2, np.random.default_rng(3))[0], maximize(f, 2, np.random.default_rng(3))[0])
