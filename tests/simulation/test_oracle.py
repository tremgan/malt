"""Latent surfaces: the peaked quadratic, and the GP sample with its intended kernel."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from malt.benchmark.oracle import GammaLikelihood, Oracle, gp_sampled_latent, quadratic_latent
from malt.engine.factors import Factor
from malt.engine.glm import build_design_matrix

FACTORS = (Factor("a", 0.0, 10.0), Factor("b", 0.1, 10.0, scale="log"))
X = pd.DataFrame({"a": [1.0, 5.0, 9.0], "b": [0.2, 1.0, 8.0]})


def at_coded(*points: tuple[float, float]) -> pd.DataFrame:
    za, zb = np.array(points, dtype=float).T
    return pd.DataFrame({"a": FACTORS[0].decode(za), "b": FACTORS[1].decode(zb)})


def test_surface_is_fixed_once_drawn():
    # The old sampler redrew the surface on every call, so query and mean
    # disagreed about the truth.
    f = gp_sampled_latent(FACTORS, np.random.default_rng(0))
    np.testing.assert_array_equal(f(X), f(X))


def test_surface_is_reproducible_from_rng():
    a = gp_sampled_latent(FACTORS, np.random.default_rng(0))(X)
    b = gp_sampled_latent(FACTORS, np.random.default_rng(0))(X)
    c = gp_sampled_latent(FACTORS, np.random.default_rng(1))(X)
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)


def test_surfaces_follow_the_rbf_kernel_in_coded_units():
    # Across many drawn surfaces: variance sd^2 = 0.25, and correlation
    # exp(-d^2 / (2 ell^2)) at coded distances d = ell and d = 2 ell.
    points = at_coded((0, 0), (0.5, 0), (0, 1.0))
    draws = np.array(
        [gp_sampled_latent(FACTORS, np.random.default_rng(s), sd=0.5, lengthscale=0.5)(points)
         for s in range(4000)]
    )
    corr = np.corrcoef(draws.T)
    assert draws[:, 0].var() == pytest.approx(0.25, rel=0.1)
    assert corr[0, 1] == pytest.approx(np.exp(-0.5), abs=0.05)
    assert corr[0, 2] == pytest.approx(np.exp(-2.0), abs=0.05)


def test_lengthscale_is_the_same_on_a_log_factor():
    # Moving one lengthscale along the log factor decorrelates exactly as much
    # as along the linear one, because distance is measured after encoding.
    points = at_coded((0, 0), (0.5, 0), (0, 0.5))
    draws = np.array(
        [gp_sampled_latent(FACTORS, np.random.default_rng(s), lengthscale=0.5)(points)
         for s in range(4000)]
    )
    corr = np.corrcoef(draws.T)
    assert corr[0, 1] == pytest.approx(corr[0, 2], abs=0.05)


def test_mean_shifts_the_surface():
    base = gp_sampled_latent(FACTORS, np.random.default_rng(0))(X)
    shifted = gp_sampled_latent(FACTORS, np.random.default_rng(0), mean=np.log(4.0))(X)
    np.testing.assert_allclose(shifted - base, np.log(4.0))


def test_gamma_oracle_on_a_gp_surface_is_positive():
    oracle = Oracle(gp_sampled_latent(FACTORS, np.random.default_rng(0)), GammaLikelihood(20.0))
    y = oracle.query(X, np.random.default_rng(1))
    assert list(y.columns) == ["y"] and len(y) == len(X)
    assert (y["y"] > 0).all() and (oracle.mean(X) > 0).all()


# Peaked quadratic

OPTIMUM = {"a": 6.0, "b": 2.0}
TILTED = np.array([[1.0, -0.4], [-0.4, 0.8]])


def dense_grid(k: int = 201) -> pd.DataFrame:
    z = np.linspace(-1, 1, k)
    za, zb = np.meshgrid(z, z)
    return at_coded(*zip(za.ravel(), zb.ravel()))


def test_quadratic_peaks_at_the_optimum_with_the_peak_value():
    f = quadratic_latent(FACTORS, optimum=OPTIMUM, peak=2.5, curvature=TILTED)
    at_optimum = pd.DataFrame({k: [v] for k, v in OPTIMUM.items()})
    assert f(at_optimum)[0] == pytest.approx(2.5)
    grid = dense_grid()
    assert f(grid).max() <= 2.5 + 1e-12
    best = grid.iloc[int(np.argmax(f(grid)))]
    assert best["a"] == pytest.approx(OPTIMUM["a"], rel=0.02)
    assert best["b"] == pytest.approx(OPTIMUM["b"], rel=0.05)


def test_quadratic_curvature_is_per_squared_coded_unit():
    # 1.0 on an axis-aligned factor: one coded unit from the optimum costs 1.0.
    f = quadratic_latent(FACTORS, optimum={"a": 5.0, "b": 1.0}, peak=0.0, curvature=[1.0, 4.0])
    np.testing.assert_allclose(f(at_coded((1, 0), (0, 0.5), (-1, 0))), [-1.0, -1.0, -1.0])


def test_quadratic_is_in_the_glms_family():
    # Exactly linear in the GLM's design matrix: least squares on the full
    # quadratic basis reproduces it with zero residual.
    f = quadratic_latent(FACTORS, optimum=OPTIMUM, peak=2.5, curvature=TILTED)
    grid = dense_grid(21)
    X = np.column_stack([np.ones(len(grid)), build_design_matrix(grid, FACTORS).X])
    _, residual, *_ = np.linalg.lstsq(X, f(grid), rcond=None)
    assert residual[0] == pytest.approx(0.0, abs=1e-18)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"optimum": {"a": 6.0}}, "missing"),
        ({"optimum": {"a": 60.0, "b": 2.0}}, "outside"),
        ({"curvature": [1.0, -0.5]}, "positive definite"),
        ({"curvature": np.array([[1.0, 2.0], [2.0, 1.0]])}, "positive definite"),
        ({"curvature": np.array([[1.0, 0.3], [0.0, 1.0]])}, "symmetric"),
    ],
)
def test_quadratic_rejects_invalid_input(kwargs, match):
    args = {"optimum": OPTIMUM, "peak": 0.0, "curvature": 1.0} | kwargs
    with pytest.raises(ValueError, match=match):
        quadratic_latent(FACTORS, **args)
