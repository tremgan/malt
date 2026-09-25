"""Baseline surrogates: posterior correctness and the SurrogateModel contract."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from malt.simulation.oracle import GammaLikelihood, Oracle
from malt.active_learning.surrogates import BayesianLinearRegression, GammaGLMSurrogate
from malt.engine.factors import Factor

FACTORS = (Factor("x1", 0.0, 10.0), Factor("x2", 0.1, 10.0, scale="log"))


def coded(x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return FACTORS[0].encode(x["x1"].to_numpy()), FACTORS[1].encode(x["x2"].to_numpy())


def design_points(n: int, rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame(
        {"x1": FACTORS[0].decode(rng.uniform(-1, 1, n)), "x2": FACTORS[1].decode(rng.uniform(-1, 1, n))}
    )


def linear_data(n: int, seed: int, noise: float = 0.5) -> pd.DataFrame:
    """y = 8 + 10 z1 - 5 z2 + Normal(0, noise), on coded units."""
    rng = np.random.default_rng(seed)
    x = design_points(n, rng)
    z1, z2 = coded(x)
    return x.assign(y=8 + 10 * z1 - 5 * z2 + rng.normal(0, noise, n))


# Bayesian linear regression


def test_blr_recovers_known_coefficients_and_noise():
    m = BayesianLinearRegression(FACTORS, "linear").condition(
        linear_data(400, seed=0), np.random.default_rng(1)
    )
    assert m.coef_mean is not None and m.noise_scale is not None and m.noise_shape is not None
    np.testing.assert_allclose(m.coef_mean, [8, 10, -5], atol=0.1)
    assert np.sqrt(m.noise_scale / (m.noise_shape - 1)) == pytest.approx(0.5, rel=0.1)


def test_blr_credible_intervals_are_calibrated():
    # The reference prior's intervals match OLS confidence intervals, so 90%
    # intervals should cover the truth ~90% of the time. 200 datasets gives a
    # standard error of ~0.02 per coefficient.
    covered = []
    for trial in range(200):
        m = BayesianLinearRegression(FACTORS, "linear", n_draws=4000).condition(
            linear_data(15, seed=trial), np.random.default_rng(10_000 + trial)
        )
        assert m.coef_draws is not None
        lo, hi = np.percentile(m.coef_draws, [5, 95], axis=0)
        covered.append((lo <= [8, 10, -5]) & ([8, 10, -5] <= hi))
    assert np.all(np.abs(np.mean(covered, axis=0) - 0.9) < 0.07)


def test_blr_conditioning_is_associative():
    d = linear_data(20, seed=3)
    rng = np.random.default_rng(0)
    stepwise = BayesianLinearRegression(FACTORS, "quadratic").condition(d.iloc[:8], rng).condition(d.iloc[8:], rng)
    at_once = BayesianLinearRegression(FACTORS, "quadratic").condition(d, rng)
    for field in ("coef_mean", "coef_cov_unscaled", "noise_shape", "noise_scale"):
        np.testing.assert_allclose(getattr(stepwise, field), getattr(at_once, field))


def test_blr_condition_leaves_self_unchanged():
    d = linear_data(20, seed=4)
    prior = BayesianLinearRegression(FACTORS)
    first = prior.condition(d.iloc[:10], np.random.default_rng(0))
    first.condition(d.iloc[10:], np.random.default_rng(0))
    assert prior.data is None and prior.coef_draws is None
    assert first.data is not None and len(first.data) == 10


@pytest.mark.parametrize(("features", "n_coefs"), [("linear", 3), ("quadratic", 6)])
def test_blr_feature_sets(features, n_coefs):
    m = BayesianLinearRegression(FACTORS, features).condition(linear_data(20, seed=5), np.random.default_rng(0))
    assert m.coef_draws is not None and m.coef_draws.shape == (2000, n_coefs)


def test_blr_refuses_unidentifiable_data():
    with pytest.raises(ValueError, match="cannot identify 6 coefficients"):
        BayesianLinearRegression(FACTORS, "quadratic").condition(linear_data(6, seed=6), np.random.default_rng(0))


# Gamma GLM (MCMC: a few seconds per fit, so fits are shared)


@pytest.fixture(scope="module")
def gamma_data() -> pd.DataFrame:
    """Gamma draws around log mu = 1 + 0.5 z1 - 0.3 z2, alpha = 20."""

    def latent(x: pd.DataFrame) -> np.ndarray:
        z1, z2 = coded(x)
        return 1 + 0.5 * z1 - 0.3 * z2

    rng = np.random.default_rng(0)
    x = design_points(30, rng)
    y = Oracle(latent, GammaLikelihood(20.0)).query(x, rng)
    return x.join(y.set_axis(x.index))


@pytest.fixture(scope="module")
def linear_glm(gamma_data) -> GammaGLMSurrogate:
    return GammaGLMSurrogate(FACTORS, terms=("linear",)).condition(gamma_data, np.random.default_rng(1))


def test_glm_recovers_known_coefficients(linear_glm):
    assert linear_glm.reliable
    assert linear_glm.fit is not None
    assert linear_glm.fit.term_names == ("x1", "x2")
    posterior = linear_glm.fit.posterior["posterior"]
    assert float(posterior["intercept"].mean()) == pytest.approx(1.0, abs=0.15)
    np.testing.assert_allclose(posterior["beta"].mean(("chain", "draw")), [0.5, -0.3], atol=0.15)


def test_glm_draws_are_positive_and_stable(linear_glm, gamma_data):
    x = gamma_data.iloc[:5]
    draws = linear_glm.sample(x)
    assert draws.shape == (4000, 5)
    assert (draws > 0).all()
    np.testing.assert_array_equal(draws, linear_glm.sample(x))


def test_glm_is_reproducible_from_rng(linear_glm, gamma_data):
    again = GammaGLMSurrogate(FACTORS, terms=("linear",)).condition(gamma_data, np.random.default_rng(1))
    x = gamma_data.iloc[:5]
    np.testing.assert_array_equal(again.sample(x), linear_glm.sample(x))


# Shared behaviour


@pytest.mark.parametrize(
    "prior", [BayesianLinearRegression(FACTORS), GammaGLMSurrogate(FACTORS)], ids=["blr", "glm"]
)
def test_unconditioned_model_is_unreliable_and_refuses_to_sample(prior):
    assert not prior.reliable
    with pytest.raises(ValueError, match="condition on the seed first"):
        prior.sample(design_points(3, np.random.default_rng(0)))
