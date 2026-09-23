"""Simulated oracles — a known ground-truth surface plus an observation model.

An oracle answers the one question the active-learning loop asks of the
physical world: run this composition, what comes back? It is split in two so
that the structure and the noise can vary independently:

    latent f(x)  ->  likelihood p(y | f)  ->  y

`f` is the latent function value — the unconstrained scale on which the
response surface (and, later, a batch offset) is additive. The likelihood owns
both the inverse link that maps `f` onto a legal mean and the draw around it.
That composition is what lets the same interface cover additive Gaussian noise
and a GLM-style generative model: additive noise is simply the identity-link
case, where `f` and the mean coincide.

Draws read numpy's global random state; seed a campaign with
`np.random.seed` rather than per-likelihood.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from typing import Protocol

import numpy as np
import pandas as pd

__all__ = [
    "GammaLikelihood",
    "GaussianLikelihood",
    "LatentFunction",
    "Likelihood",
    "Oracle",
    "constant_latent",
]


class Likelihood(Protocol):
    """Defines the noise model for the underlying function. Specifically,
    it defines the conditional distribution p(y | f(x)) for the observed data y given the latent function value f(x)."""

    def sample(self, latent: np.ndarray) -> np.ndarray:
        """Samples from the conditional distribution p(y | f(x)) for the observed data y given the latent function value f(x)."""
        ...

    def mean(self, latent: np.ndarray) -> np.ndarray:
        """Returns the mean of the conditional distribution p(y | f(x)) for the observed data y given the latent function value f(x)."""
        ...


@dataclass(frozen=True, slots=True)
class GaussianLikelihood:
    """Additive, constant-variance noise under an identity link: `y ~ Normal(f, sigma)`.

    The degenerate case of the link/likelihood split — `f` *is* the mean — which
    makes it the baseline for checking that the composition behaves when the
    link does nothing. It puts mass on negative values, so it is not a
    defensible model of biomass; use it to exercise the additive path, not to
    generate data a Gamma GLM is then asked to recover.
    """

    sigma: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.sigma) or self.sigma <= 0:
            raise ValueError(f"sigma must be finite and > 0, got {self.sigma}")

    def sample(self, latent: np.ndarray) -> np.ndarray:
        return np.random.normal(self.mean(latent), self.sigma)

    def mean(self, latent: np.ndarray) -> np.ndarray:
        return np.asarray(latent, dtype=float)


@dataclass(frozen=True, slots=True)
class GammaLikelihood:
    """Constant-CV positive noise under a log link: `y ~ Gamma(alpha, alpha / exp(f))`.

    Deliberately the same parameterization as `glm.fit_gamma_glm` — one shape,
    varying rate, so `Var(y) = mu^2 / alpha` and the coefficient of variation is
    `1 / sqrt(alpha)` everywhere. That correspondence is the point: data
    generated here is drawn from exactly the family the model assumes, so a
    recovery test isolates the inference from any family mismatch. To simulate
    a *mis*specified campaign, vary the latent function rather than this.

    `alpha` is the same weakly-identified shape the model has to estimate; at
    27-point dataset sizes its prior materially moves the posterior, so pick a
    ground truth here knowing the fit will be judged against it.
    """

    alpha: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError(f"alpha must be finite and > 0, got {self.alpha}")

    def sample(self, latent: np.ndarray) -> np.ndarray:
        # numpy parameterizes Gamma by shape and scale; scale = mu / alpha is
        # the reciprocal of PyMC's rate, beta = alpha / mu.
        return np.random.gamma(shape=self.alpha, scale=self.mean(latent) / self.alpha)

    def mean(self, latent: np.ndarray) -> np.ndarray:
        return np.exp(np.asarray(latent, dtype=float))



type LatentFunction = Callable[[pd.DataFrame], np.ndarray]
"""The true response surface, on the likelihood's link scale.

`f(x)` is not the expected measurement. It is the unconstrained quantity the
likelihood's inverse link turns into one: under `GammaLikelihood`, `f` is log
biomass, so `f = 1.6` at some composition means an expected biomass of
`exp(1.6) ~ 5 g/L`, around which the Gamma then scatters. Under
`GaussianLikelihood` the link is the identity and `f` *is* the mean — which is
the only reason the distinction is easy to lose.

Three consequences, and together they are why this scale is worth naming:

- Structure adds up here. A response surface, a batch offset and a drift term
  all sum on this scale. Past the link they multiply.
- Nothing here is bounded. A surface can be written without worrying about
  positivity and still generate legal biomass.
- It is the same quantity the model estimates — `glm.py`'s
  `intercept + X @ beta` — so a ground truth expressed in those terms is
  directly comparable to the posterior rather than needing translation.

Takes a frame of design points in real units, one column per factor, so that
an implementation can hand it straight to `glm.build_design_matrix` and
inherit the coded encoding and term order rather than restating them.
"""


def constant_latent(value: float) -> LatentFunction:
    """A flat surface — the null against which structure should be visible.

    Useful for confirming that acquisition does not invent a peak out of noise
    and that the epistemic uncertainty map stays roughly uniform.
    """
    return lambda x: np.full(len(x), value, dtype=float)

def quadratic_latent(coeffs: np.ndarray, intercept: float = 0.0) -> LatentFunction:
    """A quadratic surface in the input space.

    Useful for testing the oracle's ability to capture curvature and for
    generating synthetic data with known structure.
    """
    def latent_function(x: pd.DataFrame) -> np.ndarray:
        # Assuming x has two columns for a 2D quadratic surface
        if x.shape[1] != len(coeffs):
            raise ValueError(f"Expected {len(coeffs)} features, got {x.shape[1]}")
        return intercept + np.sum(coeffs * (x ** 2), axis=1)
    
    return latent_function

def rbf_gp_latent(length_scale: float, variance: float) -> LatentFunction:
    """A random latent function sampled from a Radial Basis Function (RBF) Gaussian Process.

    Useful for generating smooth, non-linear surfaces with controlled length scale and variance.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF

    kernel = variance * RBF(length_scale=length_scale)
    gp = GaussianProcessRegressor(kernel=kernel)

    def latent_function(x: pd.DataFrame) -> np.ndarray:
        # Fit the GP to some random points and predict on the input x
        random_points = np.random.rand(10, x.shape[1])  # 10 random training points
        random_values = np.random.rand(10)  # Random values for those points
        gp.fit(random_points, random_values)
        return gp.predict(x)

    return latent_function


@dataclass(frozen=True, slots=True)
class Oracle:

    latent: LatentFunction
    likelihood: Likelihood

    def sample(self, x: pd.DataFrame) -> np.ndarray:
        """Run the oracle on a batch of inputs `x`, returning the observed outputs `y`."""
        latent_values = self.latent(x)
        return self.likelihood.sample(latent_values)

    def query(self, x: pd.DataFrame) -> np.ndarray:
        """Portemanteau of `sample` and `mean`: run the oracle on a batch of inputs `x`, returning the observed outputs `y`."""
        return self.sample(x)

    def mean(self, x: pd.DataFrame) -> np.ndarray:
        """Run the oracle on a batch of inputs `x`, returning the ground truth mean for observed outputs `y`."""
        latent_values = self.latent(x)
        return self.likelihood.mean(latent_values)



