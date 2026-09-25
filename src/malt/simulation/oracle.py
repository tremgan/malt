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

Draws come from a `np.random.Generator` passed to each call, never from
numpy's global state, so a caller can give the oracle its own stream.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from malt.active_learning.actors import Environment
from malt.engine.factors import Factor

__all__ = [
    "GammaLikelihood",
    "GaussianLikelihood",
    "LatentFunction",
    "Likelihood",
    "Oracle",
    "constant_latent",
    "gp_sampled_latent",
    "quadratic_latent",
]


class Likelihood(ABC):
    """Defines the noise model for the underlying function. Specifically,
    it defines the conditional distribution p(y | f(x)) for the observed data y given the latent function value f(x)."""

    __slots__ = ()

    @abstractmethod
    def sample(self, latent: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Samples from the conditional distribution p(y | f(x)) for the observed data y given the latent function value f(x)."""
        ...

    @abstractmethod
    def mean(self, latent: np.ndarray) -> np.ndarray:
        """Returns the mean of the conditional distribution p(y | f(x)) for the observed data y given the latent function value f(x)."""
        ...


@dataclass(frozen=True, slots=True)
class GaussianLikelihood(Likelihood):
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

    def sample(self, latent: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(self.mean(latent), self.sigma)

    def mean(self, latent: np.ndarray) -> np.ndarray:
        return np.asarray(latent, dtype=float)


@dataclass(frozen=True, slots=True)
class GammaLikelihood(Likelihood):
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

    def sample(self, latent: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        # numpy parameterizes Gamma by shape and scale; scale = mu / alpha is
        # the reciprocal of PyMC's rate, beta = alpha / mu.
        return rng.gamma(shape=self.alpha, scale=self.mean(latent) / self.alpha)

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

def quadratic_latent(
    factors: Sequence[Factor],
    *,
    optimum: Mapping[str, float],
    peak: float,
    curvature: float | Sequence[float] | np.ndarray,
) -> LatentFunction:
    """A single peak: `f(z) = peak - (z - z*)' A (z - z*)` in coded units.

    The shape a growth response is expected to have near an optimum, and
    exactly the family the Gamma GLM fits — linear, squared and interaction
    terms on the coded scale — so it is the in-family benchmark surface.

    `optimum` is the peak's location in real units, one entry per factor; it
    must lie inside every declared range. `peak` is the latent value there
    (log biomass under a log link, so `np.log(12.0)` is a 12 g/L peak).
    `curvature` is `A`: a scalar or one value per factor for an axis-aligned
    peak, or a full symmetric matrix whose off-diagonal entries tilt it — an
    interaction between factors. It must be positive definite, or the surface
    has no peak. Entries are drops in `f` per squared coded unit: 1.0 on a
    factor means moving from the optimum to the edge of a range centred on it
    costs one unit of log biomass, a factor of e.
    """
    factors = tuple(factors)
    k = len(factors)
    missing = [f.name for f in factors if f.name not in optimum]
    if missing:
        raise ValueError(f"optimum is missing factors: {missing}")
    outside = [f.name for f in factors if not f.contains(np.array(optimum[f.name]))]
    if outside:
        raise ValueError(f"optimum lies outside the declared range of: {outside}")
    z_star = np.array([f.encode(np.array(optimum[f.name])) for f in factors], dtype=float)

    A = np.asarray(curvature, dtype=float)
    if A.ndim < 2:
        A = np.diag(np.broadcast_to(A, (k,)))
    if A.shape != (k, k) or not np.allclose(A, A.T):
        raise ValueError(f"curvature must be a scalar, {k} values, or a symmetric {k}x{k} matrix")
    if np.linalg.eigvalsh(A).min() <= 0:
        raise ValueError("curvature must be positive definite, or the surface has no peak")

    def latent_function(x: pd.DataFrame) -> np.ndarray:
        d = np.column_stack([f.encode(x[f.name].to_numpy()) for f in factors]) - z_star
        return peak - np.einsum("ni,ij,nj->n", d, A, d)

    return latent_function


def gp_sampled_latent(
    factors: Sequence[Factor],
    rng: np.random.Generator,
    *,
    lengthscale: float = 0.5,
    sd: float = 0.5,
    mean: float = 0.0,
    n_features: int = 1000,
) -> LatentFunction:
    """One fixed surface drawn from a Gaussian process with an RBF kernel.

    The draw happens once, here, from `rng`; the returned function is then
    deterministic and defined everywhere. Sampling GP values at query time
    instead would hand back a different surface on every call, so the oracle's
    `query` and `mean` would disagree about the truth.

    It uses random Fourier features: `n_features` random cosines whose sum is,
    approximately, a draw from a zero-mean GP with kernel
    `sd^2 exp(-|z - z'|^2 / (2 lengthscale^2))`, shifted by `mean`. Inputs are
    encoded through `factors` first, so `lengthscale` is in coded units — 0.5
    is a quarter of every factor's range, linear or log alike.

    On a log link, `mean` is log baseline biomass and `sd` the typical
    log-scale deviation from it: `sd=0.5` puts most of the surface within a
    factor of about e (2.7) of the baseline.
    """
    factors = tuple(factors)
    frequencies = rng.normal(0.0, 1.0 / lengthscale, (n_features, len(factors)))
    phases = rng.uniform(0.0, 2 * np.pi, n_features)
    weights = rng.normal(0.0, 1.0, n_features)
    amplitude = sd * np.sqrt(2.0 / n_features)

    def latent_function(x: pd.DataFrame) -> np.ndarray:
        z = np.column_stack([f.encode(x[f.name].to_numpy()) for f in factors])
        return mean + amplitude * np.cos(z @ frequencies.T + phases) @ weights

    return latent_function

    
@dataclass(frozen=True, slots=True)
class Oracle(Environment):

    latent: LatentFunction
    likelihood: Likelihood

    def sample(self, x: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
        """Run the oracle on a batch of inputs `x`, returning the observed outputs `y`."""
        latent_values = self.latent(x)
        return self.likelihood.sample(latent_values, rng)

    def query(self, x: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        """Portemanteau of `sample` and `mean`: run the oracle on a batch of inputs `x`, returning the observed outputs `y`."""
        return pd.DataFrame({"y": self.sample(x, rng)})

    def mean(self, x: pd.DataFrame) -> np.ndarray:
        """Run the oracle on a batch of inputs `x`, returning the ground truth mean for observed outputs `y`."""
        latent_values = self.latent(x)
        return self.likelihood.mean(latent_values)



