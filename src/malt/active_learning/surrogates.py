"""Baseline surrogate models for benchmarking.

Together with the main model — the quadratic Gamma GLM — these make a 2x2
ablation, so a benchmark can say what each ingredient buys:

                      linear features          quadratic features
    Gaussian          BayesianLinearRegression("linear")   ("quadratic")
    Gamma / log link  GammaGLMSurrogate(terms=("linear",))  GammaGLMSurrogate()

All of them give genuine posterior draws, so every arm can drive the same
acquisition (e.g. Thompson sampling) and only the model differs. Features are
built through `engine.glm.build_design_matrix`, so every arm shares the declared
coded encoding and the term-order contract.

Both classes keep every observation they have been conditioned on and refit
from the prior on all of it, which is what makes `condition` associative.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, cast

import numpy as np
import pandas as pd
import xarray as xr

from malt.active_learning.actors import SurrogateModel
from malt.engine.factors import Factor
from malt.engine.glm import ALL_TERMS, GammaGLMFit, TermKind, build_design_matrix, fit_gamma_glm

__all__ = ["BayesianLinearRegression", "GammaGLMSurrogate"]

_FEATURE_TERMS: dict[str, tuple[TermKind, ...]] = {
    "linear": ("linear",),
    "quadratic": ALL_TERMS,
}


def _accumulate(seen: pd.DataFrame | None, data: pd.DataFrame) -> pd.DataFrame:
    return data.copy() if seen is None else pd.concat([seen, data], ignore_index=True)


def _unconditioned(name: str) -> ValueError:
    # `sample` takes no rng, so prior draws cannot be made reproducibly; the loop
    # always conditions on the seed before anything samples.
    return ValueError(f"{name} has not been conditioned on any data; condition on the seed first")


@dataclass(frozen=True, slots=True, eq=False)
class BayesianLinearRegression(SurrogateModel):
    """Bayesian linear regression with a Gaussian likelihood.

    `features="linear"` fits an intercept plus one coded term per factor;
    `"quadratic"` fits the full second-order surface (squared and interaction
    terms too) — the classical RSM model, with a posterior instead of an OLS
    point estimate.

    The prior is the standard non-informative one: flat on the coefficients,
    `1/sigma^2` on the noise variance. The posterior is exact and closed-form,
    `sigma^2 ~ InvGamma((n - p) / 2, SSR / 2)` and
    `beta | sigma^2 ~ Normal(beta_OLS, sigma^2 (X'X)^-1)`, so it is centred on
    least squares and its credible intervals match OLS confidence intervals —
    the Bayesian counterpart of classical RSM, which is what a baseline should
    be. Proper priors were tried and rejected: a `Normal(0, sigma^2 c^2)` prior
    shrinks the coefficients of a low-noise surface hard toward zero, and a
    g-prior with `g = n` inflates the noise variance by about `var(y) / n`.

    Like the GLM, it refuses data that cannot identify every coefficient with
    at least one residual degree of freedom.

    """

    factors: tuple[Factor, ...]
    features: Literal["linear", "quadratic"] = "linear"
    response: str = "y"
    n_draws: int = 2000

    data: pd.DataFrame | None = None
    # Closed-form posterior, kept for inspection and testing.
    coef_mean: np.ndarray | None = None  # beta_OLS, intercept first
    coef_cov_unscaled: np.ndarray | None = None  # (X'X)^-1; Cov(beta | sigma^2) = sigma^2 of this
    noise_shape: float | None = None
    noise_scale: float | None = None
    coef_draws: np.ndarray | None = None  # (n_draws, 1 + n_terms)

    def _design(self, x: pd.DataFrame) -> np.ndarray:
        X = build_design_matrix(x, self.factors, terms=_FEATURE_TERMS[self.features]).X
        return np.column_stack([np.ones(len(x)), X])

    def condition(
        self, data: pd.DataFrame, rng: np.random.Generator
    ) -> BayesianLinearRegression:
        seen = _accumulate(self.data, data)
        X = self._design(seen)
        y = seen[self.response].to_numpy(dtype=float)
        n, p = X.shape

        gram = X.T @ X
        if n <= p or np.linalg.matrix_rank(gram) < p:
            raise ValueError(
                f"{n} rows cannot identify {p} coefficients ({self.features} features, "
                f"intercept included); need at least {p + 1} rows spanning every term"
            )
        cov = np.linalg.inv(gram)
        mean = cov @ (X.T @ y)
        residual = y - X @ mean
        shape = (n - p) / 2
        scale = float(residual @ residual) / 2

        sigma2 = scale / rng.gamma(shape, 1.0, size=self.n_draws)
        eps = rng.standard_normal((self.n_draws, p))
        draws = mean + np.sqrt(sigma2)[:, None] * (eps @ np.linalg.cholesky(cov).T)

        return replace(
            self,
            data=seen,
            coef_mean=mean,
            coef_cov_unscaled=cov,
            noise_shape=shape,
            noise_scale=scale,
            coef_draws=draws,
        )

    def sample(self, x: pd.DataFrame) -> np.ndarray:
        if self.coef_draws is None:
            raise _unconditioned(type(self).__name__)
        return self.coef_draws @ self._design(x).T

    @property
    def reliable(self) -> bool:
        # Closed-form posterior: nothing to fail once there is data.
        return self.coef_draws is not None


@dataclass(frozen=True, slots=True, eq=False)
class GammaGLMSurrogate(SurrogateModel):
    """The Gamma/log-link GLM from `engine.glm`, as a loop surrogate.

    `terms` picks the linear predictor: the default is the full quadratic
    surface (the main model); `("linear",)` is the first-order baseline. Each
    `condition` refits by MCMC on all data seen, seeded from the `rng` it is
    handed, and `reliable` is the fit's convergence gate.
    """

    factors: tuple[Factor, ...]
    terms: tuple[TermKind, ...] = ALL_TERMS
    response: str = "y"
    alpha_prior_sigma: float = 10.0
    draws: int = 1000
    tune: int = 1000
    chains: int = 4
    target_accept: float = 0.95

    data: pd.DataFrame | None = None
    fit: GammaGLMFit | None = None

    def condition(self, data: pd.DataFrame, rng: np.random.Generator) -> GammaGLMSurrogate:
        seen = _accumulate(self.data, data)
        fit = fit_gamma_glm(
            seen,
            self.response,
            self.factors,
            terms=self.terms,
            alpha_prior_sigma=self.alpha_prior_sigma,
            draws=self.draws,
            tune=self.tune,
            chains=self.chains,
            target_accept=self.target_accept,
            random_seed=int(rng.integers(2**31)),
        )
        return replace(self, data=seen, fit=fit)

    def sample(self, x: pd.DataFrame) -> np.ndarray:
        if self.fit is None:
            raise _unconditioned(type(self).__name__)
        posterior = cast(xr.DataTree, self.fit.posterior["posterior"]).to_dataset()
        flat = posterior.stack(sample=("chain", "draw"))
        intercept = flat["intercept"].to_numpy()
        beta = flat["beta"].transpose("sample", "term").to_numpy()
        X = build_design_matrix(x, self.fit.factors, terms=self.terms).X
        return np.exp(intercept[:, None] + beta @ X.T)

    @property
    def reliable(self) -> bool:
        return self.fit is not None and self.fit.convergence.converged
