"""Bayesian Gamma/log-link response-surface GLM.

Pure functions: data in, results out. No I/O, no state, no orchestration.

The model is a quadratic response surface (linear + squared + pairwise
interaction terms) over an arbitrary number of experimental variables, with a
Gamma likelihood and a log link — the right family for strictly-positive
biomass with variance growing as the square of the mean.

Predictors are encoded to coded `[-1, +1]` units *before* the squared and
interaction terms are formed. This is not cosmetic: fitting the quadratic terms
on raw concentrations makes `glucose` and `glucose^2` nearly collinear over a
bounded positive range, which wrecks NUTS's geometry (thousands of divergences,
r_hat ~3).

The encoding comes from each `Factor`'s declared range, not from the observed
data. An empirical z-score would drift as a campaign concentrates near an
optimum, which would silently change what the quadratic prior means from one
round to the next, and would throw away the balance of a log-scaled factor's
design levels. See `build_design_matrix`.

Running this module needs ``PYTENSOR_FLAGS='cxx='`` on macOS 26+ — see README.
"""

from __future__ import annotations

import itertools
import warnings
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Literal, cast

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr

from malt.engine.factors import Factor

__all__ = [
    "ALL_TERMS",
    "ConvergenceReport",
    "ConvergenceWarning",
    "DesignMatrix",
    "GammaGLM",
    "GammaGLMFit",
    "TermKind",
    "build_design_matrix",
    "build_gamma_glm",
    "check_convergence",
    "coefficient_draws",
    "fit_gamma_glm",
    "predict_mu",
    "sample_gamma_glm",
]

TermKind = Literal["linear", "quadratic", "interaction"]
ALL_TERMS: tuple[TermKind, ...] = ("linear", "quadratic", "interaction")

# Priors on the coded scale. Quadratic terms are centred negative on the
# working assumption that growth surfaces are peaked — a soft prior the data can
# and should override, not a concavity constraint.
_PRIOR_MU: dict[TermKind, float] = {"linear": 0.0, "quadratic": -0.5, "interaction": 0.0}
_PRIOR_SIGMA: dict[TermKind, float] = {"linear": 1.0, "quadratic": 0.5, "interaction": 0.5}


class ConvergenceWarning(UserWarning):
    """A fit completed but failed its convergence gate."""


@dataclass(frozen=True, slots=True)
class DesignMatrix:
    """A coded quadratic response-surface basis. No intercept column."""

    X: np.ndarray
    term_names: tuple[str, ...]
    term_kinds: tuple[TermKind, ...]
    factors: tuple[Factor, ...]


@dataclass(frozen=True, slots=True)
class ConvergenceReport:
    """Whether a posterior is trustworthy, and why not if it isn't."""

    converged: bool
    divergences: int
    max_rhat: float
    min_ess_bulk: float
    min_ess_tail: float
    max_treedepth_hits: int
    n_chains: int
    n_draws: int
    failures: tuple[str, ...]

    def summary(self) -> str:
        """A one-line summary of the convergence check."""
        if self.converged:
            return "converged"
        return f"failed: {', '.join(self.failures)}"


@dataclass(frozen=True, slots=True)
class GammaGLM:
    """An unfitted model: the PyMC model plus what is needed to read it.

    Held together in one value so the design matrix that shaped the model and
    the model itself cannot drift apart between definition and sampling.
    Unlike `GammaGLMFit` this is a transient handle, not a serializable
    record — a `pm.Model` does not belong in `state/`.
    """

    model: pm.Model
    design: DesignMatrix
    response: str


@dataclass(frozen=True, slots=True)
class GammaGLMFit:
    """A fitted model: posterior draws plus everything needed to use them.

    `posterior` is the full sample tree — `/posterior` holds draws over
    `intercept`, `beta` (with term names as a coordinate) and `alpha`;
    `/constant_data` holds the design matrix that was fitted.
    """

    posterior: xr.DataTree
    response: str
    factors: tuple[Factor, ...]
    term_names: tuple[str, ...]
    term_kinds: tuple[TermKind, ...]
    convergence: ConvergenceReport

    @property
    def predictors(self) -> tuple[str, ...]:
        """Factor names, in the order their terms appear in the design."""
        return tuple(f.name for f in self.factors)


def build_design_matrix(
    data: pd.DataFrame,
    factors: Sequence[Factor],
    *,
    terms: Collection[TermKind] = ALL_TERMS,
) -> DesignMatrix:
    """Build the coded quadratic basis for `factors`.

    Each column is encoded to `[-1, +1]` against its factor's declared range
    first, then the squared and interaction terms are formed from the coded
    values. Term order is a contract downstream code relies on to rebuild query
    rows: all linear terms in the caller's factor order, then all squared terms
    in the same order, then all pairwise interactions in `itertools.combinations`
    order.

    `terms` keeps only the listed kinds — e.g. `("linear",)` for a first-order
    baseline — preserving that order among the ones kept. The default is the
    full quadratic surface; anything narrower is a comparison model, not a
    replacement for it.
    """
    unknown = set(terms) - set(ALL_TERMS)
    if not terms or unknown:
        raise ValueError(f"terms must be a non-empty subset of {ALL_TERMS}, got {tuple(terms)}")
    factors = tuple(factors)
    if not factors:
        raise ValueError("at least one factor is required")
    names = [f.name for f in factors]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(f"factor names must be unique, repeated: {duplicates}")
    missing = [n for n in names if n not in data.columns]
    if missing:
        raise ValueError(f"factors not found in data: {missing}")

    raw = data.loc[:, names].to_numpy(dtype=float)
    if not np.isfinite(raw).all():
        bad = [n for n, ok in zip(names, np.isfinite(raw).all(axis=0)) if not ok]
        raise ValueError(f"factor columns contain NaN or infinite values: {bad}")

    # Encoding is declared, not empirical, so there is no sample standard
    # deviation here to be degenerate — a constant column is simply a constant
    # coded value, which the row-count check below catches as unidentifiable.
    z = np.column_stack([f.encode(raw[:, j]) for j, f in enumerate(factors)])

    p = len(factors)
    columns = [z[:, j] for j in range(p)]
    term_names = list(names)
    kinds: list[TermKind] = ["linear"] * p

    for j, name in enumerate(names):
        columns.append(z[:, j] ** 2)
        term_names.append(f"{name}^2")
        kinds.append("quadratic")

    for a, b in itertools.combinations(range(p), 2):
        columns.append(z[:, a] * z[:, b])
        term_names.append(f"{names[a]}:{names[b]}")
        kinds.append("interaction")

    keep = [i for i, kind in enumerate(kinds) if kind in terms]
    return DesignMatrix(
        X=np.column_stack([columns[i] for i in keep]),
        term_names=tuple(term_names[i] for i in keep),
        term_kinds=tuple(kinds[i] for i in keep),
        factors=factors,
    )


def check_convergence(
    idata: xr.DataTree,
    *,
    max_rhat: float = 1.01,
    min_ess: float = 1000.0,
    max_divergences: int = 0,
) -> ConvergenceReport:
    """Summarize whether a sample is trustworthy.

    Public so `uncertainty.py` and `diagnostics.py` can tell a good fit from a
    broken one without re-deriving the check.

    Tail ESS is gated alongside bulk ESS because acquisition functions are Monte
    Carlo over draws and read the tails of mu's posterior — bulk ESS can look
    healthy while the 95th percentile still jitters. Max-treedepth hits are
    reported but not gated: they signal biased exploration, not a broken chain.
    """
    stats = idata["sample_stats"]
    divergences = int(stats["diverging"].sum())

    # nutpie names these `depth` / `maxdepth_reached`; PyMC's own NUTS uses
    # `tree_depth` / `reached_max_treedepth`.
    treedepth_key = next(
        (k for k in ("maxdepth_reached", "reached_max_treedepth") if k in stats.data_vars),
        None,
    )
    treedepth_hits = int(stats[treedepth_key].sum()) if treedepth_key else 0

    # az.rhat/az.ess return a DataTree holding only `/posterior`; `to_dataset`
    # flattens that single node. Reduce per-variable rather than via `to_array`,
    # which would broadcast every variable against the union of all dims.
    rhat_ds = az.rhat(idata).to_dataset()
    bulk_ds = az.ess(idata, method="bulk").to_dataset()
    tail_ds = az.ess(idata, method="tail").to_dataset()
    observed_max_rhat = max(float(rhat_ds[v].max()) for v in rhat_ds.data_vars)
    observed_min_bulk = min(float(bulk_ds[v].min()) for v in bulk_ds.data_vars)
    observed_min_tail = min(float(tail_ds[v].min()) for v in tail_ds.data_vars)

    failures: list[str] = []
    if divergences > max_divergences:
        failures.append(f"{divergences} divergences > {max_divergences}")
    if not observed_max_rhat < max_rhat:
        failures.append(f"max r_hat {observed_max_rhat:.4f} >= {max_rhat}")
    if observed_min_bulk < min_ess:
        failures.append(f"min bulk ESS {observed_min_bulk:.0f} < {min_ess:.0f}")
    if observed_min_tail < min_ess:
        failures.append(f"min tail ESS {observed_min_tail:.0f} < {min_ess:.0f}")

    return ConvergenceReport(
        converged=not failures,
        divergences=divergences,
        max_rhat=observed_max_rhat,
        min_ess_bulk=observed_min_bulk,
        min_ess_tail=observed_min_tail,
        max_treedepth_hits=treedepth_hits,
        n_chains=int(stats.sizes["chain"]),
        n_draws=int(stats.sizes["draw"]),
        failures=tuple(failures),
    )


def build_gamma_glm(
    data: pd.DataFrame,
    response: str,
    factors: Sequence[Factor],
    *,
    alpha_prior_sigma: float = 10.0,
    terms: Collection[TermKind] = ALL_TERMS,
) -> GammaGLM:
    """Define the model without sampling it.

    Separate from `sample_gamma_glm` so the specification can be inspected,
    prior-predictive checked, or modified before anyone pays for a posterior.

    `alpha` (the Gamma shape, equal to 1/CV^2) is weakly identified at the
    dataset sizes this domain produces, so its prior scale materially moves the
    posterior. It is exposed for that reason; the default assumes a fairly noisy
    assay, which errs toward more exploration rather than less.
    """
    design = build_design_matrix(data, factors, terms=terms)

    if response not in data.columns:
        raise ValueError(f"response {response!r} not found in data")
    y = data.loc[:, response].to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValueError(f"response {response!r} contains NaN or infinite values")
    if not np.all(y > 0):
        raise ValueError(
            f"response {response!r} must be strictly positive for a Gamma likelihood; "
            f"found {int((y <= 0).sum())} non-positive value(s)"
        )

    n_obs, n_terms = design.X.shape
    if n_obs < n_terms + 1:
        raise ValueError(
            f"{n_obs} rows cannot identify {n_terms} terms plus an intercept; "
            f"need at least {n_terms + 1}"
        )

    prior_mu = np.array([_PRIOR_MU[k] for k in design.term_kinds])
    prior_sigma = np.array([_PRIOR_SIGMA[k] for k in design.term_kinds])

    model = pm.Model(coords={"term": list(design.term_names), "obs": np.arange(n_obs)})
    with model:
        # Held as constant data so the returned tree is a self-contained record
        # of what was fitted; predictions happen in numpy over the draws.
        X = pm.Data("X", design.X, dims=("obs", "term"))

        # The log-link intercept sits on whatever scale the response uses, so
        # centre it on the observed mean rather than an arbitrary zero.
        intercept = pm.Normal("intercept", mu=np.log(y.mean()), sigma=2.0)
        beta = pm.Normal("beta", mu=prior_mu, sigma=prior_sigma, dims="term")
        alpha = pm.HalfNormal("alpha", sigma=alpha_prior_sigma)

        mu = pm.math.exp(intercept + pm.math.dot(X, beta))
        # One shape, varying rate: Var(y) = mu^2 / alpha, i.e. constant CV.
        pm.Gamma("y_obs", alpha=alpha, beta=alpha / mu, observed=y, dims="obs")

    return GammaGLM(model=model, design=design, response=response)


def sample_gamma_glm(
    glm: GammaGLM,
    *,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.95,
    random_seed: int | None = None,
) -> GammaGLMFit:
    """Sample a model built by `build_gamma_glm`.

    Returns posterior draws alongside a `ConvergenceReport`. A fit that fails
    the gate is still returned — the diverged draws are what you need to
    diagnose it — but raises a `ConvergenceWarning`.

    `target_accept` is the first dial to turn if divergences appear; raise it
    toward 0.99 before reparameterizing.
    """
    idata = pm.sample(
        draws=draws,
        tune=tune,
        chains=chains,
        # Not a default to override: PyMC's automatic core allocation
        # divides by `cores` and raises ZeroDivisionError on a single-core
        # runtime.
        cores=1,
        target_accept=target_accept,
        nuts_sampler="nutpie",
        random_seed=random_seed,
        progressbar=False,
        model=glm.model,
    )

    convergence = check_convergence(idata)
    if not convergence.converged:
        warnings.warn(
            "Gamma GLM fit failed its convergence gate: "
            + "; ".join(convergence.failures)
            + ". Treat these draws as unreliable.",
            ConvergenceWarning,
            stacklevel=2,
        )

    return GammaGLMFit(
        posterior=idata,
        response=glm.response,
        factors=glm.design.factors,
        term_names=glm.design.term_names,
        term_kinds=glm.design.term_kinds,
        convergence=convergence,
    )


def fit_gamma_glm(
    data: pd.DataFrame,
    response: str,
    factors: Sequence[Factor],
    *,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.95,
    alpha_prior_sigma: float = 10.0,
    terms: Collection[TermKind] = ALL_TERMS,
    random_seed: int | None = None,
) -> GammaGLMFit:
    """Build and sample in one call, for the common case.

    Reach for `build_gamma_glm` and `sample_gamma_glm` separately when you want
    to inspect the model, run a prior predictive check, or sample the same
    specification more than once.
    """
    glm = build_gamma_glm(
        data, response, factors, alpha_prior_sigma=alpha_prior_sigma, terms=terms
    )
    return sample_gamma_glm(
        glm,
        draws=draws,
        tune=tune,
        chains=chains,
        target_accept=target_accept,
        random_seed=random_seed,
    )


def coefficient_draws(fit: GammaGLMFit) -> tuple[np.ndarray, np.ndarray]:
    """The posterior's coefficient draws as flat arrays: intercept `(S,)` and beta `(S, n_terms)`.

    Chains are concatenated in order, so draw `s` is the same across calls.
    Flatten once and reuse: stacking the posterior is the slow part of
    predicting at many points.
    """
    posterior = cast(xr.DataTree, fit.posterior["posterior"]).to_dataset()
    flat = posterior.stack(sample=("chain", "draw"))
    return flat["intercept"].to_numpy(), flat["beta"].transpose("sample", "term").to_numpy()


def predict_mu(intercept: np.ndarray, beta: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Draws of mu at the rows of design matrix `X`: `exp(intercept + X beta)`, shape `(S, len(X))`.

    `X` must come from `build_design_matrix` with the fit's own factors and
    terms, or the columns will not line up with `beta`.
    """
    return np.exp(intercept[:, None] + beta @ X.T)
