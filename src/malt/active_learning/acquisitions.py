"""Batch acquisition rules: `Acquisition` implementations over `engine.acquisition`.

- `QNoisyExpectedImprovement` — the main rule: greedy Monte Carlo q-EI against
  the per-surface best at the points already run.
- `QUpperConfidenceBound` — greedy Monte Carlo q-UCB, with an explicit
  exploration knob `beta`.
- `ThompsonSampling` — the argmax of one posterior surface per batch slot.
- `CentralComposite` — iterative RSM: a central composite design centred on
  the posterior mean's argmax. With `BayesianLinearRegression("quadratic")` it
  is classical RSM with a posterior.
- `RandomBatch` — uniform on every axis, ignoring the model: the floor any
  rule must beat.
- `FixedDesign` — a design chosen up front, run whole in one round, ignoring
  the model: the one-shot DOE most labs run.

No rule searches a fixed grid. The model-driven rules choose from a fresh
set of scrambled Sobol points each round, drawn from the `rng` they are
handed, and argmaxes of the posterior mean are found by continuous search
(`engine.search`). Everything is drawn in coded units and decoded through
the factors, so log-scale factors are covered evenly in log space.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from malt.active_learning.actors import Acquisition, SurrogateModel
from malt.engine.acquisition import (
    central_composite,
    greedy_q_nei,
    greedy_q_ucb,
    thompson_batch,
)
from malt.engine.factors import Factor, decode_design
from malt.engine.search import maximize, space_filling

__all__ = [
    "CentralComposite",
    "FixedDesign",
    "QNoisyExpectedImprovement",
    "QUpperConfidenceBound",
    "RandomBatch",
    "ThompsonSampling",
    "posterior_mean_optimum",
]


def posterior_mean_optimum(model: SurrogateModel, factors: tuple[Factor, ...]) -> pd.DataFrame:
    """The point with the highest posterior mean, found by continuous search: one row, real units.

    Deterministic, so it consumes no randomness: the same model always gives
    the same answer.
    """
    x, _ = maximize(lambda z: model.sample(decode_design(factors, z)).mean(axis=0), len(factors))
    return decode_design(factors, x)


def _candidates(factors: tuple[Factor, ...], n: int, rng: np.random.Generator) -> pd.DataFrame:
    return decode_design(factors, space_filling(len(factors), n, rng))


@dataclass(frozen=True, slots=True, eq=False)
class QNoisyExpectedImprovement(Acquisition):
    """Greedy batch expected improvement, maximizing mu, over fresh candidates each round.

    The bar each surface must beat is its own highest mu at the points the
    model has been conditioned on — not best observed y, which noise inflates.
    See `engine.acquisition` for a worked example. `n_candidates` (a power of
    2) Sobol points are drawn from `rng` every round.
    """

    factors: tuple[Factor, ...]
    n_candidates: int = 4096

    def propose(
        self, model: SurrogateModel, n: int, rng: np.random.Generator
    ) -> pd.DataFrame:
        if model.data is None:
            raise ValueError("q-NEI needs observed points to improve on; condition on the seed first")
        candidates = _candidates(self.factors, self.n_candidates, rng)
        # Two `sample` calls are one set of surfaces: row s is the same surface in both.
        incumbent = model.sample(model.data.loc[:, list(candidates.columns)]).max(axis=1)
        chosen = greedy_q_nei(model.sample(candidates), incumbent, n)
        return candidates.iloc[chosen].reset_index(drop=True)


@dataclass(frozen=True, slots=True, eq=False)
class QUpperConfidenceBound(Acquisition):
    """Greedy batch upper confidence bound, maximizing mu, over fresh candidates each round.

    `beta` is the exploration weight: for one Gaussian point the score is
    `mean + sqrt(beta) * sd`, and `beta = 0` proposes the top `n` by posterior
    mean. It has no default, so every benchmark arm states its own.
    """

    factors: tuple[Factor, ...]
    beta: float
    n_candidates: int = 4096

    def propose(
        self, model: SurrogateModel, n: int, rng: np.random.Generator
    ) -> pd.DataFrame:
        candidates = _candidates(self.factors, self.n_candidates, rng)
        chosen = greedy_q_ucb(model.sample(candidates), n, self.beta)
        return candidates.iloc[chosen].reset_index(drop=True)


@dataclass(frozen=True, slots=True, eq=False)
class ThompsonSampling(Acquisition):
    """Each of the `n` points is the argmax of a random posterior surface over fresh candidates."""

    factors: tuple[Factor, ...]
    n_candidates: int = 4096

    def propose(
        self, model: SurrogateModel, n: int, rng: np.random.Generator
    ) -> pd.DataFrame:
        candidates = _candidates(self.factors, self.n_candidates, rng)
        chosen = thompson_batch(model.sample(candidates), n, rng)
        return candidates.iloc[chosen].reset_index(drop=True)


@dataclass(frozen=True, slots=True, eq=False)
class CentralComposite(Acquisition):
    """A central composite design at the posterior mean's argmax.

    `radius` is the half-width of the factorial cube and `alpha` the axial
    distance relative to it, both in coded units; the default is a
    face-centred design over half of each factor's range. A batch of `n` is
    the `2^k + 2k` design runs plus `n - (2^k + 2k)` centre replicates. Near
    the boundary the design is moved inward whole, so every run stays inside
    the declared ranges.
    """

    factors: tuple[Factor, ...]
    radius: float = 0.5
    alpha: float = 1.0

    def __post_init__(self) -> None:
        central_composite(np.zeros(len(self.factors)), self.radius, self.alpha)  # validates

    def propose(
        self, model: SurrogateModel, n: int, rng: np.random.Generator
    ) -> pd.DataFrame:
        k = len(self.factors)
        n_design = 2**k + 2 * k
        if n < n_design:
            raise ValueError(f"a central composite design in {k} factors needs a batch of at least {n_design}, got {n}")
        best = posterior_mean_optimum(model, self.factors)
        center = np.array([float(f.encode(best[f.name].to_numpy())[0]) for f in self.factors])
        return decode_design(self.factors, central_composite(center, self.radius, self.alpha, n_center=n - n_design))


@dataclass(frozen=True, slots=True, eq=False)
class RandomBatch(Acquisition):
    """`n` points drawn uniformly on every axis in coded units. Ignores the model.

    Log-scale factors are therefore log-uniform: as many runs between 0.1 and
    1 g/L as between 1 and 10.
    """

    factors: tuple[Factor, ...]

    def propose(
        self, model: SurrogateModel, n: int, rng: np.random.Generator
    ) -> pd.DataFrame:
        # surrogate is explicitly ignored here
        return decode_design(self.factors, rng.uniform(-1.0, 1.0, (n, len(self.factors))))


@dataclass(frozen=True, slots=True, eq=False)
class FixedDesign(Acquisition):
    """A design chosen up front, proposed whole as one batch. Ignores the model.

    `design` is in real units, one column per factor. The batch size must be
    `len(design)`, and a one-shot DOE is one round: run it with
    `MaxRoundsRule(1)`. A face-centred CCD over the full ranges comes from
    `engine.acquisition.central_composite`::

        design = decode_design(factors, central_composite(np.zeros(len(factors)), radius=1.0, n_center=3))

    For a Box-Behnken design, decode `pyDOE3.bbdesign(len(factors))` the same way.
    """

    factors: tuple[Factor, ...]
    design: pd.DataFrame

    def __post_init__(self) -> None:
        missing = [f.name for f in self.factors if f.name not in self.design.columns]
        if missing:
            raise ValueError(f"design is missing factors: {missing}")
        outside = [f.name for f in self.factors if not f.contains(self.design[f.name].to_numpy()).all()]
        if outside:
            raise ValueError(f"design leaves the declared range of: {outside}")

    def propose(
        self, model: SurrogateModel, n: int, rng: np.random.Generator
    ) -> pd.DataFrame:
        if n != len(self.design):
            raise ValueError(f"a fixed design is run whole: batch size must be {len(self.design)}, got {n}")
        return self.design.loc[:, [f.name for f in self.factors]]
