"""Batch acquisition rules: `Acquisition` implementations over `engine.acquisition`.

- `QNoisyExpectedImprovement` — the main rule: greedy Monte Carlo q-EI against
  the per-surface best at the points already run.
- `QUpperConfidenceBound` — greedy Monte Carlo q-UCB, with an explicit
  exploration knob `beta`.
- `ThompsonSampling` — the argmax of one posterior surface per batch slot.
- `CentralComposite` — iterative RSM: a central composite design centred on
  the posterior mean's argmax. With `BayesianLinearRegression("quadratic")` it
  is classical RSM with a posterior.
- `RandomBatch` — uniform picks, ignoring the model: the floor any rule must beat.
- `FixedDesign` — a design chosen up front, run whole in one round, ignoring
  the model: the one-shot DOE most labs run.

The candidate-based rules choose rows of a fixed `candidates` frame (e.g.
`factors.candidate_grid`) and return them with their index labels; the
campaign joins observations positionally, so non-contiguous labels are fine.
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
from malt.engine.factors import Factor

__all__ = [
    "CentralComposite",
    "FixedDesign",
    "QNoisyExpectedImprovement",
    "QUpperConfidenceBound",
    "RandomBatch",
    "ThompsonSampling",
]


@dataclass(frozen=True, slots=True, eq=False)
class QNoisyExpectedImprovement(Acquisition):
    """Greedy batch expected improvement over `candidates`, maximizing mu.

    The bar each surface must beat is its own highest mu at the points the
    model has been conditioned on — not best observed y, which noise inflates.
    See `engine.acquisition` for a worked example.
    """

    candidates: pd.DataFrame

    def propose(
        self, model: SurrogateModel, n: int, rng: np.random.Generator
    ) -> pd.DataFrame:
        if model.data is None:
            raise ValueError("q-NEI needs observed points to improve on; condition on the seed first")
        # Two `sample` calls are one set of surfaces: row s is the same surface in both.
        incumbent = model.sample(model.data.loc[:, self.candidates.columns]).max(axis=1)
        chosen = greedy_q_nei(model.sample(self.candidates), incumbent, n)
        return self.candidates.iloc[chosen]


@dataclass(frozen=True, slots=True, eq=False)
class QUpperConfidenceBound(Acquisition):
    """Greedy batch upper confidence bound over `candidates`, maximizing mu.

    `beta` is the exploration weight: for one Gaussian point the score is
    `mean + sqrt(beta) * sd`, and `beta = 0` proposes the top `n` by posterior
    mean. It has no default, so every benchmark arm states its own.
    """

    candidates: pd.DataFrame
    beta: float

    def propose(
        self, model: SurrogateModel, n: int, rng: np.random.Generator
    ) -> pd.DataFrame:
        chosen = greedy_q_ucb(model.sample(self.candidates), n, self.beta)
        return self.candidates.iloc[chosen]


@dataclass(frozen=True, slots=True, eq=False)
class ThompsonSampling(Acquisition):
    """Each of the `n` points is the argmax over `candidates` of a random posterior surface."""

    candidates: pd.DataFrame

    def propose(
        self, model: SurrogateModel, n: int, rng: np.random.Generator
    ) -> pd.DataFrame:
        chosen = thompson_batch(model.sample(self.candidates), n, rng)
        return self.candidates.iloc[chosen]


@dataclass(frozen=True, slots=True, eq=False)
class CentralComposite(Acquisition):
    """A central composite design at the posterior mean's argmax over `candidates`.

    `radius` is the half-width of the factorial cube and `alpha` the axial
    distance relative to it, both in coded units; the default is a
    face-centred design over half of each factor's range. A batch of `n` is
    the `2^k + 2k` design runs plus `n - (2^k + 2k)` centre replicates. Near
    the boundary the design is moved inward whole, so every run stays inside
    the declared ranges.
    """

    factors: tuple[Factor, ...]
    candidates: pd.DataFrame
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
        best = self.candidates.iloc[int(np.argmax(model.sample(self.candidates).mean(axis=0)))]
        center = np.array([float(f.encode(best[f.name])) for f in self.factors])
        coded = central_composite(center, self.radius, self.alpha, n_center=n - n_design)
        x = pd.DataFrame({f.name: f.decode(coded[:, j]) for j, f in enumerate(self.factors)})
        return x


@dataclass(frozen=True, slots=True, eq=False)
class RandomBatch(Acquisition):
    """`n` distinct rows of `candidates`, uniformly at random. Ignores the model."""

    candidates: pd.DataFrame

    def propose(
        self, model: SurrogateModel, n: int, rng: np.random.Generator
    ) -> pd.DataFrame:
        return self.candidates.iloc[rng.choice(len(self.candidates), n, replace=False)]


@dataclass(frozen=True, slots=True, eq=False)
class FixedDesign(Acquisition):
    """A design chosen up front, proposed whole as one batch. Ignores the model.

    `design` is in real units, one column per factor. The batch size must be
    `len(design)`, and a one-shot DOE is one round: run it with
    `MaxRoundsRule(1)`. A face-centred CCD over the full ranges comes from
    `engine.acquisition.central_composite`::

        coded = central_composite(np.zeros(len(factors)), radius=1.0, n_center=3)
        design = pd.DataFrame({f.name: f.decode(coded[:, j]) for j, f in enumerate(factors)})

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
