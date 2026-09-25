"""The actors in the active-learning loop.

    Experimenter = SurrogateModel + Acquisition   --propose x-->   Environment
                 <-------------------- observe (x, y) ----------------

- `SurrogateModel` — what the loop believes about the response surface.
- `Acquisition` — how it chooses the next experiments given that belief.
- `Experimenter` — the two together: the thing that interacts with the world, and
  the unit a benchmark compares (BO vs. iterative DOE is two experimenters).
- `Environment` — the world it acts on, synthetic or real.

"""

from __future__ import annotations

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Self

import numpy as np
import pandas as pd

__all__ = [
    "Acquisition",
    "Environment",
    "Experimenter",
    "SurrogateModel",
]


class SurrogateModel(ABC):
    """A distribution over response surfaces that can be conditioned and sampled.

    Before it has seen any data it is the prior; `condition` turns it into the
    posterior given some observations.

    Conditioning is associative, so `m.condition(d1).condition(d2)` must equal
    `m.condition(pd.concat([d1, d2]))`. Models without a closed-form update
    (e.g. a Gamma GLM sampled by MCMC) meet this by keeping their prior
    configuration and every observation seen so far, and refitting on all of it.
    """

    __slots__ = ()

    @abstractmethod
    def condition(self, data: pd.DataFrame, rng: np.random.Generator) -> Self:
        """Return this model additionally conditioned on `data`.

        `data` holds one row per observation: a column per factor in real
        units, plus the response. `self` is left unchanged. Any randomness in
        conditioning — e.g. seeding an MCMC sampler — comes from `rng`.
        """
        ...

    @abstractmethod
    def sample(self, x: pd.DataFrame) -> np.ndarray:
        """Joint draws of the mean response mu at `x`, shape `(n_draws, len(x))`.

        `x` has a column per factor, in real units. Draws are of mu, not of a
        new observation y. They are prior draws if nothing has been observed.

        Row `s` is the same surface for every point and every call: repeated
        calls reuse the same underlying draws rather than resampling. Thompson
        sampling takes the argmax of a row across candidates, which is only
        meaningful if the whole row comes from one surface.
        """
        ...

    @property
    @abstractmethod
    def reliable(self) -> bool:
        """Whether `sample` can be trusted — e.g. whether MCMC passed its convergence gate."""
        ...


class Acquisition(ABC):
    """A rule for choosing the next batch of experiments.

    Model-driven rules (Thompson, UCB, EI) read the model; a fixed design
    ignores it. Rules that carry state across rounds — a fixed design's next
    block, an iterative RSM's current region — return their successor from
    `propose` rather than mutating themselves. Points must lie inside the
    declared `Factor` ranges; a rule working in a local sub-region keeps that
    region to itself.
    """

    __slots__ = ()

    @abstractmethod
    def propose(
        self, model: SurrogateModel, n: int, rng: np.random.Generator
    ) -> tuple[pd.DataFrame, Self]:
        """Choose `n` design points, plus the rule to use next round.

        Returns a frame with a column per factor, in real units. Any randomness
        in the choice — e.g. Thompson sampling — comes from `rng`.
        """
        ...


class Environment(ABC):
    """Something that can be queried at design points for noisy observations, real or synthetic."""

    __slots__ = ()

    @abstractmethod
    def query(self, x: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        """Observe the response at each row of `x`, one output row per input row.

        `x` has a column per factor, in real units. The result has a `y`
        column — one noisy draw of y, not the mean mu: querying the same point
        twice gives two different observations — plus any covariates the
        environment knows about how the runs were done, e.g. a batch label,
        operator or inoculum. Rows correspond to `x` by position. A synthetic
        environment draws its noise from `rng`; a real one ignores it.
        """
        ...


@dataclass
class Experimenter:
    """A surrogate model paired with an acquisition rule — the loop's decision-maker."""

    surrogate_model: SurrogateModel
    acquisition: Acquisition

    def propose(self, n: int, rng: np.random.Generator) -> pd.DataFrame:
        """Choose `n` design points, advancing the acquisition rule in place."""
        x, self.acquisition = self.acquisition.propose(self.surrogate_model, n, rng)
        return x

    def observe(self, data: pd.DataFrame, rng: np.random.Generator) -> None:
        """Condition this experimenter's surrogate model on `data`, in place."""
        self.surrogate_model = self.surrogate_model.condition(data, rng)
