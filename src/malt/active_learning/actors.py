"""The actors in the active-learning campaign.

    Experimenter = SurrogateModel + Acquisition   --propose x-->   Environment
                 <-------------------- observe (x, y) ----------------

- `SurrogateModel` — what the campaign believes about the response surface.
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

    def thinned(self, n_draws: int) -> Self:
        """This model carrying about `n_draws` of its draws, for optimizing an acquisition.

        An optimizer evaluates the acquisition hundreds of times at single
        points, and every draw is multiplied through on each one. A few hundred
        draws are enough to locate an acquisition's optimum — BoTorch defaults
        to 128-512 base samples for the same reason — while the full posterior
        is what the reported answer should be computed from.

        Returns `self` by default, which is correct and merely slower: using
        every draw is the more accurate answer, so a model that cannot thin
        cheaply loses nothing but time. Thinning must be a **stride over frozen
        draws, never a resample**, so that row `s` is the same surface before
        and after and repeated calls agree.
        """
        return self

    def sample_jacobian(self, x: pd.DataFrame) -> np.ndarray:
        """Derivative of `sample` with respect to each factor, in real units: `(n_draws, len(x), k)`.

        `x` has exactly one column per factor, and the last axis follows its
        column order. Row `s` is the derivative of row `s` of `sample`, on the
        same surface — so a gradient-based optimizer walks one fixed surface per
        draw, not a resampled one.

        This default is central finite differences over `sample`, with a step
        relative to each value's magnitude. It is here so that every surrogate
        works with a gradient-based acquisition, including a baseline or a test
        dummy; it costs `2k` `sample` calls per evaluation, so a model with a
        closed-form derivative should override it. Correctness, not speed, is
        the reason it exists.
        """
        values = x.to_numpy(dtype=float)
        # Cube root of machine epsilon: the step that balances truncation
        # against round-off for a central difference.
        step = np.cbrt(np.finfo(float).eps) * np.maximum(np.abs(values), 1.0)
        columns = []
        for j in range(values.shape[1]):
            shift = np.zeros_like(values)
            shift[:, j] = step[:, j]
            plus = self.sample(pd.DataFrame(values + shift, columns=x.columns))
            minus = self.sample(pd.DataFrame(values - shift, columns=x.columns))
            columns.append((plus - minus) / (2.0 * step[:, j]))
        return np.stack(columns, axis=-1)

    @property
    @abstractmethod
    def data(self) -> pd.DataFrame | None:
        """Every observation conditioned on so far, or None before any."""
        ...

    @property
    @abstractmethod
    def reliable(self) -> bool:
        """Whether `sample` can be trusted — e.g. whether MCMC passed its convergence gate."""
        ...


class Acquisition(ABC):
    """A rule for choosing the next batch of experiments.

    Model-driven rules (Thompson, UCB, EI) read the model; a fixed design
    ignores it. A rule is stateless and fixed for the whole campaign: all it
    learns between rounds, it learns through the model. Points must lie inside
    the declared `Factor` ranges.
    """

    __slots__ = ()

    @abstractmethod
    def propose(
        self, model: SurrogateModel, n: int, rng: np.random.Generator
    ) -> pd.DataFrame:
        """Choose `n` design points.

        Returns a frame with a column per factor, in real units. Any randomness
        in the choice — e.g. Thompson sampling — comes from `rng`.
        """
        ...


class Environment(ABC):
    """A simulated lab: somewhere a batch can be run and read back in one call.

    **Do not implement this for real hardware or a real LIMS.** `query` is
    synchronous, so a real implementation would let `run_campaign` submit work
    and take results with nobody in between, and a proposed batch reaching a
    bench without an explicit human approval is the one thing this design rules
    out. `simulation.oracle.Oracle` is what this exists for.

    A real campaign uses the two `Experimenter` methods directly, days apart:
    `propose` returns a batch, a person approves it, the lab runs it, and
    `observe` takes the results back whenever they arrive. Nothing in this class
    sits in that loop.
    """

    __slots__ = ()

    @abstractmethod
    def query(self, x: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        """Observe the response at each row of `x`, one output row per input row.

        `x` has a column per factor, in real units. The result has a `y`
        column — one noisy draw of y, not the mean mu: querying the same point
        twice gives two different observations — plus any covariates the
        environment knows about how the runs were done, e.g. a batch label,
        operator or inoculum. Rows correspond to `x` by position. Noise is drawn
        from `rng`, which is what makes a campaign reproducible from its seed.
        """
        ...


@dataclass
class Experimenter:
    """A surrogate model paired with an acquisition rule — the campaign's decision-maker.

    These two methods are the whole interface to a real campaign. `propose`
    hands back a batch and stops; approving and running it happen outside, on
    whatever timescale the lab works at; `observe` takes the results back. The
    pair is what `run_campaign` drives in a simulation, and what the state store
    and the agent-facing tools will drive for real.
    """

    surrogate_model: SurrogateModel
    acquisition: Acquisition

    def propose(self, n: int, rng: np.random.Generator) -> pd.DataFrame:
        """Choose `n` design points."""
        return self.acquisition.propose(self.surrogate_model, n, rng)

    def observe(self, data: pd.DataFrame, rng: np.random.Generator) -> None:
        """Condition this experimenter's surrogate model on `data`, in place."""
        self.surrogate_model = self.surrogate_model.condition(data, rng)
