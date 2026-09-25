"""Dummy actors for testing the loop itself, not any real model.

Each one is the smallest subclass of its actor ABC that makes the
property under test observable: a model that counts what it has seen, a design
that walks through fixed blocks, an environment whose response is a known
function of x.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from malt.active_learning.actors import Acquisition, Environment, SurrogateModel

GRID = pd.DataFrame({"glucose": np.linspace(0.0, 10.0, 500)})


@dataclass(frozen=True)
class CountingModel(SurrogateModel):
    """A stand-in surrogate that models nothing. Each field exposes one loop
    property to the tests:

    - `n_seen` counts the points conditioned on, so the journal's snapshots can
      be checked (0, 3, 6, ...) — an in-place `condition` would make them all equal.
    - `draw` is one number taken from the experimenter's random stream per
      `condition`, so reproducibility covers the model's randomness too.
    - `fail_at` makes `reliable` turn False once that many points are seen.
    """

    n_seen: int = 0
    draw: float = 0.0
    fail_at: float = np.inf

    def condition(self, data: pd.DataFrame, rng: np.random.Generator) -> CountingModel:
        return CountingModel(
            n_seen=self.n_seen + len(data),
            draw=float(rng.normal()),
            fail_at=self.fail_at,
        )

    def sample(self, x: pd.DataFrame) -> np.ndarray:
        return np.zeros((1, len(x)))

    @property
    def reliable(self) -> bool:
        return self.n_seen < self.fail_at


@dataclass(frozen=True)
class BlockDesign(Acquisition):
    """A fixed design: block `i` is grid rows `[i*n, (i+1)*n)`. Uses no randomness."""

    block: int = 0

    def propose(self, model, n, rng):
        x = GRID.iloc[self.block * n : (self.block + 1) * n]
        return x, BlockDesign(self.block + 1)


@dataclass(frozen=True)
class RandomPick(Acquisition):
    """Picks random grid rows — so `x` carries non-contiguous grid index labels —
    and burns `burn` extra draws to consume a different amount of randomness."""

    burn: int = 0

    def propose(self, model, n, rng):
        rng.random(self.burn)
        return GRID.iloc[rng.choice(len(GRID), n, replace=False)], self


class LinearEnvironment(Environment):
    """Noise-free `y = 10 * glucose`, so every y reveals which x it belongs to."""

    def query(self, x, rng):
        return pd.DataFrame({"y": 10.0 * x["glucose"].to_numpy()})


class NoiseEnvironment(Environment):
    """Pure noise, independent of x — isolates the environment's random stream."""

    def query(self, x, rng):
        return pd.DataFrame({"y": rng.normal(size=len(x))})


@pytest.fixture
def seed_data() -> pd.DataFrame:
    return GRID.iloc[[0, 250, 499]].assign(y=[0.0, 50.0, 100.0])
