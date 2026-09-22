"""Experimental factors — the variables a campaign may vary, and their ranges.

A `Factor` declares one variable: its name, the range it may span, and whether
that range is linear or logarithmic. It is the shared vocabulary between
whatever produced the experiments (a seed design, a historical campaign, a
spreadsheet) and `acquisition.py`, which needs the same bounds to avoid
recommending an infeasible composition.

Seed designs are deliberately not generated here — that happens once per
campaign, and the design usually arrives from whoever is running the lab. To
build one on the fly, `decode` maps a coded design onto real units::

    coded = pyDOE3.bbdesign(len(factors))              # or any [-1, +1] design
    df = pd.DataFrame({f.name: f.decode(coded[:, j])
                       for j, f in enumerate(factors)})

Keep that column order aligned with the `predictors` you pass to
`glm.fit_gamma_glm`, whose term expansion is positional.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

__all__ = ["Factor", "FactorScale"]

FactorScale = Literal["linear", "log"]


@dataclass(frozen=True, slots=True)
class Factor:
    """One experimental variable and the range it may be varied over.

    `scale` decides where the midpoint of the range falls. Media
    concentrations often span an order of magnitude, where the arithmetic
    midpoint is not the experimentally meaningful one: a factor over
    0.1-10 g/L has arithmetic center 5.05 but geometric center 1.0. Designs
    like Box-Behnken replicate their center runs, so the choice spends real
    reagents.
    """

    name: str
    low: float
    high: float
    units: str = ""
    scale: FactorScale = "linear"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("factor name must not be empty")
        if not math.isfinite(self.low) or not math.isfinite(self.high):
            raise ValueError(f"{self.name}: low and high must be finite, got {self.low}, {self.high}")
        if not self.low < self.high:
            raise ValueError(f"{self.name}: low must be < high, got {self.low} and {self.high}")
        if self.scale not in ("linear", "log"):
            raise ValueError(f"{self.name}: scale must be 'linear' or 'log', got {self.scale!r}")
        if self.scale == "log" and self.low <= 0:
            raise ValueError(f"{self.name}: log scale requires low > 0, got {self.low}")

    @property
    def center(self) -> float:
        """The midpoint a design's center runs sit at — geometric under log scale."""
        if self.scale == "log":
            return math.sqrt(self.low * self.high)
        return (self.low + self.high) / 2

    def decode(self, coded: np.ndarray) -> np.ndarray:
        """Coded `[-1, +1]` to real units. `-1` is `low`, `0` is `center`, `+1` is `high`.

        Clipped to the range: without it, floating-point error at the corners
        would put decoded design points a few ulps outside their own bounds and
        make `contains` reject them.
        """
        t = (np.asarray(coded, dtype=float) + 1.0) / 2.0
        if self.scale == "log":
            real = self.low * (self.high / self.low) ** t
        else:
            real = self.low + t * (self.high - self.low)
        return np.clip(real, self.low, self.high)

    def encode(self, real: np.ndarray) -> np.ndarray:
        """Real units to coded `[-1, +1]`. The inverse of `decode`.

        This is the standardization the model fits on. It is derived from the
        declared range rather than from the observed data, so it does not drift
        as a campaign's sampling concentrates near an optimum — which keeps
        coefficients comparable across model versions and keeps the priors
        meaning the same thing every round.

        Values outside the range encode to magnitudes above 1 rather than
        being clipped; that is real information about an extrapolative point.
        """
        real = np.asarray(real, dtype=float)
        if self.scale == "log":
            if np.any(real <= 0):
                raise ValueError(f"{self.name}: log scale requires strictly positive values")
            return (np.log(real) - math.log(self.center)) / (
                (math.log(self.high) - math.log(self.low)) / 2
            )
        return (real - self.center) / ((self.high - self.low) / 2)

    def contains(self, value: np.ndarray) -> np.ndarray:
        """Whether values lie in the feasible range, elementwise."""
        value = np.asarray(value, dtype=float)
        return (value >= self.low) & (value <= self.high)
