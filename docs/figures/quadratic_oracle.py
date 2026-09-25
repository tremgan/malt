"""Render docs/figures/quadratic_oracle.png: a single peaked growth surface (README headline).

    uv run python docs/figures/quadratic_oracle.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from _surface import FACTORS, render_surface

from malt.simulation.oracle import quadratic_latent

render_surface(
    quadratic_latent(
        FACTORS,
        optimum={"glucose": 12.0, "nitrogen": 2.0},
        peak=np.log(12.0),
        # A strong glucose-nitrogen interaction stretches the peak into a
        # diagonal ridge: more glucose only pays off with more nitrogen.
        curvature=np.array([[3.4, -2.2], [-2.2, 3.4]]),
    ),
    title="A realistic synthetic growth surface",
    subtitle="Mean biomass. Quadratic on the log scale, optimum 12 g/L glucose and 2 g/L nitrogen;\n"
    "the diagonal ridge is the interaction: more glucose only pays off with more nitrogen.",
    out=Path(__file__).with_suffix(".png"),
    height=0.5,
    azim=-62,  # close to side-on, so the ridge isn't foreshortened
)
