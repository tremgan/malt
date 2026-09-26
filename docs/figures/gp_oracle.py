"""Render docs/figures/gp_oracle.svg: a GP-sampled growth surface.

    uv run python docs/figures/gp_oracle.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from _surface import FACTORS, render_surface

from malt.simulation.oracle import gp_sampled_latent

SEED = 11

render_surface(
    gp_sampled_latent(FACTORS, np.random.default_rng(SEED), lengthscale=0.6, sd=0.5, mean=np.log(4.0)),
    title="A GP-sampled growth surface",
    subtitle="Mean biomass. RBF lengthscale 0.6 (coded units), log-scale sd 0.5, baseline 4 g/L.",
    out=Path(__file__).with_suffix(".svg"),
)
