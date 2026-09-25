"""Shared 3D surface plot for the README figures: mean biomass over glucose x nitrogen."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from mpl_toolkits.mplot3d.axes3d import Axes3D  # noqa: E402

from malt.simulation.oracle import LatentFunction  # noqa: E402
from malt.engine.factors import Factor  # noqa: E402

# Wheat to green, after 🌾: beige where biomass is low, deep green at the peak.
# Lightness falls monotonically (OKLab L 0.94 -> 0.41), so it reads as one
# sequential scale despite the hue change.
WHEAT = LinearSegmentedColormap.from_list(
    "wheat", ["#f4ecd6", "#e6d5a8", "#d2c07c", "#a9b062", "#6f9a4f", "#3f7a3c", "#24562a"]
)
INK, INK_MUTED, GRID = "#1f1f1e", "#6b6a66", "#e4e3df"

GLUCOSE = Factor("glucose", 0.5, 20.0, units="g/L")
NITROGEN = Factor("nitrogen", 0.1, 10.0, units="g/L", scale="log")
FACTORS = (GLUCOSE, NITROGEN)


def render_surface(
    latent: LatentFunction,
    *,
    title: str,
    subtitle: str,
    out: Path,
    azim: float = -128,
    height: float = 0.75,
) -> None:
    """Plot `exp(latent)` — mean biomass under the log link — and save it to `out`.

    `height` is the vertical box size relative to the floor; matplotlib's own
    default is 0.75, and lower flattens the plot.
    """
    # Plot in coded units so the log factor is evenly spaced; label ticks in real units.
    z = np.linspace(-1, 1, 80)
    Z1, Z2 = np.meshgrid(z, z)
    grid = pd.DataFrame({"glucose": GLUCOSE.decode(Z1.ravel()), "nitrogen": NITROGEN.decode(Z2.ravel())})
    mu = np.exp(latent(grid)).reshape(Z1.shape)

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "text.color": INK})
    fig = plt.figure(figsize=(8, 6.4), dpi=200)
    ax = cast(Axes3D, fig.add_axes((0.0, 0.0, 1.0, 0.9), projection="3d"))
    ax.set_box_aspect((1, 1, height), zoom=1.12)  # pyright: ignore[reportArgumentType] — stub types zoom as int
    ax.plot_surface(Z1, Z2, mu, cmap=WHEAT, linewidth=0, antialiased=True)

    glucose_ticks = [0.5, 5, 10, 15, 20]
    nitrogen_ticks = [0.1, 1, 10]
    ax.set_xticks(GLUCOSE.encode(np.array(glucose_ticks)), [f"{t:g}" for t in glucose_ticks])
    ax.set_yticks(NITROGEN.encode(np.array(nitrogen_ticks)), [f"{t:g}" for t in nitrogen_ticks])
    ax.set_xlabel("glucose (g/L)", color=INK_MUTED, labelpad=6)
    ax.set_ylabel("nitrogen (g/L, log scale)", color=INK_MUTED, labelpad=6)
    ax.set_zlabel("biomass (g/L)", color=INK_MUTED, labelpad=4)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1, 1, 1, 0))  # pyright: ignore[reportAttributeAccessIssue] — Axis3D, typed as its 2D parent
        axis._axinfo["grid"].update(color=GRID, linewidth=0.6)  # type: ignore[attr-defined]
    ax.view_init(elev=28, azim=azim)

    fig.text(0.04, 0.975, title, fontsize=12.5, fontweight="bold", color=INK, va="top")
    fig.text(0.04, 0.938, subtitle, fontsize=8.5, color=INK_MUTED, va="top")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")
