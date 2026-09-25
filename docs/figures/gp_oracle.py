"""Render docs/figures/gp_oracle.png: a GP-sampled growth surface seen through Gamma noise.

    uv run python docs/figures/gp_oracle.py
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from mpl_toolkits.mplot3d.axes3d import Axes3D  # noqa: E402

from malt.benchmark.oracle import GammaLikelihood, Oracle, gp_sampled_latent  # noqa: E402
from malt.engine.factors import Factor  # noqa: E402

SEED = 7
OUT = Path(__file__).with_suffix(".png")

# Wheat to green, after 🌾: beige where biomass is low, deep green at the peak.
# Lightness falls monotonically (OKLab L 0.94 -> 0.41), so it reads as one
# sequential scale despite the hue change.
WHEAT = LinearSegmentedColormap.from_list(
    "wheat", ["#f4ecd6", "#e6d5a8", "#d2c07c", "#a9b062", "#6f9a4f", "#3f7a3c", "#24562a"]
)
INK, INK_MUTED, GRID = "#1f1f1e", "#6b6a66", "#e4e3df"

glucose = Factor("glucose", 0.5, 20.0, units="g/L")
nitrogen = Factor("nitrogen", 0.1, 10.0, units="g/L", scale="log")
factors = (glucose, nitrogen)

rng = np.random.default_rng(SEED)
latent = gp_sampled_latent(factors, rng, lengthscale=0.6, sd=0.5, mean=np.log(4.0))
oracle = Oracle(latent, GammaLikelihood(alpha=20.0))

# Plot in coded units so the log factor is evenly spaced, and label ticks in real units.
z = np.linspace(-1, 1, 80)
Z1, Z2 = np.meshgrid(z, z)
grid = pd.DataFrame({"glucose": glucose.decode(Z1.ravel()), "nitrogen": nitrogen.decode(Z2.ravel())})
mu = oracle.mean(grid).reshape(Z1.shape)

coded_points = rng.uniform(-1, 1, (40, 2))
points = pd.DataFrame(
    {"glucose": glucose.decode(coded_points[:, 0]), "nitrogen": nitrogen.decode(coded_points[:, 1])}
)
observed = oracle.query(points, rng)["y"].to_numpy()
truth = oracle.mean(points)

plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "text.color": INK})
fig = plt.figure(figsize=(8, 6.4), dpi=200)
ax = cast(Axes3D, fig.add_axes((0.0, 0.0, 1.0, 0.86), projection="3d", computed_zorder=False))
ax.set_box_aspect(None, zoom=1.12)  # pyright: ignore[reportArgumentType] — stub types zoom as int

ax.plot_surface(Z1, Z2, mu, cmap=WHEAT, linewidth=0, antialiased=True, alpha=0.8, zorder=1)
for (c1, c2), t, y in zip(coded_points, truth, observed):
    ax.plot([c1, c1], [c2, c2], [t, y], color=INK_MUTED, linewidth=0.8, zorder=2)
ax.scatter(
    coded_points[:, 0], coded_points[:, 1], observed,  # pyright: ignore[reportArgumentType] — stub types zs as int
    s=22, color=INK, edgecolors="white", linewidths=0.8, depthshade=False, zorder=3,
)

glucose_ticks = [0.5, 5, 10, 15, 20]
nitrogen_ticks = [0.1, 1, 10]
ax.set_xticks(glucose.encode(np.array(glucose_ticks)), [f"{t:g}" for t in glucose_ticks])
ax.set_yticks(nitrogen.encode(np.array(nitrogen_ticks)), [f"{t:g}" for t in nitrogen_ticks])
ax.set_xlabel("glucose (g/L)", color=INK_MUTED, labelpad=6)
ax.set_ylabel("nitrogen (g/L, log scale)", color=INK_MUTED, labelpad=6)
ax.set_zlabel("biomass (g/L)", color=INK_MUTED, labelpad=4)
ax.tick_params(colors=INK_MUTED, labelsize=8)
for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis.set_pane_color((1, 1, 1, 0))  # pyright: ignore[reportAttributeAccessIssue] — Axis3D, typed as its 2D parent
    axis._axinfo["grid"].update(color=GRID, linewidth=0.6)  # type: ignore[attr-defined]
ax.view_init(elev=28, azim=-128)

fig.legend(
    handles=[
        Patch(facecolor="#6f9a4f", label="true mean biomass"),
        Line2D([], [], marker="o", linestyle="", markersize=5, color=INK,
               markeredgecolor="white", label="Gamma observations (α = 20)"),
        Line2D([], [], color=INK_MUTED, linewidth=0.8, label="noise: observation − mean"),
    ],
    loc="upper left", bbox_to_anchor=(0.04, 0.905), frameon=False, fontsize=8.5,
    ncols=3, handlelength=1.4, columnspacing=1.6,
)
fig.text(0.04, 0.975, "A GP-sampled growth surface, observed through Gamma noise",
         fontsize=12.5, fontweight="bold", color=INK, va="top")
fig.text(0.04, 0.938, "Noise grows with the mean: the tallest stems sit where biomass is highest. "
         "RBF lengthscale 0.6 (coded), log-scale sd 0.5, baseline 4 g/L.",
         fontsize=8.5, color=INK_MUTED, va="top")

fig.savefig(OUT, bbox_inches="tight", facecolor="white")
print(f"wrote {OUT}")
