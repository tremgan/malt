"""Render docs/figures/loop.svg: where malt sits in a round of experiments.

    uv run python docs/figures/loop.py

The only loop drawing in the repo was the ASCII one in `active_learning.actors`,
which shows `Experimenter` querying an `Environment`. That is the simulation
path, and it is the wrong picture for a lab: `Environment` closes the loop with
no approval step. This draws the real one, where the loop crosses out of malt
twice and a person stands in it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402


@dataclass(frozen=True)
class Theme:
    """Chart tokens for one colour scheme; see `benchmarks.harness.Theme`."""

    surface: str
    ink: str
    ink_muted: str
    lane: str
    box: str
    malt: str
    suffix: str


LIGHT = Theme("#fcfcfb", "#1f1f1e", "#6b6a66", "#f2f1ed", "#ffffff", "#2a78d6", "")
DARK = Theme("#1a1a19", "#ffffff", "#898781", "#232321", "#2c2c2a", "#3987e5", "_dark")

BOX_W, BOX_H = 3.0, 0.92
# Wide enough apart that a label fits in the gap without touching either box.
LEFT, RIGHT = 2.7, 7.7
# One rectangular circuit: down the malt lane, across, down, and back across.
STEPS = [
    ("fit", LEFT, 4.6, "a Gamma GLM over every run", True),
    ("propose", LEFT, 3.2, "q-NEI picks the next batch", True),
    ("approve", RIGHT, 3.2, "a person signs the batch off", False),
    ("run", RIGHT, 1.5, "the deck, or somebody's hands", False),
    ("observe", LEFT, 1.5, "the results go back in", True),
]


def arrow(ax, start, end, theme, label=None, rad=0.0, offset=(0, 0)):
    ax.add_patch(
        FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=13, linewidth=1.3,
            color=theme.ink_muted, connectionstyle=f"arc3,rad={rad}", zorder=1,
        )
    )
    if label:
        mid = ((start[0] + end[0]) / 2 + offset[0], (start[1] + end[1]) / 2 + offset[1])
        ax.text(*mid, label, ha="center", va="center", fontsize=8.5, color=theme.ink_muted)


def render(out: Path) -> None:
    for theme in (LIGHT, DARK):
        plt.rcParams.update({"font.family": "sans-serif"})
        fig, ax = plt.subplots(figsize=(9.4, 5.0), facecolor=theme.surface)
        ax.set_xlim(-0.1, 10.5)
        ax.set_ylim(0.4, 6.0)
        ax.axis("off")

        for x, name in ((LEFT, "malt"), (RIGHT, "your lab")):
            ax.add_patch(FancyBboxPatch(
                (x - BOX_W / 2 - 0.35, 0.8), BOX_W + 0.7, 4.5,
                boxstyle="round,pad=0.02,rounding_size=0.12",
                facecolor=theme.lane, edgecolor="none", zorder=0))
            ax.text(x, 5.55, name, ha="center", va="center", fontsize=11,
                    fontweight="bold", color=theme.ink)

        for name, x, y, caption, is_malt in STEPS:
            colour = theme.malt if is_malt else theme.ink_muted
            # The gate is drawn, not annotated: a dashed edge reads as a stop.
            style = (0, (4, 2.5)) if name == "approve" else "solid"
            ax.add_patch(FancyBboxPatch(
                (x - BOX_W / 2, y - BOX_H / 2), BOX_W, BOX_H,
                boxstyle="round,pad=0.02,rounding_size=0.1",
                facecolor=theme.box, edgecolor=colour, linewidth=1.6,
                linestyle=style, zorder=2))
            ax.text(x, y + 0.17, name, ha="center", va="center", fontsize=10.5,
                    fontweight="bold", color=colour, zorder=3)
            ax.text(x, y - 0.22, caption, ha="center", va="center", fontsize=8.5,
                    color=theme.ink_muted, zorder=3)

        half, edge = BOX_H / 2, BOX_W / 2
        arrow(ax, (LEFT, 4.6 - half), (LEFT, 3.2 + half), theme)
        arrow(ax, (LEFT + edge, 3.2), (RIGHT - edge, 3.2), theme,
              "a batch of recipes", offset=(0, 0.25))
        arrow(ax, (RIGHT, 3.2 - half), (RIGHT, 1.5 + half), theme)
        arrow(ax, (RIGHT - edge, 1.5), (LEFT + edge, 1.5), theme,
              "what they measured", offset=(0, 0.25))
        # Back round the outside of the lane, or it draws straight through the boxes.
        arrow(ax, (LEFT - edge, 1.5), (LEFT - edge - 0.2, 4.6), theme, rad=-0.55)
        ax.text(0.28, 3.05, "next round", ha="center", va="center", rotation=90,
                fontsize=8.5, color=theme.ink_muted)
        fig.text(0.5, 0.045,
                 "malt proposes and observes. Approving a batch and running it are not function calls, "
                 "and the two can be days apart.",
                 ha="center", fontsize=9, color=theme.ink_muted)

        target = out.with_name(f"{out.stem}{theme.suffix}{out.suffix}")
        fig.savefig(target, bbox_inches="tight", facecolor=theme.surface)
        plt.close(fig)
        print(f"wrote {target}")


if __name__ == "__main__":
    render(Path(__file__).with_suffix(".svg"))
