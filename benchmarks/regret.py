"""Regret benchmark: BO (Gamma GLM + q-NEI) against iterative RSM and random.

    uv run python -m benchmarks.regret --replicates 20

Each replicate draws a fresh surface, a 15-run random seed, and a noise
stream, and every arm sees all three: the arms are paired, so differences
between them are not luck of the draw. Two families of surface:

- quadratic — a single peak at a random location, in the GLM's own family;
- gp — a GP-sampled surface, which no quadratic fits exactly.

Each run is cached under `results/runs/`, so an interrupted benchmark resumes
where it stopped. The aggregate goes to `results/regret.csv` and the figure
to `results/regret.png`.
"""

from __future__ import annotations

import os

# nutpie needs PyTensor's C backend off on macOS 26+ (see CLAUDE.md).
os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")

import argparse  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402
from scipy import stats  # noqa: E402

from malt.active_learning.acquisitions import (  # noqa: E402
    CentralComposite,
    QNoisyExpectedImprovement,
    RandomBatch,
)
from malt.active_learning.surrogates import BayesianLinearRegression, GammaGLMSurrogate  # noqa: E402
from malt.simulation.regret import Arm, run_arm  # noqa: E402
from malt.engine.factors import Factor, candidate_grid  # noqa: E402
from malt.engine.glm import ConvergenceWarning  # noqa: E402
from malt.simulation.oracle import (  # noqa: E402
    GammaLikelihood,
    LatentFunction,
    Oracle,
    gp_sampled_latent,
    quadratic_latent,
)

RESULTS = Path(__file__).parent / "results"
MASTER_SEED = 20260925

FACTORS = (
    Factor("glucose", 0.5, 20.0, units="g/L"),
    Factor("nitrogen", 0.1, 10.0, units="g/L", scale="log"),
    Factor("phosphate", 0.05, 2.0, units="g/L", scale="log"),
)
CANDIDATES = candidate_grid(FACTORS, 11)
SEED_SIZE = 15
BATCH = 16  # a 3-factor CCD is 14 runs; two centre replicates fill the batch
ROUNDS = 4
LIKELIHOOD = GammaLikelihood(20.0)  # CV about 22%

ARMS = (
    Arm("q-NEI, Gamma GLM", GammaGLMSurrogate(FACTORS), QNoisyExpectedImprovement(CANDIDATES)),
    Arm("RSM, BLR + CCD", BayesianLinearRegression(FACTORS, "quadratic"), CentralComposite(FACTORS, CANDIDATES)),
    Arm("Random, Gamma GLM", GammaGLMSurrogate(FACTORS), RandomBatch(CANDIDATES)),
)


def quadratic_surface(rng: np.random.Generator) -> LatentFunction:
    """A 10 g/L peak somewhere in the inner 60% of every range, falling by e at its edges."""
    coded = rng.uniform(-0.6, 0.6, len(FACTORS))
    optimum = {f.name: float(f.decode(coded[j])) for j, f in enumerate(FACTORS)}
    return quadratic_latent(FACTORS, optimum=optimum, peak=np.log(10.0), curvature=1.0)


def gp_surface(rng: np.random.Generator) -> LatentFunction:
    """A GP draw around 5 g/L, varying by a factor of about e."""
    return gp_sampled_latent(FACTORS, rng, lengthscale=0.6, sd=0.5, mean=np.log(5.0))


SURFACES = {"quadratic": quadratic_surface, "gp": gp_surface}


def slug(name: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in name.lower()).strip("-")


def run_replicate(surface: str, replicate: int) -> list[pd.DataFrame]:
    """Every arm on one replicate's surface, seed and noise stream."""
    # Independent streams for the surface, the seed data and the campaigns.
    kind = list(SURFACES).index(surface)
    surface_ss, seed_ss, campaign_ss = np.random.SeedSequence([MASTER_SEED, kind, replicate]).spawn(3)
    oracle = Oracle(SURFACES[surface](np.random.default_rng(surface_ss)), LIKELIHOOD)
    seed_rng = np.random.default_rng(seed_ss)
    x = CANDIDATES.iloc[seed_rng.choice(len(CANDIDATES), SEED_SIZE, replace=False)]
    seed_data = x.join(oracle.query(x, seed_rng).set_axis(x.index))
    random_seed = int(campaign_ss.generate_state(1)[0])

    curves = []
    for arm in ARMS:
        path = RESULTS / "runs" / surface / slug(arm.name) / f"{replicate:03d}.csv"
        if path.exists():
            curve = pd.read_csv(path)
        else:
            curve = run_arm(
                arm, oracle, seed_data, CANDIDATES,
                batch_size=BATCH, rounds=ROUNDS, random_seed=random_seed,
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            curve.to_csv(path, index=False)
        print(f"{surface:9} rep {replicate:3d}  {arm.name:18}  final regret {curve['regret'].iloc[-1]:6.1%}"
              + ("  (stopped: failed fit)" if curve["stopped"].to_numpy().any() else ""), flush=True)
        curves.append(curve.assign(surface=surface, replicate=replicate, arm=arm.name))
    return curves


# Categorical slots 1-3 of the dataviz reference palette, validated on this surface.
COLORS = dict(zip([a.name for a in ARMS], ["#2a78d6", "#eb6834", "#1baf7a"]))
SURFACE_BG, INK, INK_MUTED, GRID = "#fcfcfb", "#1f1f1e", "#6b6a66", "#e4e3df"
TITLES = {"quadratic": "In family: one quadratic peak", "gp": "Misspecified: GP-sampled surface"}


def spread(ys: list[float], gap: float) -> list[float]:
    """Nudge label positions apart so no two are closer than `gap`, keeping their order."""
    order = np.argsort(ys)
    placed = np.array(ys, dtype=float)[order]
    for i in range(1, len(placed)):
        placed[i] = max(placed[i], placed[i - 1] + gap)
    out = np.empty_like(placed)
    out[order] = placed
    return list(out)


def plot(results: pd.DataFrame, out: Path) -> None:
    summary = (
        results.groupby(["surface", "arm", "round"])["regret"]
        .agg(["mean", "sem", "count"])
        .reset_index()
    )
    # 90% t-interval for the mean across replicates.
    half = stats.t.ppf(0.95, summary["count"].to_numpy() - 1) * summary["sem"].to_numpy()
    summary["lo"], summary["hi"] = summary["mean"] - half, summary["mean"] + half
    top = np.nanmax(summary["hi"].to_numpy()) * 1.08
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True, facecolor=SURFACE_BG)
    for ax, surface in zip(axes, SURFACES):
        ax.set_facecolor(SURFACE_BG)
        ends = {}
        for arm, color in COLORS.items():
            s = summary.loc[(summary["surface"] == surface) & (summary["arm"] == arm)]
            ax.fill_between(s["round"], s["lo"], s["hi"], color=color, alpha=0.14, linewidth=0)
            ax.plot(s["round"], s["mean"], color=color, linewidth=2, marker="o", markersize=5,
                    markeredgecolor=SURFACE_BG, markeredgewidth=1.5, label=arm)
            ends[arm] = (s["round"].to_numpy()[-1], s["mean"].to_numpy()[-1])
        label_ys = spread([y for _, y in ends.values()], gap=0.07 * top)
        for (arm, (x, _)), y in zip(ends.items(), label_ys):
            ax.annotate(arm.split(",")[0], (x, y), xytext=(8, 0), textcoords="offset points",
                        va="center", fontsize=9, color=INK)
        n = results.loc[results["surface"] == surface, "replicate"].nunique()
        ax.set_title(f"{TITLES[surface]}  (n = {n})", loc="left", fontsize=11, color=INK)
        ax.set_xlabel(f"Round (0 = seed; {BATCH} runs per round)", color=INK_MUTED)
        ax.set_xticks(sorted(results["round"].unique()))
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK_MUTED, labelsize=9)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.set_xlim(right=ax.get_xlim()[1] + 0.6)  # room for the direct labels
    axes[0].set_ylim(min(0.0, np.nanmin(summary["lo"].to_numpy())), top)
    axes[0].set_ylabel("Relative regret: biomass lost vs. best\n(mean, 90% CI)", color=INK_MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False,
               fontsize=9, labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=160, facecolor=SURFACE_BG)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replicates", type=int, default=20)
    args = parser.parse_args()

    # A failed convergence gate stops that campaign; the curve records it as `stopped`.
    warnings.simplefilter("ignore", ConvergenceWarning)
    curves = [c for surface in SURFACES for r in range(args.replicates) for c in run_replicate(surface, r)]
    results = pd.concat(curves, ignore_index=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS / "regret.csv", index=False)
    plot(results, RESULTS / "regret.png")

    final = results[results["round"] == ROUNDS]
    print("\nMedian final regret, and campaigns stopped by a failed fit:")
    print(final.groupby(["surface", "arm"]).agg(median_regret=("regret", "median"), stopped=("stopped", "sum")))


if __name__ == "__main__":
    main()
