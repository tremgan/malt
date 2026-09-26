"""Regret benchmark: BO (Gamma GLM + q-NEI) against iterative RSM and random.

    uv run python -m benchmarks.three_factor.regret --replicates 20

Every campaign starts where a lab would: a face-centred CCD with 3 centre runs
(17 runs) around a fixed house recipe, not spanning the whole feasible space.
A full-range CCD on an in-family quadratic surface already recommends within
about 2% of the best, which leaves the rounds nothing to do; the local seed
leaves about a third of the achievable biomass to find. Each replicate draws a
fresh surface and noise stream, and every arm sees both: the arms are paired,
so differences between them are not luck of the draw. Two families of surface:

- quadratic — a single tilted peak at a random location, in the GLM's own family;
- gp — a GP-sampled surface, which no quadratic fits exactly.

The score is cumulative regret over the runs each arm chose (see
`malt.simulation.regret`): every run adds the fraction of the best achievable
biomass it gave up, so the curves read in optimal runs' worth of biomass lost.
It scores the acquisition's choices, not a model's recommendation, so a
model-free rule scores the same whatever model it is paired with.

Sharper peaks are not a fair way to make the quadratic harder: beyond a
curvature of about 2 the GLM's coefficient priors, not the acquisition, decide
the outcome.

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
from scipy import stats  # noqa: E402

from malt.active_learning.acquisitions import (  # noqa: E402
    CentralComposite,
    QNoisyExpectedImprovement,
    RandomBatch,
)
from malt.active_learning.surrogates import BayesianLinearRegression, GammaGLMSurrogate  # noqa: E402
from malt.simulation.regret import Arm, run_arm  # noqa: E402
from malt.engine.acquisition import central_composite  # noqa: E402
from malt.engine.factors import Factor, decode_design  # noqa: E402
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
BATCH = 16  # a 3-factor CCD is 14 runs; two centre replicates fill the batch
ROUNDS = 4
LIKELIHOOD = GammaLikelihood(20.0)  # CV about 22%

ARMS = (
    Arm("q-NEI, Gamma GLM", GammaGLMSurrogate(FACTORS), QNoisyExpectedImprovement(FACTORS)),
    Arm("RSM, BLR + CCD", BayesianLinearRegression(FACTORS, "quadratic"), CentralComposite(FACTORS)),
    Arm("Random, Gamma GLM", GammaGLMSurrogate(FACTORS), RandomBatch(FACTORS)),
)


def quadratic_surface(rng: np.random.Generator) -> LatentFunction:
    """A 10 g/L peak somewhere in the inner 80% of every range, tilted at random.

    The curvature matrix has a random orientation and axes between 0.3 and 1.7,
    so the peak interacts across factors instead of lining up with them.
    """
    k = len(FACTORS)
    coded = rng.uniform(-0.8, 0.8, k)
    optimum = {f.name: float(f.decode(coded[j])) for j, f in enumerate(FACTORS)}
    rotation, _ = np.linalg.qr(rng.normal(size=(k, k)))
    curvature = rotation @ np.diag(rng.uniform(0.3, 1.7, k)) @ rotation.T
    return quadratic_latent(FACTORS, optimum=optimum, peak=np.log(10.0), curvature=(curvature + curvature.T) / 2)


def gp_surface(rng: np.random.Generator) -> LatentFunction:
    """A GP draw around 5 g/L, varying by a factor of about e."""
    return gp_sampled_latent(FACTORS, rng, lengthscale=0.6, sd=0.5, mean=np.log(5.0))


SURFACES = {"quadratic": quadratic_surface, "gp": gp_surface}


HOUSE_RECIPE = -0.4  # coded, in every factor: a standard medium off the centre of the space


def seed_design() -> pd.DataFrame:
    """A face-centred CCD of half-width 0.4 (coded) around the house recipe, with 3 centre runs."""
    return decode_design(FACTORS, central_composite(np.full(len(FACTORS), HOUSE_RECIPE), radius=0.4, n_center=3))


def slug(name: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in name.lower()).strip("-")


def run_replicate(surface: str, replicate: int) -> list[pd.DataFrame]:
    """Every arm on one replicate's surface, seed and noise stream."""
    # Independent streams for the surface, the seed data and the campaigns.
    kind = list(SURFACES).index(surface)
    surface_ss, seed_ss, campaign_ss = np.random.SeedSequence([MASTER_SEED, kind, replicate]).spawn(3)
    oracle = Oracle(SURFACES[surface](np.random.default_rng(surface_ss)), LIKELIHOOD)
    x = seed_design()
    seed_data = x.join(oracle.query(x, np.random.default_rng(seed_ss)).set_axis(x.index))
    random_seed = int(campaign_ss.generate_state(1)[0])

    curves = []
    for arm in ARMS:
        path = RESULTS / "runs" / surface / slug(arm.name) / f"{replicate:03d}.csv"
        if path.exists():
            curve = pd.read_csv(path)
        else:
            curve = run_arm(
                arm, oracle, seed_data, FACTORS,
                batch_size=BATCH, rounds=ROUNDS, random_seed=random_seed,
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            curve.to_csv(path, index=False)
        print(f"{surface:9} rep {replicate:3d}  {arm.name:18}  cumulative regret "
              f"{curve['cumulative_regret'].iloc[-1]:6.2f}"
              + ("  (stopped: failed fit)" if curve["stopped"].to_numpy().any() else ""), flush=True)
        curves.append(curve.assign(surface=surface, replicate=replicate, arm=arm.name))
    return curves


# Categorical slots 1-3 of the dataviz reference palette, validated on this surface.
COLORS = dict(zip([a.name for a in ARMS], ["#2a78d6", "#eb6834", "#1baf7a"]))
SURFACE_BG, INK, INK_MUTED, GRID = "#fcfcfb", "#1f1f1e", "#6b6a66", "#e4e3df"
TITLES = {"quadratic": "One peak, quadratic in log biomass", "gp": "GP-sampled surface: no quadratic fits"}


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
        results.groupby(["surface", "arm", "round"])["cumulative_regret"]
        .agg(["mean", "sem", "count"])
        .reset_index()
    )
    # 90% t-interval for the mean across replicates.
    half = stats.t.ppf(0.95, summary["count"].to_numpy() - 1) * summary["sem"].to_numpy()
    half = np.nan_to_num(half)  # a single replicate has no interval; draw the mean alone
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
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK_MUTED, labelsize=9)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.set_xlim(right=ax.get_xlim()[1] + 0.6)  # room for the direct labels
    axes[0].set_ylim(min(0.0, np.nanmin(summary["lo"].to_numpy())), top)
    axes[0].set_ylabel("Cumulative regret\n(optimal runs' worth of biomass lost)", color=INK_MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False,
               fontsize=9, labelcolor=INK)
    n_reps = results["replicate"].nunique()
    fig.text(
        0.01, 0.015,
        f"Cumulative regret: the sum over every run an arm chose of 1 − f*(x)/max f*; the shared seed counts 0. "
        f"Lines: mean over {n_reps} paired campaigns.\nShaded bands: 90% confidence interval for that mean "
        f"(t-interval, mean ± t × standard error), not the spread of individual campaigns.",
        fontsize=8.5, color=INK_MUTED, ha="left", va="bottom",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
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
    print("\nMedian cumulative regret after the last round, and campaigns stopped by a failed fit:")
    print(final.groupby(["surface", "arm"]).agg(
        median_cumulative_regret=("cumulative_regret", "median"), stopped=("stopped", "sum")))


if __name__ == "__main__":
    main()
