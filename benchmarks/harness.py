"""The paired regret benchmark, shared by every `benchmarks/<n>_factor/regret.py`.

A `Benchmark` is a configuration — factors, arms, seed design, batch size,
rounds — and `main` runs it: for each surface family and replicate it draws a
fresh surface, seed data and noise stream, and every arm sees all three, so
the arms are paired and differences between them are not luck of the draw.

The score is cumulative regret over the runs each arm chose (see
`malt.simulation.regret`): every run adds the fraction of the best achievable
biomass it gave up, so the curves read in optimal runs' worth of biomass lost.
It scores the acquisition's choices, not a model's recommendation, so a
model-free rule scores the same whatever model it is paired with.

Each run is cached under `<results>/runs/`, so an interrupted benchmark
resumes where it stopped. The aggregate goes to `<results>/regret.csv` and
the figure to `<results>/regret.svg`.
"""

from __future__ import annotations

import os

# nutpie needs PyTensor's C backend off on macOS 26+ (see CLAUDE.md).
os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")

import argparse  # noqa: E402
import warnings  # noqa: E402
from collections.abc import Callable  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from itertools import repeat  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

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
from malt.simulation.regret import Arm, run_arm  # noqa: E402

MASTER_SEED = 20260925
LIKELIHOOD = GammaLikelihood(20.0)  # CV about 22%
HOUSE_RECIPE = -0.4  # coded, in every factor: a standard medium off the centre of the space

SurfaceFamily = Callable[[np.random.Generator], LatentFunction]


INOCULUM = 0.3  # g/L: biomass never falls below what was seeded, however poor the medium
PEAK = 10.0  # g/L


def quadratic_surfaces(factors: tuple[Factor, ...]) -> SurfaceFamily:
    """A sharp 10 g/L peak above a 0.3 g/L inoculum floor, somewhere in the inner 80% of every range.

    `mu = floor + (peak - floor) * exp(-q(z))`, with `q` a quadratic in coded
    units. Without the floor a sharp log-quadratic sends distant runs to
    1e-6 g/L — seven orders of magnitude no culture shows — and the fits break
    on it. With it the surface is log-quadratic near the peak and levels off
    at the inoculum far away, so no model here is exactly in family.

    Curvature scales are 2-4 per factor: one coded unit from the optimum costs
    a factor of e^2 to e^4 in biomass. Every pair of factors interacts
    positively — a soft AND: more of one only pays off with more of the other,
    so the peak is a ridge along "more of everything together". The curvature
    matrix is `D C D`, with `C` unit-diagonal and off-diagonal `-r`; it stays
    positive definite for `r < 1 / (k - 1)`, and `r` is drawn at 40-80% of that.
    """

    def draw(rng: np.random.Generator) -> LatentFunction:
        k = len(factors)
        coded = rng.uniform(-0.8, 0.8, k)
        optimum = {f.name: float(f.decode(coded[j])) for j, f in enumerate(factors)}
        scales = np.sqrt(rng.uniform(2.0, 4.0, k))
        r = rng.uniform(0.4, 0.8) / (k - 1)
        coupling = (1 + r) * np.eye(k) - r * np.ones((k, k))
        curvature = scales[:, None] * coupling * scales[None, :]
        peaked = quadratic_latent(factors, optimum=optimum, peak=np.log(PEAK), curvature=curvature)

        def latent(x: pd.DataFrame) -> np.ndarray:
            return np.log(INOCULUM + (1 - INOCULUM / PEAK) * np.exp(peaked(x)))

        return latent

    return draw


def gp_surfaces(factors: tuple[Factor, ...]) -> SurfaceFamily:
    """A GP draw around 5 g/L, varying by a factor of about e."""
    return lambda rng: gp_sampled_latent(factors, rng, lengthscale=0.6, sd=0.5, mean=np.log(5.0))


def local_seed(factors: tuple[Factor, ...]) -> pd.DataFrame:
    """A face-centred CCD of half-width 0.4 (coded) around the house recipe, with 3 centre runs.

    Where a lab would start: around its standard medium, not spanning the whole
    feasible space. A full-range CCD on an in-family quadratic already
    recommends within about 2% of the best, which leaves the rounds nothing to do.
    """
    return decode_design(factors, central_composite(np.full(len(factors), HOUSE_RECIPE), radius=0.4, n_center=3))


@dataclass(frozen=True)
class Benchmark:
    """One benchmark configuration. `surfaces` maps a family name to a surface drawer."""

    factors: tuple[Factor, ...]
    arms: tuple[Arm, ...]
    batch: int
    rounds: int
    results: Path

    @property
    def surfaces(self) -> dict[str, SurfaceFamily]:
        return {"quadratic": quadratic_surfaces(self.factors), "gp": gp_surfaces(self.factors)}


def slug(name: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in name.lower()).strip("-")


def run_replicate(bench: Benchmark, surface: str, replicate: int) -> list[pd.DataFrame]:
    """Every arm on one replicate's surface, seed and noise stream."""
    # Independent streams for the surface, the seed data and the campaigns.
    kind = list(bench.surfaces).index(surface)
    surface_ss, seed_ss, campaign_ss = np.random.SeedSequence([MASTER_SEED, kind, replicate]).spawn(3)
    oracle = Oracle(bench.surfaces[surface](np.random.default_rng(surface_ss)), LIKELIHOOD)
    x = local_seed(bench.factors)
    seed_data = x.join(oracle.query(x, np.random.default_rng(seed_ss)).set_axis(x.index))
    random_seed = int(campaign_ss.generate_state(1)[0])

    curves = []
    for arm in bench.arms:
        path = bench.results / "runs" / surface / slug(arm.name) / f"{replicate:03d}.csv"
        if path.exists():
            curve = pd.read_csv(path)
        else:
            curve = run_arm(
                arm, oracle, seed_data, bench.factors,
                batch_size=bench.batch, rounds=bench.rounds, random_seed=random_seed,
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            curve.to_csv(path, index=False)
        print(f"{surface:9} rep {replicate:3d}  {arm.name:18}  cumulative regret "
              f"{curve['cumulative_regret'].iloc[-1]:6.2f}"
              + ("  (stopped: failed fit)" if curve["stopped"].to_numpy().any() else ""), flush=True)
        curves.append(curve.assign(surface=surface, replicate=replicate, arm=arm.name))
    return curves


# Categorical slots 1-3 of the dataviz reference palette, validated on this surface.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE_BG, INK, INK_MUTED, GRID = "#fcfcfb", "#1f1f1e", "#6b6a66", "#e4e3df"
TITLES = {"quadratic": "One sharp peak above an inoculum floor", "gp": "GP-sampled surface: no quadratic fits"}


def spread(ys: list[float], gap: float) -> list[float]:
    """Nudge label positions apart so no two are closer than `gap`, keeping their order."""
    order = np.argsort(ys)
    placed = np.array(ys, dtype=float)[order]
    for i in range(1, len(placed)):
        placed[i] = max(placed[i], placed[i - 1] + gap)
    out = np.empty_like(placed)
    out[order] = placed
    return list(out)


def plot(bench: Benchmark, results: pd.DataFrame, out: Path) -> None:
    colors = dict(zip([a.name for a in bench.arms], PALETTE))
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
    for ax, surface in zip(axes, bench.surfaces):
        ax.set_facecolor(SURFACE_BG)
        ends = {}
        for arm, color in colors.items():
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
        ax.set_xlabel(f"Round (0 = seed; {bench.batch} runs per round)", color=INK_MUTED)
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
    # Campaigns stopped by a failed fit have no regret for the rounds they never ran,
    # so later means cover fewer campaigns; say so rather than let it pass unseen.
    last = results[results["round"] == bench.rounds]
    stopped = [
        f"{arm.name.split(',')[0]} {n} on {surface}"
        for arm in bench.arms
        for surface in bench.surfaces
        if (n := int(last.loc[(last["arm"] == arm.name) & (last["surface"] == surface), "stopped"].to_numpy().sum()))
    ]
    stop_note = (
        f"Stopped by a failed fit and excluded from later rounds: {'; '.join(stopped)}."
        if stopped else "No campaign was stopped by a failed fit."
    )
    fig.text(
        0.01, 0.015,
        f"{len(bench.factors)} factors. Cumulative regret: the sum over every run an arm chose of 1 − f*(x)/max f*; "
        f"the shared seed counts 0. Lines: mean over {n_reps} paired campaigns.\nShaded bands: 90% confidence "
        f"interval for that mean (t-interval, mean ± t × standard error), not the spread of individual campaigns.\n"
        f"{stop_note}",
        fontsize=8.5, color=INK_MUTED, ha="left", va="bottom",
    )
    fig.tight_layout(rect=(0, 0.11, 1, 0.93))
    fig.savefig(out, dpi=160, facecolor=SURFACE_BG)


def _quiet() -> None:
    # A failed convergence gate stops that campaign; the curve records it as `stopped`.
    warnings.simplefilter("ignore", ConvergenceWarning)


def main(bench: Benchmark, description: str | None = None) -> None:
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--workers", type=int, default=1, help="replicates run in parallel processes")
    args = parser.parse_args()

    jobs = [(surface, r) for surface in bench.surfaces for r in range(args.replicates)]
    if args.workers > 1:
        # Replicates are independent and seeded by (surface, replicate), so the
        # results do not depend on how they are scheduled. One BLAS thread per
        # worker keeps processes from oversubscribing the cores.
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(var, "1")
        with ProcessPoolExecutor(args.workers, initializer=_quiet) as pool:
            parts = list(pool.map(run_replicate, repeat(bench), *zip(*jobs)))
    else:
        _quiet()
        parts = [run_replicate(bench, surface, r) for surface, r in jobs]
    results = pd.concat([c for part in parts for c in part], ignore_index=True)
    bench.results.mkdir(parents=True, exist_ok=True)
    results.to_csv(bench.results / "regret.csv", index=False)
    plot(bench, results, bench.results / "regret.svg")

    final = results[results["round"] == bench.rounds]
    print("\nMedian cumulative regret after the last round, and campaigns stopped by a failed fit:")
    print(final.groupby(["surface", "arm"]).agg(
        median_cumulative_regret=("cumulative_regret", "median"), stopped=("stopped", "sum")))
