"""Round-by-round posterior maps for one 2-factor campaign.

    uv run python -m benchmarks.two_factor.rounds --arm qnei --surface gp

Arms: `qnei` (Gamma GLM + q-NEI) or `rsm` (quadratic BLR + a CCD at the
posterior-mean optimum, i.e. iterative RSM).

Every campaign starts from the classic seed: a face-centred CCD over the full
ranges with 3 centre runs (11 runs).

One row per round, 0 being the fit to the seed. Left: the posterior mean of
biomass, beside the true mean surface on the same scale. Right: posterior
uncertainty. For the Gamma GLM that is the sd of
log biomass — roughly its coefficient of variation, and a monotone transform
of the entropy of the (near-Gaussian) linear predictor, so it ranks points
exactly as an entropy map would; raw variance of mu would mostly track the
mean under a log link. BLR's Gaussian draws can be zero or negative, so for
it the column is the sd of biomass itself, in g/L, and a dashed line marks
where its mean crosses zero. Both columns share one colour scale down the
rows, so rounds compare directly. Dots are the runs observed so far; rings
are the batch that round's model proposes next; the star is the true optimum.

Writes `results/rounds_<arm>_<surface>.png`.
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
from matplotlib import patheffects  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from malt.active_learning.acquisitions import CentralComposite, QNoisyExpectedImprovement  # noqa: E402
from malt.active_learning.actors import Experimenter, SurrogateModel  # noqa: E402
from malt.active_learning.campaign import LabJournal, run_campaign  # noqa: E402
from malt.active_learning.surrogates import BayesianLinearRegression, GammaGLMSurrogate  # noqa: E402
from malt.active_learning.termination import MaxRoundsRule, UnreliableFitRule  # noqa: E402
from malt.engine.acquisition import central_composite  # noqa: E402
from malt.engine.factors import Factor, candidate_grid, decode_design  # noqa: E402
from malt.engine.glm import ConvergenceWarning  # noqa: E402
from malt.simulation.regret import true_optimum  # noqa: E402
from malt.simulation.oracle import (  # noqa: E402
    GammaLikelihood,
    LatentFunction,
    Oracle,
    gp_sampled_latent,
    quadratic_latent,
)

RESULTS = Path(__file__).parent / "results"
SEED = 7

GLUCOSE = Factor("glucose", 0.5, 20.0, units="g/L")
NITROGEN = Factor("nitrogen", 0.1, 10.0, units="g/L", scale="log")
FACTORS = (GLUCOSE, NITROGEN)
PIXELS = candidate_grid(FACTORS, 61)  # only for drawing the maps; no rule searches a grid
BATCH = 8  # a 2-factor CCD is exactly 8 runs
ROUNDS = 4


def quadratic_surface(rng: np.random.Generator) -> LatentFunction:
    coded = rng.uniform(-0.6, 0.6, len(FACTORS))
    optimum = {f.name: float(f.decode(coded[j])) for j, f in enumerate(FACTORS)}
    return quadratic_latent(FACTORS, optimum=optimum, peak=np.log(10.0), curvature=1.0)


def gp_surface(rng: np.random.Generator) -> LatentFunction:
    return gp_sampled_latent(FACTORS, rng, lengthscale=0.6, sd=0.5, mean=np.log(5.0))


SURFACES = {"quadratic": quadratic_surface, "gp": gp_surface}


def seed_design() -> pd.DataFrame:
    """The classic starting point: a face-centred CCD over the full ranges, with 3 centre runs."""
    return decode_design(FACTORS, central_composite(np.zeros(len(FACTORS)), radius=1.0, n_center=3))
TITLES = {"quadratic": "quadratic", "gp": "GP-sampled"}

# name -> (title, experimenter factory, whether its draws are positive so log sd applies)
ARMS = {
    "qnei": (
        "Gamma GLM + q-NEI",
        lambda: Experimenter(GammaGLMSurrogate(FACTORS), QNoisyExpectedImprovement(FACTORS)),
        True,
    ),
    "rsm": (
        "BLR + CCD (iterative RSM)",
        lambda: Experimenter(BayesianLinearRegression(FACTORS, "quadratic"), CentralComposite(FACTORS)),
        False,
    ),
}

# Two sequential scales on one figure: blue for the mean, orange for the
# uncertainty (the dataviz reference's first two hues), each light to dark.
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#eef4fc", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)
ORANGES = LinearSegmentedColormap.from_list(
    "oranges", ["#fdf1ea", "#fbd9c6", "#f6b391", "#f08c5e", "#eb6834", "#c94f1f", "#9c3a14", "#6e280d"]
)
# The truth is drawn in greys, apart from the model's colours.
GREYS = LinearSegmentedColormap.from_list(
    "greys", ["#f4f4f2", "#d9d9d6", "#b3b3af", "#8a8a86", "#62625f", "#3d3d3b", "#1f1f1e", "#0b0b0b"]
)
SURFACE_BG, INK, INK_MUTED = "#fcfcfb", "#1f1f1e", "#6b6a66"


def maps(model: SurrogateModel, log_sd: bool) -> tuple[np.ndarray, np.ndarray]:
    """Posterior mean of mu and sd of (log) mu at every pixel, as 2-D arrays (nitrogen x glucose)."""
    draws = model.sample(PIXELS)
    shape = (61, 61)  # candidate_grid varies the last factor fastest
    sd = (np.log(draws) if log_sd else draws).std(axis=0)
    return draws.mean(axis=0).reshape(shape).T, sd.reshape(shape).T


def log_levels(lo: float, hi: float) -> list[float]:
    """1-2-5 values strictly between `lo` and `hi`: contour levels and ticks for a log scale."""
    return [m * 10.0**e for e in range(-3, 4) for m in (1, 2, 5) if lo < m * 10.0**e < hi]


def plot(journal: LabJournal, oracle: Oracle, arm: str, surface: str, out: Path) -> None:
    title, _, log_sd = ARMS[arm]
    models = [journal.seed.surrogate_model] + [r.surrogate_model for r in journal.rounds]
    seen = [journal.seed.data] + [r.data for r in journal.rounds]
    fields = [maps(m, log_sd) for m in models]

    truth = oracle.mean(PIXELS)
    best = true_optimum(oracle, FACTORS)[0].iloc[0]
    true_field = truth.reshape(61, 61).T
    # Early rounds extrapolate wildly at the corners; cap the mean scale so the
    # peak region stays readable, and show the uncertainty on a log scale so
    # its shrinkage over rounds is visible end to end.
    # The truth shares the posterior mean's scale so the two compare directly.
    mean_top = max([np.percentile(mean, 99) for mean, _ in fields] + [truth.max()])
    mean_norm = Normalize(0.0, min(mean_top, 1.5 * truth.max()))
    sd_all = np.concatenate([sd.ravel() for _, sd in fields])
    sd_lo, sd_hi = float(np.percentile(sd_all, 1)), float(np.percentile(sd_all, 99))
    sd_norm = LogNorm(sd_lo, sd_hi)
    # Fixed contour levels per column, so a level can be followed down the rows.
    mean_levels = MaxNLocator(6).tick_values(0.0, mean_norm.vmax or 1.0)[1:-1]
    sd_levels = log_levels(sd_lo, sd_hi)

    g = np.unique(PIXELS["glucose"].to_numpy())
    n = np.unique(PIXELS["nitrogen"].to_numpy())
    rows = len(models)
    fig, axes = plt.subplots(rows, 3, figsize=(12.6, 2.9 * rows + 1.2), sharex=True, sharey=True,
                             facecolor=SURFACE_BG, squeeze=False)
    for r, ((mean, sd), ax_row) in enumerate(zip(fields, axes)):
        observed = pd.concat(seen[: r + 1])
        proposed = seen[r + 1] if r + 1 < len(seen) else None
        for ax, field, cmap, norm, levels in (
            (ax_row[0], true_field, GREYS, mean_norm, mean_levels),
            (ax_row[1], mean, BLUES, mean_norm, mean_levels),
            (ax_row[2], sd, ORANGES, sd_norm, sd_levels),
        ):
            ax.pcolormesh(g, n, field, cmap=cmap, norm=norm, shading="nearest", rasterized=True)
            lines = ax.contour(g, n, field, levels=levels, colors=INK, linewidths=0.7, alpha=0.55)
            labels = ax.clabel(lines, fontsize=7, fmt="%g", inline_spacing=2)
            if cmap is GREYS:  # dark lines vanish on the black peak: outline them in white
                halo = [patheffects.withStroke(linewidth=1.4, foreground="white")]
                lines.set_path_effects(halo)
                for label in labels:
                    label.set_path_effects(halo)
            if cmap is BLUES and field.min() < 0:
                ax.contour(g, n, field, levels=[0.0], colors=INK, linewidths=1.0, linestyles="--")
            ax.scatter(observed["glucose"], observed["nitrogen"], s=14, color=INK, edgecolors="white",
                       linewidths=0.6, zorder=3, clip_on=False)
            if proposed is not None:
                ax.scatter(proposed["glucose"], proposed["nitrogen"], s=70, facecolors="none",
                           edgecolors="white", linewidths=2.6, zorder=4, clip_on=False)
                ax.scatter(proposed["glucose"], proposed["nitrogen"], s=70, facecolors="none",
                           edgecolors=INK, linewidths=1.3, zorder=5, clip_on=False)
            ax.scatter([best["glucose"]], [best["nitrogen"]], marker="*", s=160, color="white",
                       edgecolors=INK, linewidths=1.0, zorder=6, clip_on=False)
            ax.set_yscale("log")
            ax.tick_params(colors=INK_MUTED, labelsize=8)
            for spine in ax.spines.values():
                spine.set_visible(False)
        ax_row[0].set_ylabel(f"Round {r}\n{len(observed)} runs\n\nnitrogen (g/L)", color=INK, fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel("glucose (g/L)", color=INK_MUTED, fontsize=9)
    axes[0][0].set_title("True mean biomass (g/L)", loc="left", fontsize=10, color=INK)
    axes[0][1].set_title("Posterior mean biomass (g/L)", loc="left", fontsize=10, color=INK)
    axes[0][2].set_title("Posterior sd of log biomass (≈ CV)" if log_sd else "Posterior sd of biomass (g/L)",
                         loc="left", fontsize=10, color=INK)

    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    for col, norm, cmap, extend in ((0, mean_norm, GREYS, "max"), (1, mean_norm, BLUES, "max"),
                                    (2, sd_norm, ORANGES, "both")):
        box = axes[-1][col].get_position()
        cax = fig.add_axes((box.x0, 0.025, box.width, 0.012))
        bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal",
                           extend=extend)
        bar.outline.set_visible(False)
        if isinstance(norm, LogNorm):
            ticks = log_levels(sd_lo, sd_hi)
            bar.set_ticks(ticks, labels=[f"{t:g}" for t in ticks])
            bar.ax.minorticks_off()
        bar.ax.tick_params(colors=INK_MUTED, labelsize=8)
    fig.suptitle(
        f"{title} on the {TITLES[surface]} surface: dots observed, rings proposed next, "
        "star true optimum",
        fontsize=10, color=INK, x=0.02, ha="left",
    )
    fig.savefig(out, dpi=160, facecolor=SURFACE_BG)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", choices=list(ARMS), default="qnei")
    parser.add_argument("--surface", choices=list(SURFACES), default="gp")
    args = parser.parse_args()

    surface_ss, seed_ss, campaign_ss = np.random.SeedSequence([SEED, list(SURFACES).index(args.surface)]).spawn(3)
    oracle = Oracle(SURFACES[args.surface](np.random.default_rng(surface_ss)), GammaLikelihood(20.0))
    x = seed_design()
    seed_data = x.join(oracle.query(x, np.random.default_rng(seed_ss)).set_axis(x.index))

    warnings.simplefilter("ignore", ConvergenceWarning)
    journal = run_campaign(
        ARMS[args.arm][1](),
        oracle,
        seed_data,
        BATCH,
        random_seed=int(campaign_ss.generate_state(1)[0]),
        termination_rule=UnreliableFitRule() | MaxRoundsRule(ROUNDS),
    )
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"rounds_{args.arm}_{args.surface}.png"
    plot(journal, oracle, args.arm, args.surface, out)
    print(f"{len(journal.rounds)} rounds, stopped by {journal.stopped_by}; wrote {out}")


if __name__ == "__main__":
    main()
