"""What optimizing q-NEI's picks past the candidate pool buys, and what it costs.

    uv run python -m benchmarks.polish_tradeoff              # quality and search cost vs factors
    uv run python -m benchmarks.polish_tradeoff --batch 16   # the cost driver is the batch size
    uv run python -m benchmarks.polish_tradeoff --campaign   # cost as a share of a real round
    uv run python -m benchmarks.polish_tradeoff --convex     # how to measure this wrong

The other benchmarks score whole campaigns by regret. This one isolates one
step: given the same posterior and the same candidate pool, how much better a
batch does `continuous_greedy_q_nei` find than `greedy_q_nei`, which can only
return points someone already drew — and what the difference costs.

It exists because the regret benchmarks cannot answer either question. They run
2 and 3 factors, where a pool of 4096 Sobol points gives 64 and 16 levels per
axis — enough that the discrete pick is already sitting on the acquisition
optimum, so polishing has nothing to add. The pool's resolution is
`n_candidates ** (1 / k)`, which falls off a cliff: 8 levels per axis at 4
factors, 2.8 at 8. Media optimization with six or eight components is the case
this repo is aimed at, and that is where the choice starts to matter.

**On reading the cost columns.** Both arms evaluate the pool exactly once —
raising the bar is arithmetic on the cached `(S, N)` matrix, not a re-scan — so
the difference is the L-BFGS-B steps, which evaluate one row at a time and are
dominated by per-call overhead rather than by draws. That makes the wall-clock
ratios here *pessimistic* for a real campaign: the synthetic posterior is a bare
matmul over 512 draws, so the discrete baseline is milliseconds and the fixed
overhead of the polish dominates the ratio. In a campaign the surrogate is
thinned for the search (`SurrogateModel.thinned`) and the round also carries an
MCMC fit; `--campaign` gives that figure, where the cost is a wash.

The default mode runs no PyMC, so `k` can go past anything a campaign benchmark
could afford: the posteriors are synthetic, drawn around a concave quadratic
with a random tilt, the spread standing in for epistemic uncertainty. That is a
weaker claim than a regret number — a better acquisition value is necessary for
lower regret, not sufficient — but it is the quantity polishing directly
maximizes, so it is where an effect must show up first.

Two ways to measure this wrong, both of which this script guards against:

- **Convex draws measure something else.** The quadratic prior
  `Normal(-0.5, 0.5)` leaves about a sixth of each coefficient's mass positive,
  and a convex draw's optimum is a corner of the box. Sobol cannot reach a
  corner in high dimensions, so the polish walks to one and reports an enormous
  gain — real behaviour, but it is measuring corner-seeking, not search
  accuracy. `--convex` shows this; the default forces every draw concave, so
  every surface has an interior peak. Watch the `mean |z|` column.
- **A pool that differs between the two.** Both arms are handed the same
  `n_candidates`, and the polish gets no larger raw scan than the discrete rule
  gets candidates.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable

import numpy as np

from malt.active_learning.acquisitions import QNoisyExpectedImprovement
from malt.active_learning.surrogates import GammaGLMSurrogate
from malt.engine.acquisition import (
    central_composite,
    continuous_greedy_q_nei,
    greedy_q_nei,
    q_expected_improvement,
)
from malt.engine.factors import Factor, decode_design
from malt.engine.glm import build_design_matrix, design_jacobian, predict_mu, predict_mu_grad
from malt.engine.search import space_filling
from malt.simulation.oracle import GammaLikelihood, Oracle, quadratic_latent

# Coefficient means by term kind, on the coded scale: a peaked surface with a
# mild tilt and weak interactions, i.e. what the GLM's priors expect.
CENTRE = {"linear": 0.6, "quadratic": -0.8, "interaction": 0.2}

# The two regret benchmarks' factors and batch sizes, so --campaign reports the
# cost of the configuration that is actually run.
CAMPAIGNS = (
    ((Factor("glucose", 0.5, 20.0, units="g/L"),
      Factor("nitrogen", 0.1, 10.0, units="g/L", scale="log")), 10),
    ((Factor("glucose", 0.5, 20.0, units="g/L"),
      Factor("nitrogen", 0.1, 10.0, units="g/L", scale="log"),
      Factor("phosphate", 0.05, 2.0, units="g/L", scale="log")), 16),
)


def synthetic_factors(k: int) -> tuple[Factor, ...]:
    """`k` plain linear factors over 0-10: the design space the sweep searches."""
    return tuple(Factor(f"x{j}", 0.0, 10.0) for j in range(k))


def quadratic_posterior(
    k: int, draws: int, seed: int, *, spread: float = 0.35, concave: bool = True
) -> tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    """Synthetic draws over quadratic surfaces in `k` factors: `(mu, d_mu)` of coded points.

    Both are deterministic functions of `z` built on one frozen set of draws,
    which is what `continuous_greedy_q_nei` requires. `concave` forces every
    draw's squared terms negative and shrinks its interactions, so each surface
    has an interior peak; see the module docstring for why that matters.
    """
    fs = synthetic_factors(k)
    rng = np.random.default_rng(seed)
    kinds = build_design_matrix(decode_design(fs, np.zeros((1, k))), fs).term_kinds

    beta = np.array([CENTRE[kind] for kind in kinds]) + rng.normal(0.0, spread, (draws, len(kinds)))
    if concave:
        squared = [i for i, kind in enumerate(kinds) if kind == "quadratic"]
        interactions = [i for i, kind in enumerate(kinds) if kind == "interaction"]
        beta[:, squared] = -np.abs(beta[:, squared]) - 0.5
        beta[:, interactions] *= 0.15
    intercept = rng.normal(0.0, 0.3, draws)

    def mu(z: np.ndarray) -> np.ndarray:
        x = decode_design(fs, np.atleast_2d(z))
        return predict_mu(intercept, beta, build_design_matrix(x, fs).X)

    def d_mu(z: np.ndarray) -> np.ndarray:
        # Recomputes mu rather than sharing it with `mu` above, which is what
        # the real surrogate does too: `sample` and `sample_jacobian` are
        # separate calls. Keeping that here keeps the cost columns honest.
        z = np.atleast_2d(z)
        x = decode_design(fs, z)
        value = predict_mu(intercept, beta, build_design_matrix(x, fs).X)
        return predict_mu_grad(value, beta, design_jacobian(z, fs))

    return mu, d_mu


def compare(
    k: int,
    *,
    q: int = 8,
    draws: int = 512,
    n_candidates: int = 4096,
    seed: int = 0,
    concave: bool = True,
) -> dict[str, float]:
    """Polished against discrete q-NEI in `k` factors: what it buys and what it costs.

    Both arms get the same posterior, the same pool size and the same draws, so
    the only difference is whether picks are taken from the pool or optimized
    from it.

    Quality is `gain`, the relative improvement in the q-NEI a batch achieves.
    Cost is reported two ways, because wall clock alone does not explain itself:
    `seconds` is what a caller waits, and `mu_evaluations` counts rows of mu
    computed (rows times draws), which is the work that scales. The split
    matters — polishing re-scans the pool once per greedy pick, so its scan work
    is `q` times the discrete rule's, while its L-BFGS-B steps evaluate one row
    at a time and are dominated by per-call overhead rather than draws.

    `mean_abs_z` is a tell for corner-seeking (see the module docstring): near
    0.3 is an interior batch, near 1 is on the boundary.
    """
    mu_raw, d_mu_raw = quadratic_posterior(k, draws, seed + k, concave=concave)
    counts = {"mu_rows": 0, "d_mu_rows": 0, "mu_calls": 0, "d_mu_calls": 0}

    def mu(z: np.ndarray) -> np.ndarray:
        counts["mu_calls"] += 1
        counts["mu_rows"] += len(np.atleast_2d(z))
        return mu_raw(z)

    def d_mu(z: np.ndarray) -> np.ndarray:
        counts["d_mu_calls"] += 1
        counts["d_mu_rows"] += len(np.atleast_2d(z))
        return d_mu_raw(z)

    pool = space_filling(k, n_candidates, np.random.default_rng(seed))
    # An incumbent the pool can already beat somewhere, so neither arm starts
    # from a bar so high that every improvement is zero.
    incumbent = np.quantile(mu_raw(pool), 0.9, axis=1)

    mu_raw(pool[:2])  # warm up numpy before timing
    d_mu_raw(pool[:2])

    start = time.perf_counter()
    discrete = pool[greedy_q_nei(mu(pool), incumbent, q)]
    discrete_seconds = time.perf_counter() - start
    discrete_rows = counts["mu_rows"]

    counts.update({key: 0 for key in counts})
    start = time.perf_counter()
    polished = continuous_greedy_q_nei(mu, d_mu, pool, incumbent, q)
    polished_seconds = time.perf_counter() - start

    before = q_expected_improvement(mu_raw(discrete), incumbent)
    after = q_expected_improvement(mu_raw(polished), incumbent)
    return {
        "k": k,
        "q": q,
        "levels_per_axis": n_candidates ** (1 / k),
        "discrete": before,
        "polished": after,
        "gain": (after - before) / before,
        "mean_abs_z": float(np.abs(polished).mean()),
        "discrete_seconds": discrete_seconds,
        "polished_seconds": polished_seconds,
        "cost_ratio": polished_seconds / discrete_seconds,
        "discrete_mu_evaluations": discrete_rows * draws,
        "polished_mu_evaluations": (counts["mu_rows"] + counts["d_mu_rows"]) * draws,
        "polished_point_calls": counts["mu_calls"] + counts["d_mu_calls"],
    }



def campaign_cost(
    factors: tuple[Factor, ...], q: int, *, repeats: int = 3, seed: int = 7
) -> dict[str, float]:
    """Cost of one `propose` against a fitted Gamma GLM, beside the MCMC fit it shares a round with.

    This is the figure that decides anything. A campaign round is a fit plus a
    proposal, and the fit is seconds of NUTS, so a ratio measured on `propose`
    alone overstates what a campaign actually feels. One fit, reused for every
    timed proposal, and the proposals are seeded so the run is reproducible.
    """
    k = len(factors)
    rng = np.random.default_rng(seed)
    latent = quadratic_latent(
        factors, optimum={f.name: f.center * 1.4 for f in factors}, peak=2.0, curvature=1.5
    )
    oracle = Oracle(latent, GammaLikelihood(alpha=20.0))
    seed_design = decode_design(factors, central_composite(np.full(k, -0.4), 0.4, n_center=3))
    seed_data = seed_design.join(oracle.query(seed_design, rng).set_axis(seed_design.index))

    start = time.perf_counter()
    model = GammaGLMSurrogate(factors).condition(seed_data, np.random.default_rng(1))
    fit_seconds = time.perf_counter() - start

    rule = QNoisyExpectedImprovement(factors)
    rule.propose(model, q, np.random.default_rng(3))  # warm up
    times = []
    for r in range(repeats):
        start = time.perf_counter()
        rule.propose(model, q, np.random.default_rng(3 + r))
        times.append(time.perf_counter() - start)

    propose_seconds = float(np.median(times))
    return {
        "k": k,
        "q": q,
        "fit_seconds": fit_seconds,
        "propose_seconds": propose_seconds,
        "round_seconds": fit_seconds + propose_seconds,
        "propose_share": propose_seconds / (fit_seconds + propose_seconds),
    }


def report_campaign(configs: tuple[tuple[tuple[Factor, ...], int], ...]) -> None:
    """Where a round's time actually goes: NUTS, or choosing the next batch."""
    print("One round against a fitted Gamma GLM (4 chains x 2000 draws), median of 3 proposals.")
    print("A round is one fit plus one proposal, so `round s` is what a campaign feels.\n")
    print(f"{'factors':>7} {'batch':>5} | {'fit s':>7} {'propose s':>10} {'round s':>8} {'propose%':>9}")
    for factors, q in configs:
        row = campaign_cost(factors, q)
        print(
            f"{row['k']:>7.0f} {row['q']:>5.0f} | {row['fit_seconds']:>7.2f}"
            f" {row['propose_seconds']:>10.3f} {row['round_seconds']:>8.2f}"
            f" {100 * row['propose_share']:>8.1f}%"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factors", type=int, nargs="+", default=[2, 3, 4, 5, 6, 8])
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--draws", type=int, default=512)
    parser.add_argument("--candidates", type=int, default=4096, help="a power of 2")
    parser.add_argument("--replicates", type=int, default=3, help="posteriors per factor count")
    parser.add_argument(
        "--convex",
        action="store_true",
        help="allow convex draws, whose optima are box corners; see the module docstring",
    )
    parser.add_argument(
        "--campaign",
        action="store_true",
        help="cost against a fitted Gamma GLM instead of a synthetic posterior; needs PyMC",
    )
    args = parser.parse_args()

    if args.campaign:
        report_campaign(CAMPAIGNS)
        return

    print(f"batch of {args.batch}, {args.draws} draws, {args.candidates} candidates, "
          f"median of {args.replicates} posteriors")
    print("draws are " + ("unconstrained (some convex)" if args.convex else "strictly concave"))
    print("\nQuality, and what it costs:\n")
    print(f"{'factors':>7} {'terms':>6} {'levels/axis':>12} {'gain':>8} {'mean |z|':>9}"
          f" | {'discrete s':>10} {'polished s':>10} {'x cost':>7}"
          f" | {'disc. Mmu':>10} {'pol. Mmu':>9} {'1-row calls':>12}")
    for k in args.factors:
        rows = [
            compare(k, q=args.batch, draws=args.draws, n_candidates=args.candidates,
                    seed=100 * r, concave=not args.convex)
            for r in range(args.replicates)
        ]
        mid = {key: float(np.median([r[key] for r in rows])) for key in rows[0]}
        n_terms = 2 * k + k * (k - 1) // 2
        print(
            f"{k:>7} {n_terms:>6} {mid['levels_per_axis']:>12.1f} "
            f"{100 * mid['gain']:>+7.1f}% {mid['mean_abs_z']:>9.3f}"
            f" | {mid['discrete_seconds']:>10.3f} {mid['polished_seconds']:>10.3f} "
            f"{mid['cost_ratio']:>6.1f}x"
            f" | {mid['discrete_mu_evaluations'] / 1e6:>10.1f} "
            f"{mid['polished_mu_evaluations'] / 1e6:>9.1f} {mid['polished_point_calls']:>12.0f}"
        )


if __name__ == "__main__":
    main()
