"""Batch acquisition maths over joint posterior draws — pure functions, no model.

Every function here works on a draws matrix of shape `(S, N)`: row `s` is one
plausible surface (one coefficient draw pushed through the link), column `j`
one candidate point, entry the mean response mu there. Scores are Monte Carlo
averages over rows, not closed-form Gaussian formulas: under a log link mu's
posterior is lognormal-shaped, not normal. All rules maximize.

Expected improvement needs a bar to improve on: the best result so far. Best
observed y is the wrong bar — a run that came back high by luck sets it above
anything the model believes reachable, and EI stalls. Instead the bar is taken
row by row, from the same surface: `bar_s` is surface `s`'s highest mu at the
points already run. With two runs A, B and a candidate C::

    surface   mu(A)  mu(B)  bar = max(A, B)  mu(C)  improvement
       1       10      8         10           12        2
       2        9     11         11           10        0
       3       10      9         10           13        3

    EI(C) = mean(2, 0, 3)

Observed y never enters, so noise in the data cannot inflate the bar, and
uncertainty about how good the current best really is carries through. This
is "noisy expected improvement" (q-NEI).

These follow BoTorch's formulations, reimplemented in numpy over this repo's own
PyMC posterior draws rather than through torch: `greedy_q_nei` and
`continuous_greedy_q_nei` are `qNoisyExpectedImprovement` under
`optimize_acqf(..., sequential=True)`, and `greedy_q_ucb` is its `qUpperConfidenceBound`.

    Balandat, Karrer, Jiang, Daulton, Letham, Wilson and Bakshy. "BoTorch: A
    Framework for Efficient Monte-Carlo Bayesian Optimization." NeurIPS 2020.
    Wilson, Hutter and Deisenroth. "Maximizing Acquisition Functions for
    Bayesian Optimization." NeurIPS 2018 — Monte Carlo acquisition over fixed
    draws (the sample-average approximation), and the submodular greedy
    argument for batches.

Not ported: BoTorch's `qLogNEI` smooths both the `max(0, .)` and the max over
batch members so a joint batch can be optimized at once. Greedy selection
optimizes one point at a time, so only the `max(0, .)` kink remains, and
averaging over draws already leaves it differentiable enough for L-BFGS-B.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable

import numpy as np

from malt.engine.search import maximize

# L-BFGS-B restarts per greedy pick, from the best candidates for that pick's
# bar. The starts are already near-optimal, so a handful captures the gain;
# each one costs a run of the optimizer.
_N_STARTS = 4

__all__ = [
    "central_composite",
    "continuous_greedy_q_nei",
    "greedy_q_nei",
    "greedy_q_ucb",
    "q_expected_improvement",
    "thompson_batch",
]


def q_expected_improvement(batch_draws: np.ndarray, incumbent: np.ndarray) -> float:
    """Monte Carlo q-EI of one batch: mean over rows of `max(0, max_j f_sj - incumbent_s)`.

    `batch_draws` is `(S, q)`, the batch's columns of the draws matrix;
    `incumbent` is `(S,)`, the per-surface bar.
    """
    return float(np.maximum(batch_draws.max(axis=1) - incumbent, 0.0).mean())


def greedy_q_nei(candidate_draws: np.ndarray, incumbent: np.ndarray, q: int) -> np.ndarray:
    """Choose `q` distinct candidates, one at a time, each maximizing the batch's q-EI.

    After each pick the bar rises to include it — `bar_s = max(incumbent_s,
    f_s(picked))` — so the next pick is rewarded only for surfaces where it
    beats everything already in the batch. That is what spreads the batch over
    rival hypotheses instead of stacking it on the single best-looking point.

    Returns column indices into `candidate_draws`, in pick order. See
    `_greedy_batch` for replacement and the fallback once nothing improves.

    No campaign rule calls this any more: `continuous_greedy_q_nei` starts from
    the same pick and then optimizes it. This stays public as that rule's
    reference point — the floor its tests assert against and the arm
    `benchmarks/polish_tradeoff.py` measures — and as the cheaper choice when
    only draws at fixed candidates are available.
    """
    return _greedy_batch(candidate_draws, np.asarray(incumbent, dtype=float), q)


def greedy_q_ucb(candidate_draws: np.ndarray, q: int, beta: float) -> np.ndarray:
    """Choose `q` distinct candidates, one at a time, each maximizing the batch's q-UCB.

    Monte Carlo q-UCB (Wilson et al., 2018) scores a batch as the mean over
    surfaces of `max_j u_sj`, where `u_sj = m_j + sqrt(beta * pi / 2) * |f_sj - m_j|`
    and `m_j` is the posterior mean. For a single Gaussian point that is
    `m + sqrt(beta) * sd`, the familiar UCB. `beta` trades exploitation
    (`beta = 0` is top-`q` by posterior mean) against exploration.

    Greedy selection spreads the batch the same way as `greedy_q_nei`.
    Returns column indices into `candidate_draws`, in pick order.
    """
    if beta < 0.0:
        raise ValueError(f"beta must be >= 0, got {beta}")
    mean = candidate_draws.mean(axis=0)
    upper = mean + np.sqrt(beta * np.pi / 2) * np.abs(candidate_draws - mean)
    # The floor is below every value, so the first pick is the plain argmax of mean u.
    return _greedy_batch(upper, upper.min(axis=1), q)


def _greedy_batch(values: np.ndarray, floor: np.ndarray, q: int) -> np.ndarray:
    """Greedily maximize `mean_s max(floor_s, max_{j in batch} values_sj)` over batches of `q`.

    The objective is monotone submodular, so the greedy batch scores within
    `1 - 1/e` of the best possible. Picks are without replacement: a repeat
    adds nothing, since values are of noise-free mu. Once no remaining
    candidate raises the objective — always the case after the first pick when
    `S == 1` — the rest are filled by highest mean value.
    """
    n_draws, n_candidates = values.shape
    if not 1 <= q <= n_candidates:
        raise ValueError(f"cannot choose {q} of {n_candidates} candidates")
    bar = floor.copy()
    mean = values.mean(axis=0)
    available = np.ones(n_candidates, dtype=bool)
    chosen = []
    for _ in range(q):
        gain = np.maximum(values - bar[:, None], 0.0).mean(axis=0)
        gain[~available] = -np.inf
        j = int(np.argmax(gain))
        if gain[j] <= 0.0:
            j = int(np.argmax(np.where(available, mean, -np.inf)))
        chosen.append(j)
        available[j] = False
        bar = np.maximum(bar, values[:, j])
    return np.array(chosen)


def continuous_greedy_q_nei(
    mu: Callable[[np.ndarray], np.ndarray],
    d_mu: Callable[[np.ndarray], np.ndarray],
    candidates: np.ndarray,
    incumbent: np.ndarray,
    q: int,
) -> np.ndarray:
    """Choose `q` points anywhere in `[-1, +1]^k` by greedy q-NEI: `(q, k)` coded.

    `greedy_q_nei` returns column indices, so its picks can only ever be
    candidates someone enumerated, and its accuracy is bounded by the pool's
    per-axis resolution — `len(candidates) ** (1 / k)`, which falls off a cliff
    as factors are added: 4096 points give 64 levels over 2 factors but 2.8 over
    8. This one starts from the same pool and then optimizes each pick with
    L-BFGS-B on the model's derivative, so a pick lands where the acquisition is
    actually highest rather than at the nearest candidate.

    Each pick starts from exactly the candidate `greedy_q_nei` would have taken
    at that step, and L-BFGS-B only moves off a start to improve on it. **For
    `q == 1` that makes it provably never worse.** For a batch it does not
    quite: a better first pick raises the bar, so later steps face a different
    objective, and greedy maximization of a submodular objective carries only a
    `1 - 1/e` guarantee — a better prefix can in principle end worse. Measured
    across both regret benchmarks and both surfaces it never does, but treat
    that as evidence rather than a theorem.

    This is the algorithm BoTorch implements as `optimize_acqf(...,
    sequential=True)` over `qNoisyExpectedImprovement`, in numpy over this
    repo's own posterior draws (Balandat et al., "BoTorch: A Framework for
    Efficient Monte-Carlo Bayesian Optimization", NeurIPS 2020). The greedy
    structure and its `1 - 1/e` guarantee, and the fixed-draws argument below,
    are from Wilson et al., "Maximizing Acquisition Functions for Bayesian
    Optimization", NeurIPS 2018 — the same paper `greedy_q_ucb` follows. One
    deliberate difference: BoTorch re-draws starting points for every pick,
    while this reuses the one scan of `candidates`, because the draws at those
    points do not change when the bar rises, only the score computed from them,
    which is arithmetic on a cached matrix.

    `mu(z)` maps coded points `(m, k)` to draws of mu `(S, m)`, and `d_mu(z)` to
    `d mu/dz` `(S, m, k)`. Both must be **deterministic** functions of `z` over
    the whole call: one fixed set of posterior draws throughout, never
    resampled. That is the sample-average approximation — the optimizer is
    descending a fixed surface, and with fresh draws per evaluation it would be
    chasing a moving target and never converge. Draws are of mu, the latent
    mean, never of observed `y`; see the module docstring for why the bar is
    taken row by row from the same surface.

    Once no point improves on the bar — always the case after the first pick
    when `S == 1` — the rest are filled from the pool by highest mean mu, as in
    `_greedy_batch`.
    """
    bar = np.asarray(incumbent, dtype=float).copy()
    if bar.ndim != 1:
        raise ValueError(f"incumbent must be one value per draw, got shape {bar.shape}")
    candidates = np.atleast_2d(candidates)
    n_candidates, k = candidates.shape
    if not 1 <= q <= n_candidates:
        raise ValueError(f"cannot choose {q} of {n_candidates} candidates")

    # One evaluation of the pool, reused by every pick: mu at a candidate does
    # not depend on the bar, so raising the bar is arithmetic on this matrix
    # rather than a re-scan. This is what keeps the cost close to `greedy_q_nei`.
    candidate_draws = mu(candidates)
    mean_mu = candidate_draws.mean(axis=0)
    available = np.ones(n_candidates, dtype=bool)

    picks = []
    for _ in range(q):
        gain = np.maximum(candidate_draws - bar[:, None], 0.0).mean(axis=0)
        gain[~available] = -np.inf
        starts = np.argsort(gain)[::-1][:_N_STARTS]
        starts = starts[gain[starts] > 0.0]
        if len(starts) == 0:
            # Nothing improves: fall back to the best unused candidate by mean mu.
            j = int(np.argmax(np.where(available, mean_mu, -np.inf)))
            available[j] = False
            picks.append(candidates[j])
            bar = np.maximum(bar, candidate_draws[:, j])
            continue

        available[starts[0]] = False
        value, gradient = _improvement_over(mu, d_mu, bar)
        z_best, _ = maximize(value, k, grad=gradient, starts=candidates[starts])
        picks.append(z_best)
        bar = np.maximum(bar, mu(z_best[None, :])[:, 0])
    return np.array(picks)


def _improvement_over(
    mu: Callable[[np.ndarray], np.ndarray],
    d_mu: Callable[[np.ndarray], np.ndarray],
    bar: np.ndarray,
) -> tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    """`mean_s max(0, mu_s(z) - bar_s)` and its gradient, as a factory over `bar`.

    A factory so each pick closes over that pick's bar rather than over a
    variable the greedy loop rebinds. The value is `q_expected_improvement` of a
    one-point batch, evaluated per candidate column. The gradient is the mean of
    `d mu_s/dz` over the draws that currently beat the bar: continuous and
    piecewise smooth, with a draw crossing the bar moving it by order `1 / S`.
    Sequential greedy optimizes one `z` at a time, so the non-differentiable max
    over batch members never enters and no smoothing is needed — unlike a joint
    formulation, which is why BoTorch's `qLogNEI` softens both.
    """

    def value(z: np.ndarray) -> np.ndarray:
        return np.maximum(mu(z) - bar[:, None], 0.0).mean(axis=0)

    def gradient(z: np.ndarray) -> np.ndarray:
        improving = mu(z) > bar[:, None]
        return (d_mu(z) * improving[:, :, None]).mean(axis=0)

    return value, gradient


def thompson_batch(candidate_draws: np.ndarray, q: int, rng: np.random.Generator) -> np.ndarray:
    """Batch Thompson sampling: the argmax of each of `q` randomly chosen surfaces.

    Surfaces are chosen without replacement when there are enough, so a
    single-draw model still works (every pick is its argmax). Two surfaces can
    share an argmax, so the batch may repeat a point — replication where the
    posterior is confident.

    Returns column indices into `candidate_draws`.
    """
    n_draws = candidate_draws.shape[0]
    rows = rng.choice(n_draws, size=q, replace=q > n_draws)
    return candidate_draws[rows].argmax(axis=1)


def central_composite(
    center: np.ndarray, radius: float, alpha: float = 1.0, n_center: int = 0
) -> np.ndarray:
    """A central composite design around `center`, in coded units: `(2^k + 2k + n_center, k)`.

    Factorial corners at `center ± radius` in every factor, plus axial points
    at `± alpha * radius` along each axis. `alpha = 1` is face-centred: three
    levels per factor. The last `n_center` rows are centre replicates.

    If `center` is too close to the boundary, it is moved inward until the
    whole design fits in `[-1, +1]`. The design keeps its shape rather than
    having points clipped onto a face, which would duplicate runs.
    """
    center = np.asarray(center, dtype=float)
    reach = alpha * radius
    extent = max(radius, reach)  # half-width of the design along any axis
    if radius <= 0.0 or alpha <= 0.0 or extent > 1.0:
        raise ValueError(
            f"need radius > 0, alpha > 0 and max(radius, alpha * radius) <= 1, "
            f"got radius={radius}, alpha={alpha}"
        )
    k = len(center)
    corners = radius * np.array(list(itertools.product((-1.0, 1.0), repeat=k)))
    axial = reach * np.vstack([np.eye(k), -np.eye(k)])
    runs = np.vstack([corners, axial, np.zeros((n_center, k))])
    return np.clip(center, -1.0 + extent, 1.0 - extent) + runs
