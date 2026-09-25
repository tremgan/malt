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
"""

from __future__ import annotations

import itertools

import numpy as np

__all__ = [
    "central_composite",
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
