"""Searching the coded design space `[-1, +1]^k` without a grid.

A fixed grid puts a floor under every argmax: the best a search can do is the
nearest lattice point, and a recommendation that happens to land on the right
one scores as exactly optimal. These helpers work in continuous coded units
instead. Callers decode through their `Factor`s, so log-scale factors are
searched evenly in log space.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

__all__ = ["maximize", "space_filling"]


def space_filling(k: int, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """`n` Sobol points in `[-1, +1]^k`, shape `(n, k)`; `n` must be a power of 2.

    Scrambled from `rng` when one is given, so each call covers the space
    afresh; the plain, deterministic sequence when `rng` is None.
    """
    if n < 1 or n & (n - 1):
        raise ValueError(f"n must be a power of 2, got {n}")
    sobol = qmc.Sobol(d=k, scramble=rng is not None, rng=rng)
    return 2.0 * sobol.random_base2(int(np.log2(n))) - 1.0


def maximize(
    f: Callable[[np.ndarray], np.ndarray],
    k: int,
    rng: np.random.Generator | None = None,
    *,
    grad: Callable[[np.ndarray], np.ndarray] | None = None,
    starts: np.ndarray | None = None,
    n_raw: int = 2048,
    n_starts: int = 8,
) -> tuple[np.ndarray, float]:
    """Maximize `f` over `[-1, +1]^k`: `(x_best, f_best)`.

    `f` maps coded points `(m, k)` to values `(m,)`. It is evaluated on
    `n_raw` space-filling points, then L-BFGS-B polishes the best `n_starts`
    within the bounds. The result is never worse than the best raw point.
    Deterministic when `rng` is None.

    `grad` is `f`'s gradient, mapping `(m, k)` to `(m, k)`. Supplying it makes
    each polish step one call instead of the `k + 1` that scipy's
    finite-difference fallback needs, so the saving grows with the number of
    factors. The raw scan never needs it.

    `starts` `(m, k)` skips the scan and polishes from those rows instead, for a
    caller that has already evaluated `f` and knows where the promising points
    are — `continuous_greedy_q_nei` hands over the best of its candidate pool.
    The guarantee then reads "never worse than the best of `starts`", so such a
    caller keeps whatever its own candidates were worth. `rng`, `n_raw` and
    `n_starts` are ignored.
    """
    if starts is None:
        pool = space_filling(k, n_raw, rng)
    else:
        pool = np.atleast_2d(starts)
        n_starts = len(pool)
    values = f(pool)
    best = int(np.argmax(values))
    x_best, f_best = pool[best], float(values[best])
    jac = None if grad is None else (lambda z: -np.asarray(grad(z[None, :])[0], dtype=float))
    for start in pool[np.argsort(values)[::-1][:n_starts]]:
        result = minimize(
            lambda z: -float(f(z[None, :])[0]),
            start,
            method="L-BFGS-B",
            jac=jac,
            bounds=[(-1.0, 1.0)] * k,
        )
        if -result.fun > f_best:
            x_best, f_best = np.clip(result.x, -1.0, 1.0), -float(result.fun)
    return x_best, f_best
