"""Scoring experimenters against a simulated lab with a known ground truth.

The score is cumulative regret over the runs a campaign chose:

    R_T = sum over runs t of  1 - f*(x_t) / max f*

where `f*` is the oracle's true mean. Each run's term is the fraction of the
best achievable biomass it gave up, so `R_T` reads as "optimal runs' worth of
biomass lost". Only the acquisition's choices count: the seed is shared by
every arm and chosen by no algorithm, so it adds nothing. A good campaign has
sublinear regret — `R_T / T` falls toward 0 as it concentrates near the
optimum — while a rule that keeps sampling poor compositions grows linearly.

Because it scores runs, not a model's recommendation, a model-free rule has
the same expected regret whatever surrogate it is paired with. Regret is
measured on the true mean, never observed y, which would reward noise. The
maximum is found by continuous search, so no grid puts a floor under it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from malt.active_learning.actors import Acquisition, Experimenter, SurrogateModel
from malt.active_learning.campaign import run_campaign
from malt.active_learning.termination import MaxRoundsRule, UnreliableFitRule
from malt.engine.factors import Factor, decode_design
from malt.engine.search import maximize
from malt.simulation.oracle import Oracle

__all__ = ["Arm", "instantaneous_regret", "run_arm", "true_optimum"]


@dataclass(frozen=True, slots=True)
class Arm:
    """One experimenter under test: a prior surrogate and an acquisition rule."""

    name: str
    surrogate_model: SurrogateModel
    acquisition: Acquisition


def true_optimum(oracle: Oracle, factors: tuple[Factor, ...]) -> tuple[pd.DataFrame, float]:
    """Where the oracle's true mean peaks, and its value there: `(one-row frame, max mu)`."""
    x, best = maximize(lambda z: oracle.mean(decode_design(factors, z)), len(factors))
    return decode_design(factors, x), best


def instantaneous_regret(oracle: Oracle, x: pd.DataFrame, best: float) -> np.ndarray:
    """`1 - f*(x_t) / best` for every run in `x`; `best` from `true_optimum`."""
    return 1.0 - oracle.mean(x) / best


def run_arm(
    arm: Arm,
    oracle: Oracle,
    seed_data: pd.DataFrame,
    factors: tuple[Factor, ...],
    *,
    batch_size: int,
    rounds: int,
    random_seed: int,
) -> pd.DataFrame:
    """Run one campaign and score the runs it chose.

    Returns one row per round, `0` being the seed: `n_experiments` run so far
    (seed included), `batch_regret` (the round's summed instantaneous
    regret), `cumulative_regret` (`R_T` after the round; 0 at the seed), and
    `stopped`. A campaign stopped early by a failed fit ran no further
    rounds, so their regrets are NaN — nothing to score, rather than a
    guess — and `stopped` marks them.
    """
    journal = run_campaign(
        Experimenter(arm.surrogate_model, arm.acquisition),
        oracle,
        seed_data,
        batch_size,
        random_seed=random_seed,
        termination_rule=UnreliableFitRule() | MaxRoundsRule(rounds),
    )
    _, best = true_optimum(oracle, factors)
    batch = [0.0] + [float(instantaneous_regret(oracle, r.data, best).sum()) for r in journal.rounds]
    batch += [np.nan] * (rounds + 1 - len(batch))
    return pd.DataFrame(
        {
            "round": np.arange(rounds + 1),
            "n_experiments": len(seed_data) + batch_size * np.arange(rounds + 1),
            "batch_regret": batch,
            "cumulative_regret": np.cumsum(batch),
            "stopped": np.arange(rounds + 1) > len(journal.rounds),
        }
    )
