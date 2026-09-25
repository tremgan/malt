"""Scoring experimenters against a simulated lab with a known ground truth.

Every arm is scored the same way, whatever its acquisition rule: after each
round it recommends the candidate with the highest posterior mean, and regret
is measured on the oracle's true mean there — never on observed y, which
rewards noise. Regret is relative, `1 - mu(recommended) / mu(best candidate)`:
the fraction of the best achievable biomass given up, comparable across
surfaces whatever their scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from malt.active_learning.actors import Acquisition, Experimenter, SurrogateModel
from malt.active_learning.campaign import run_campaign
from malt.active_learning.termination import MaxRoundsRule, UnreliableFitRule
from malt.simulation.oracle import Oracle

__all__ = ["Arm", "recommend", "relative_regret", "run_arm"]


@dataclass(frozen=True, slots=True)
class Arm:
    """One experimenter under test: a prior surrogate and an acquisition rule."""

    name: str
    surrogate_model: SurrogateModel
    acquisition: Acquisition


def recommend(model: SurrogateModel, candidates: pd.DataFrame) -> int:
    """Position in `candidates` of the highest posterior mean."""
    return int(np.argmax(model.sample(candidates).mean(axis=0)))


def relative_regret(model: SurrogateModel, oracle: Oracle, candidates: pd.DataFrame) -> float:
    """`1 - mu(recommended) / mu(best candidate)`, on the oracle's true mean."""
    truth = oracle.mean(candidates)
    return float(1.0 - truth[recommend(model, candidates)] / truth.max())


def run_arm(
    arm: Arm,
    oracle: Oracle,
    seed_data: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    batch_size: int,
    rounds: int,
    random_seed: int,
) -> pd.DataFrame:
    """Run one campaign and score it after the seed and after every round.

    Returns one row per round, `0` being the seed: `n_experiments` run so far
    (seed included), `regret`, and `stopped`. A campaign that stops early
    because a fit failed its convergence gate keeps its last recommendation
    for the rounds it did not run, with `stopped` set — so every arm has a
    full curve, and failures are counted rather than dropped.
    """
    journal = run_campaign(
        Experimenter(arm.surrogate_model, arm.acquisition),
        oracle,
        seed_data,
        batch_size,
        random_seed=random_seed,
        termination_rule=UnreliableFitRule() | MaxRoundsRule(rounds),
    )
    models = [journal.seed.surrogate_model] + [r.surrogate_model for r in journal.rounds]
    regrets = [relative_regret(m, oracle, candidates) for m in models]
    ran = len(models)
    regrets += [regrets[-1]] * (rounds + 1 - ran)
    return pd.DataFrame(
        {
            "round": np.arange(rounds + 1),
            "n_experiments": len(seed_data) + batch_size * np.arange(rounds + 1),
            "regret": regrets,
            "stopped": np.arange(rounds + 1) >= ran,
        }
    )
