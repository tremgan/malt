"""Regret scoring and the per-arm runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import pytest

from malt.active_learning.acquisitions import CentralComposite, RandomBatch
from malt.active_learning.actors import SurrogateModel
from malt.active_learning.surrogates import BayesianLinearRegression
from malt.simulation.regret import Arm, relative_regret, run_arm
from malt.engine.factors import Factor, candidate_grid
from malt.simulation.oracle import GammaLikelihood, Oracle, quadratic_latent

FACTORS = (Factor("x1", 0.0, 10.0), Factor("x2", 0.1, 10.0, scale="log"))
CANDIDATES = candidate_grid(FACTORS, 11)
ORACLE = Oracle(
    quadratic_latent(FACTORS, optimum={"x1": 7.0, "x2": 0.5}, peak=np.log(10.0), curvature=1.0),
    GammaLikelihood(20.0),
)


@dataclass(frozen=True)
class FixedSurface(SurrogateModel):
    """Believes exactly `surface`, with one draw; unreliable once `fail_at` points are seen."""

    surface: Callable[[pd.DataFrame], np.ndarray]
    n_seen: int = 0
    fail_at: float = np.inf

    def condition(self, data, rng):
        return FixedSurface(self.surface, self.n_seen + len(data), self.fail_at)

    def sample(self, x):
        return self.surface(x)[None, :]

    @property
    def data(self):
        return None

    @property
    def reliable(self):
        return self.n_seen < self.fail_at


def seed_data() -> pd.DataFrame:
    x = candidate_grid(FACTORS, 4)
    return x.join(ORACLE.query(x, np.random.default_rng(0)).set_axis(x.index))


def test_regret_is_zero_at_the_true_optimum_and_worst_at_the_minimum():
    truth = ORACLE.mean(CANDIDATES)
    assert relative_regret(FixedSurface(ORACLE.mean), ORACLE, CANDIDATES) == 0.0
    worst = relative_regret(FixedSurface(lambda x: -ORACLE.mean(x)), ORACLE, CANDIDATES)
    assert worst == pytest.approx(1 - truth.min() / truth.max())


def test_run_arm_scores_the_seed_and_every_round():
    arm = Arm("rsm", BayesianLinearRegression(FACTORS, "quadratic"), CentralComposite(FACTORS, CANDIDATES))
    curve = run_arm(arm, ORACLE, seed_data(), CANDIDATES, batch_size=8, rounds=3, random_seed=0)
    assert list(curve["round"]) == [0, 1, 2, 3]
    assert list(curve["n_experiments"]) == [16, 24, 32, 40]
    assert ((curve["regret"] >= 0) & (curve["regret"] < 1)).all()
    assert list(curve["stopped"]) == [False] * 4


def test_a_campaign_stopped_by_a_failed_fit_keeps_its_last_recommendation():
    # Unreliable once the seed plus one round (16 + 8) is seen: round 1 runs,
    # then the rule stops the campaign before round 2.
    arm = Arm("fails", FixedSurface(ORACLE.mean, fail_at=24), RandomBatch(CANDIDATES))
    curve = run_arm(arm, ORACLE, seed_data(), CANDIDATES, batch_size=8, rounds=3, random_seed=0)
    assert list(curve["stopped"]) == [False, False, True, True]
    assert curve["regret"].nunique() == 1
