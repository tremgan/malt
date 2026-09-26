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
from malt.simulation.regret import Arm, instantaneous_regret, run_arm, true_optimum
from malt.engine.factors import Factor, candidate_grid, decode_design
from malt.engine.search import maximize
from malt.simulation.oracle import GammaLikelihood, Oracle, quadratic_latent

FACTORS = (Factor("x1", 0.0, 10.0), Factor("x2", 0.1, 10.0, scale="log"))
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


def test_true_optimum_is_the_planted_peak():
    optimum, best = true_optimum(ORACLE, FACTORS)
    assert best == pytest.approx(10.0, rel=1e-9)
    assert optimum.iloc[0].to_dict() == pytest.approx({"x1": 7.0, "x2": 0.5}, rel=1e-4)


def test_instantaneous_regret_is_zero_at_the_optimum_and_worst_at_the_minimum():
    optimum, best = true_optimum(ORACLE, FACTORS)
    z_worst, neg_worst = maximize(lambda z: -ORACLE.mean(decode_design(FACTORS, z)), len(FACTORS))
    regret = instantaneous_regret(ORACLE, pd.concat([optimum, decode_design(FACTORS, z_worst)]), best)
    np.testing.assert_allclose(regret, [0.0, 1 + neg_worst / best], atol=1e-9)


def test_run_arm_accumulates_the_regret_of_the_runs_it_chose():
    arm = Arm("rsm", BayesianLinearRegression(FACTORS, "quadratic"), CentralComposite(FACTORS))
    curve = run_arm(arm, ORACLE, seed_data(), FACTORS, batch_size=8, rounds=3, random_seed=0)
    assert list(curve["round"]) == [0, 1, 2, 3]
    assert list(curve["n_experiments"]) == [16, 24, 32, 40]
    batch = curve["batch_regret"].to_numpy()
    assert batch[0] == 0.0  # the seed is nobody's choice
    assert ((batch >= 0) & (batch < 8)).all()  # each of 8 runs gives up less than one optimum
    np.testing.assert_allclose(curve["cumulative_regret"], np.cumsum(batch))
    assert list(curve["stopped"]) == [False] * 4


def test_a_model_free_rule_scores_the_same_whatever_the_model():
    # Neither model consumes randomness, so RandomBatch proposes the same runs
    # beside a model that believes the truth and one that believes its opposite.
    # Scoring the runs, not a recommendation, gives identical regret.
    right = run_arm(Arm("right", FixedSurface(ORACLE.mean), RandomBatch(FACTORS)),
                    ORACLE, seed_data(), FACTORS, batch_size=8, rounds=3, random_seed=0)
    wrong = run_arm(Arm("wrong", FixedSurface(lambda x: -ORACLE.mean(x)), RandomBatch(FACTORS)),
                    ORACLE, seed_data(), FACTORS, batch_size=8, rounds=3, random_seed=0)
    np.testing.assert_array_equal(right["cumulative_regret"], wrong["cumulative_regret"])


def test_rounds_a_stopped_campaign_never_ran_have_no_regret():
    # Unreliable once the seed plus one round (16 + 8) is seen: round 1 runs,
    # then the rule stops the campaign before round 2.
    arm = Arm("fails", FixedSurface(ORACLE.mean, fail_at=24), RandomBatch(FACTORS))
    curve = run_arm(arm, ORACLE, seed_data(), FACTORS, batch_size=8, rounds=3, random_seed=0)
    assert list(curve["stopped"]) == [False, False, True, True]
    assert curve["cumulative_regret"].isna().tolist() == [False, False, True, True]
