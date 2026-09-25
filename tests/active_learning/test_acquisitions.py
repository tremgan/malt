"""Acquisition rules against a real (closed-form, fast) surrogate and a known peak."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from malt.active_learning.acquisitions import (
    CentralComposite,
    FixedDesign,
    QNoisyExpectedImprovement,
    QUpperConfidenceBound,
    RandomBatch,
    ThompsonSampling,
)
from malt.active_learning.actors import Experimenter
from malt.active_learning.campaign import run_campaign
from malt.active_learning.termination import MaxRoundsRule
from malt.simulation.oracle import GaussianLikelihood, Oracle, quadratic_latent
from malt.active_learning.surrogates import BayesianLinearRegression
from malt.engine.acquisition import central_composite
from malt.engine.factors import Factor, candidate_grid

FACTORS = (Factor("x1", 0.0, 10.0), Factor("x2", 0.1, 10.0, scale="log"))
CANDIDATES = candidate_grid(FACTORS, 21)
OPTIMUM = {"x1": 7.0, "x2": 0.5}
ORACLE = Oracle(
    quadratic_latent(FACTORS, optimum=OPTIMUM, peak=10.0, curvature=3.0),
    GaussianLikelihood(1.0),
)
BATCH = 8  # the smallest CCD batch in two factors


def coded(x: pd.DataFrame) -> np.ndarray:
    return np.column_stack([f.encode(x[f.name].to_numpy()) for f in FACTORS])


def seed_data(rng: np.random.Generator) -> pd.DataFrame:
    """A 4x4 grid in the low corner, far from the peak: the seed alone misplaces it."""
    z = np.array(list(itertools.product(np.linspace(-1.0, -0.3, 4), repeat=2)))
    x = pd.DataFrame({f.name: f.decode(z[:, j]) for j, f in enumerate(FACTORS)})
    return x.join(ORACLE.query(x, rng).set_axis(x.index))


def recommendation_error(model) -> float:
    """Coded distance from the posterior-mean argmax to the true best candidate."""
    recommended = CANDIDATES.iloc[[int(np.argmax(model.sample(CANDIDATES).mean(axis=0)))]]
    true_best = CANDIDATES.iloc[[int(np.argmax(ORACLE.mean(CANDIDATES)))]]
    return float(np.abs(coded(recommended) - coded(true_best)).max())


@pytest.fixture
def model() -> BayesianLinearRegression:
    rng = np.random.default_rng(0)
    return BayesianLinearRegression(FACTORS, "quadratic").condition(seed_data(rng), rng)


def full_range_ccd(n_center: int) -> pd.DataFrame:
    coded = central_composite(np.zeros(len(FACTORS)), radius=1.0, n_center=n_center)
    return pd.DataFrame({f.name: f.decode(coded[:, j]) for j, f in enumerate(FACTORS)})


MODEL_DRIVEN = [
    pytest.param(QNoisyExpectedImprovement(CANDIDATES), id="qnei"),
    pytest.param(QUpperConfidenceBound(CANDIDATES, beta=2.0), id="qucb"),
    pytest.param(ThompsonSampling(CANDIDATES), id="thompson"),
    pytest.param(CentralComposite(FACTORS, CANDIDATES), id="ccd"),
]
RULES = [
    *MODEL_DRIVEN,
    pytest.param(RandomBatch(CANDIDATES), id="random"),
    pytest.param(FixedDesign(FACTORS, full_range_ccd(n_center=0)), id="fixed"),
]


@pytest.mark.parametrize("rule", RULES)
def test_proposals_are_a_batch_inside_the_declared_ranges(rule, model):
    x = rule.propose(model, BATCH, np.random.default_rng(1))
    assert list(x.columns) == [f.name for f in FACTORS] and len(x) == BATCH
    for f in FACTORS:
        assert f.contains(x[f.name].to_numpy()).all()


@pytest.mark.parametrize("rule", RULES)
def test_proposals_are_reproducible_from_rng(rule, model):
    a = rule.propose(model, BATCH, np.random.default_rng(1))
    b = rule.propose(model, BATCH, np.random.default_rng(1))
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.parametrize(
    "rule",
    [
        QNoisyExpectedImprovement(CANDIDATES),
        QUpperConfidenceBound(CANDIDATES, beta=2.0),
        RandomBatch(CANDIDATES),
    ],
    ids=["qnei", "qucb", "random"],
)
def test_candidate_batches_are_distinct_points(rule, model):
    x = rule.propose(model, BATCH, np.random.default_rng(1))
    assert not x.index.duplicated().any()


def test_ucb_with_zero_beta_proposes_the_top_posterior_means(model):
    x = QUpperConfidenceBound(CANDIDATES, beta=0.0).propose(model, BATCH, np.random.default_rng(0))
    top = np.argsort(model.sample(CANDIDATES).mean(axis=0))[::-1][:BATCH]
    pd.testing.assert_frame_equal(x, CANDIDATES.iloc[top])


def test_fixed_design_is_proposed_whole(model):
    design = full_range_ccd(n_center=4)
    rule = FixedDesign(FACTORS, design)
    pd.testing.assert_frame_equal(rule.propose(model, 12, np.random.default_rng(0)), design)
    with pytest.raises(ValueError, match="batch size must be 12, got 8"):
        rule.propose(model, 8, np.random.default_rng(0))


def test_fixed_design_refuses_runs_outside_the_ranges():
    with pytest.raises(ValueError, match="range of: \\['x1'\\]"):
        FixedDesign(FACTORS, full_range_ccd(0).assign(x1=11.0))
    with pytest.raises(ValueError, match="missing factors: \\['x2'\\]"):
        FixedDesign(FACTORS, full_range_ccd(0).drop(columns="x2"))


def test_qnei_refuses_an_unconditioned_model():
    with pytest.raises(ValueError, match="condition on the seed first"):
        QNoisyExpectedImprovement(CANDIDATES).propose(
            BayesianLinearRegression(FACTORS), BATCH, np.random.default_rng(0)
        )


def test_ccd_is_centred_on_the_posterior_mean_argmax(model):
    x = CentralComposite(FACTORS, CANDIDATES, radius=0.3).propose(model, BATCH + 2, np.random.default_rng(0))
    best = CANDIDATES.iloc[[int(np.argmax(model.sample(CANDIDATES).mean(axis=0)))]]
    np.testing.assert_allclose(coded(x).mean(axis=0), coded(best)[0], atol=1e-9)
    np.testing.assert_allclose(coded(x.iloc[-2:]), np.repeat(coded(best), 2, axis=0), atol=1e-9)


def test_ccd_needs_a_batch_that_fits_the_design(model):
    with pytest.raises(ValueError, match="at least 8"):
        CentralComposite(FACTORS, CANDIDATES).propose(model, 7, np.random.default_rng(0))


def test_ccd_refuses_a_design_that_cannot_fit():
    with pytest.raises(ValueError, match="radius"):
        CentralComposite(FACTORS, CANDIDATES, radius=0.8, alpha=1.5)


@pytest.mark.parametrize("rule", MODEL_DRIVEN)
def test_campaign_finds_the_peak_the_seed_misses(rule, model):
    assert recommendation_error(model) > 0.5
    journal = run_campaign(
        Experimenter(BayesianLinearRegression(FACTORS, "quadratic"), rule),
        ORACLE,
        seed_data(np.random.default_rng(0)),
        BATCH,
        random_seed=0,
        termination_rule=MaxRoundsRule(3),
    )
    assert recommendation_error(journal.rounds[-1].surrogate_model) <= 0.25
