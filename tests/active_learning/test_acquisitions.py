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
    posterior_mean_optimum,
)
from malt.active_learning.actors import Experimenter
from malt.active_learning.campaign import run_campaign
from malt.active_learning.termination import MaxRoundsRule
from malt.simulation.oracle import GaussianLikelihood, Oracle, quadratic_latent
from malt.active_learning.surrogates import BayesianLinearRegression
from malt.engine.acquisition import central_composite, greedy_q_nei, q_expected_improvement
from malt.engine.factors import Factor, decode_design
from malt.engine.search import space_filling

FACTORS = (Factor("x1", 0.0, 10.0), Factor("x2", 0.1, 10.0, scale="log"))
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
    x = decode_design(FACTORS, np.array(list(itertools.product(np.linspace(-1.0, -0.3, 4), repeat=2))))
    return x.join(ORACLE.query(x, rng).set_axis(x.index))


def recommendation_error(model) -> float:
    """Coded distance from the posterior-mean argmax to the true optimum."""
    recommended = coded(posterior_mean_optimum(model, FACTORS))[0]
    truth = coded(pd.DataFrame({name: [value] for name, value in OPTIMUM.items()}))[0]
    return float(np.abs(recommended - truth).max())


@pytest.fixture
def model() -> BayesianLinearRegression:
    rng = np.random.default_rng(0)
    return BayesianLinearRegression(FACTORS, "quadratic").condition(seed_data(rng), rng)


def full_range_ccd(n_center: int) -> pd.DataFrame:
    return decode_design(FACTORS, central_composite(np.zeros(len(FACTORS)), radius=1.0, n_center=n_center))


MODEL_DRIVEN = [
    pytest.param(QNoisyExpectedImprovement(FACTORS), id="qnei"),
    pytest.param(QUpperConfidenceBound(FACTORS, beta=2.0), id="qucb"),
    pytest.param(ThompsonSampling(FACTORS), id="thompson"),
    pytest.param(CentralComposite(FACTORS), id="ccd"),
]
RULES = [
    *MODEL_DRIVEN,
    pytest.param(RandomBatch(FACTORS), id="random"),
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
    [QNoisyExpectedImprovement(FACTORS), QUpperConfidenceBound(FACTORS, beta=2.0), RandomBatch(FACTORS)],
    ids=["qnei", "qucb", "random"],
)
def test_batches_are_distinct_points(rule, model):
    x = rule.propose(model, BATCH, np.random.default_rng(1))
    assert not x.duplicated().any()


@pytest.mark.parametrize("rule", [QNoisyExpectedImprovement(FACTORS), RandomBatch(FACTORS)], ids=["qnei", "random"])
def test_each_round_searches_afresh(rule, model):
    # No fixed lattice: a different stream gives different points, not the same
    # grid rows. Compared on interior points only — a continuous optimum can sit
    # on a face of the box, and `Factor.decode` clips to the range, so two
    # streams pushing to the same face legitimately agree to the last bit. A
    # lattice would repeat interior points too, which is the failure this guards.
    a = rule.propose(model, BATCH, np.random.default_rng(1))
    b = rule.propose(model, BATCH, np.random.default_rng(2))
    interior = a[(np.abs(coded(a)) < 1.0 - 1e-9).all(axis=1)]
    assert not interior.empty
    assert not np.isin(interior.to_numpy(), b.to_numpy()).all(axis=1).any()


def test_random_batch_is_uniform_on_every_axis_in_coded_units(model):
    x = RandomBatch(FACTORS).propose(model, 20_000, np.random.default_rng(0))
    # Log-uniform on the log factor: its median is the geometric centre, 1.0, not the midpoint 5.05.
    assert np.median(x["x2"]) == pytest.approx(1.0, rel=0.05)
    assert np.median(x["x1"]) == pytest.approx(5.0, rel=0.05)
    counts, _ = np.histogram(coded(x), bins=10, range=(-1, 1))
    assert counts.min() > 0.9 * counts.mean()


def test_ucb_with_zero_beta_proposes_the_top_posterior_means(model):
    x = QUpperConfidenceBound(FACTORS, beta=0.0).propose(model, BATCH, np.random.default_rng(0))
    candidates = decode_design(FACTORS, space_filling(2, 4096, np.random.default_rng(0)))
    top = np.argsort(model.sample(candidates).mean(axis=0))[::-1][:BATCH]
    pd.testing.assert_frame_equal(x, candidates.iloc[top].reset_index(drop=True))


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
        QNoisyExpectedImprovement(FACTORS).propose(
            BayesianLinearRegression(FACTORS), BATCH, np.random.default_rng(0)
        )


def test_ccd_is_centred_on_the_posterior_mean_argmax(model):
    x = CentralComposite(FACTORS, radius=0.3).propose(model, BATCH + 2, np.random.default_rng(0))
    # Where the argmax sits near an edge, the design is moved inward by exactly as much as it needs.
    best = np.clip(coded(posterior_mean_optimum(model, FACTORS))[0], -0.7, 0.7)
    np.testing.assert_allclose(coded(x).mean(axis=0), best, atol=1e-9)
    np.testing.assert_allclose(coded(x.iloc[-2:]), np.repeat(best[None, :], 2, axis=0), atol=1e-9)


def test_ccd_needs_a_batch_that_fits_the_design(model):
    with pytest.raises(ValueError, match="at least 8"):
        CentralComposite(FACTORS).propose(model, 7, np.random.default_rng(0))


def test_ccd_refuses_a_design_that_cannot_fit():
    with pytest.raises(ValueError, match="radius"):
        CentralComposite(FACTORS, radius=0.8, alpha=1.5)


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


# q-NEI picks are optimized off the candidate pool, not taken from it


def q_nei_of(model, batch: pd.DataFrame) -> float:
    """The batch's q-NEI under `model`, on all its draws — the number the rule maximizes."""
    incumbent = model.sample(model.data.loc[:, [f.name for f in FACTORS]]).max(axis=1)
    return q_expected_improvement(model.sample(batch), incumbent)


def discrete_batch(model, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """What the rule would return if picks were confined to the candidate pool.

    The rule it must never lose to, reconstructed here rather than kept in the
    library: `continuous_greedy_q_nei` starts from exactly this batch's picks, so
    this is the floor it is measured against.
    """
    candidates = decode_design(FACTORS, space_filling(len(FACTORS), 4096, rng))
    incumbent = model.sample(model.data.loc[:, list(candidates.columns)]).max(axis=1)
    return candidates.iloc[greedy_q_nei(model.sample(candidates), incumbent, n)].reset_index(drop=True)


def fitted(seed: int = 0):
    return BayesianLinearRegression(FACTORS, features="quadratic").condition(
        seed_data(np.random.default_rng(seed)), np.random.default_rng(seed + 1)
    )


def test_q_nei_is_not_worse_than_picking_from_its_own_pool():
    """Why this needs no switch: each pick starts from the pool-only rule's own choice.

    Provable only for a batch of one — a better first pick raises the bar, so
    later greedy steps face a different objective. This asserts it for a real
    batch, which is evidence, not a theorem.
    """
    model = fitted(0)
    polished = QNoisyExpectedImprovement(FACTORS).propose(model, BATCH, np.random.default_rng(2))
    discrete = discrete_batch(model, BATCH, np.random.default_rng(2))
    assert q_nei_of(model, polished) >= q_nei_of(model, discrete)


def test_q_nei_picks_are_not_confined_to_the_candidate_pool():
    """Otherwise the polish is silently doing nothing."""
    rng = np.random.default_rng(2)
    model = fitted(4)
    pool = space_filling(len(FACTORS), 4096, np.random.default_rng(2))
    picks = coded(QNoisyExpectedImprovement(FACTORS).propose(model, BATCH, rng))
    # At least one pick is strictly off every candidate, i.e. was moved by L-BFGS-B.
    nearest = np.abs(picks[:, None, :] - pool[None, :, :]).max(axis=2).min(axis=1)
    assert nearest.max() > 1e-6


def test_q_nei_lands_close_to_a_known_optimum():
    """The toy check: an in-family surface whose peak the seed alone misplaces."""
    model = fitted(6)
    target = coded(pd.DataFrame({k: [v] for k, v in OPTIMUM.items()}))[0]
    batch = QNoisyExpectedImprovement(FACTORS).propose(model, BATCH, np.random.default_rng(5))
    assert np.linalg.norm(coded(batch) - target, axis=1).min() < 0.5


def test_q_nei_stays_inside_the_declared_ranges():
    model = fitted(8)
    batch = QNoisyExpectedImprovement(FACTORS).propose(model, BATCH, np.random.default_rng(8))
    assert len(batch) == BATCH
    assert list(batch.columns) == [f.name for f in FACTORS]
    for f in FACTORS:
        assert f.contains(batch[f.name].to_numpy()).all()
    assert np.abs(coded(batch)).max() <= 1.0 + 1e-12


def test_q_nei_does_not_stack_the_batch_on_one_point():
    model = fitted(10)
    z = coded(QNoisyExpectedImprovement(FACTORS).propose(model, BATCH, np.random.default_rng(11)))
    distances = [np.linalg.norm(z[i] - z[j]) for i, j in itertools.combinations(range(len(z)), 2)]
    assert min(distances) > 1e-3


def test_q_nei_is_reproducible_from_its_rng():
    model = fitted(14)
    rule = QNoisyExpectedImprovement(FACTORS)
    first = rule.propose(model, BATCH, np.random.default_rng(16))
    pd.testing.assert_frame_equal(first, rule.propose(model, BATCH, np.random.default_rng(16)))


def test_thinning_is_a_stride_over_frozen_draws_not_a_resample():
    """Row `s` must be the same surface thinned or not, or joint draws break."""
    model = BayesianLinearRegression(FACTORS, features="quadratic", n_draws=2000).condition(
        seed_data(np.random.default_rng(12)), np.random.default_rng(13)
    )
    x = seed_data(np.random.default_rng(12)).loc[:, [f.name for f in FACTORS]]
    # The baseline cannot thin, so it must return itself rather than resampling.
    assert model.thinned(500) is model
    np.testing.assert_array_equal(model.thinned(500).sample(x), model.sample(x))
