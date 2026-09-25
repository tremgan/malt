"""Campaign-level properties whose failure would be silent rather than an error."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import (
    GRID,
    FirstRows,
    CountingModel,
    LinearEnvironment,
    NoiseEnvironment,
    RandomPick,
)

from malt.active_learning.actors import Environment, Experimenter
from malt.active_learning.campaign import run_campaign
from malt.active_learning.termination import MaxRoundsRule


def narrow[T](obj: object, cls: type[T]) -> T:
    """The journal types actors by their ABC; these tests know which dummy they are."""
    assert isinstance(obj, cls)
    return obj


def run(seed_data, *, model=None, acquisition=None, environment=None, n=3,
        random_seed=0, rule=MaxRoundsRule(4)):
    experimenter = Experimenter(model or CountingModel(), acquisition or RandomPick())
    return run_campaign(
        experimenter,
        environment or LinearEnvironment(),
        seed_data,
        n,
        random_seed=random_seed,
        termination_rule=rule,
    )


def all_y(journal) -> np.ndarray:
    return np.concatenate([r.data["y"].to_numpy() for r in journal.rounds])


# Rows paired with their own results


def test_observations_pair_with_their_own_design_points(seed_data):
    # RandomPick proposes rows with grid index labels like 412, 87 — an
    # index-aligned join would pair them with the wrong y, or with NaN.
    journal = run(seed_data, acquisition=RandomPick())
    for r in journal.rounds:
        assert not r.data.isna().any().any()
        np.testing.assert_array_equal(r.data["y"], 10.0 * r.data["glucose"])


def test_design_points_keep_their_candidate_index(seed_data):
    journal = run(seed_data, acquisition=RandomPick())
    for r in journal.rounds:
        pd.testing.assert_frame_equal(r.data[["glucose"]], GRID.loc[r.data.index])


# Reproducibility


def test_same_seed_reruns_identically(seed_data):
    a = run(seed_data, environment=NoiseEnvironment(), random_seed=7)
    b = run(seed_data, environment=NoiseEnvironment(), random_seed=7)
    for ra, rb in zip(a.rounds, b.rounds, strict=True):
        assert ra.id == rb.id
        pd.testing.assert_frame_equal(ra.data, rb.data)
        assert (
            narrow(ra.surrogate_model, CountingModel).draw
            == narrow(rb.surrogate_model, CountingModel).draw
        )


def test_different_seed_differs(seed_data):
    a = run(seed_data, environment=NoiseEnvironment(), random_seed=7)
    b = run(seed_data, environment=NoiseEnvironment(), random_seed=8)
    assert not np.array_equal(all_y(a), all_y(b))
    assert {r.id for r in a.rounds}.isdisjoint(r.id for r in b.rounds)


def test_round_ids_are_unique_uuid4(seed_data):
    journal = run(seed_data)
    ids = [r.id for r in journal.rounds]
    assert len(set(ids)) == len(ids)
    assert all(i.version == 4 for i in ids)


# Paired noise across benchmark arms


def test_arms_see_the_same_noise_however_much_randomness_they_use(seed_data):
    # One arm consumes no randomness choosing points, the other a lot. With a
    # shared stream their observation noise would diverge.
    frugal = run(seed_data, acquisition=FirstRows(), environment=NoiseEnvironment())
    greedy = run(seed_data, acquisition=RandomPick(burn=1000), environment=NoiseEnvironment())
    np.testing.assert_array_equal(all_y(frugal), all_y(greedy))


# Journal entries are snapshots, not views


def test_journal_entries_are_independent_snapshots(seed_data):
    journal = run(seed_data, model=CountingModel(), n=3, rule=MaxRoundsRule(3))
    def n_seen(model):
        return narrow(model, CountingModel).n_seen

    assert n_seen(journal.prior) == 0
    assert n_seen(journal.seed.surrogate_model) == 3
    assert [n_seen(r.surrogate_model) for r in journal.rounds] == [6, 9, 12]


def test_mutating_seed_data_afterwards_does_not_change_the_journal(seed_data):
    journal = run(seed_data)
    seed_data.loc[:, "y"] = -1.0
    assert (journal.seed.data["y"] != -1.0).all()


def test_journal_records_its_inputs(seed_data):
    rule = MaxRoundsRule(2)
    journal = run(seed_data, n=5, random_seed=11, rule=rule)
    assert (journal.random_seed, journal.batch_size, journal.termination_rule) == (11, 5, rule)
    assert all(len(r.data) == 5 for r in journal.rounds)


# Environment misuse fails loudly


def test_environment_returning_wrong_row_count_raises(seed_data):
    class TooFew(Environment):
        def query(self, x, rng):
            return pd.DataFrame({"y": [1.0]})

    with pytest.raises(ValueError, match="Length mismatch"):
        run(seed_data, environment=TooFew())


def test_environment_returning_a_factor_column_raises(seed_data):
    class Clobbers(Environment):
        def query(self, x, rng):
            return pd.DataFrame({"y": np.ones(len(x)), "glucose": 0.0})

    with pytest.raises(ValueError, match="overlap"):
        run(seed_data, environment=Clobbers())


def test_environment_covariates_pass_through(seed_data):
    class Lab(Environment):
        def query(self, x, rng):
            return pd.DataFrame({"y": np.ones(len(x)), "operator": "alice"})

    journal = run(seed_data, environment=Lab())
    assert all((r.data["operator"] == "alice").all() for r in journal.rounds)


# Termination wiring


def test_default_rule_runs_ten_rounds(seed_data):
    experimenter = Experimenter(CountingModel(), RandomPick())
    journal = run_campaign(experimenter, LinearEnvironment(), seed_data, 2, random_seed=0)
    assert len(journal.rounds) == 10
    assert journal.stopped_by == (MaxRoundsRule(10),)


def test_rule_is_checked_before_the_first_round(seed_data):
    journal = run(seed_data, rule=MaxRoundsRule(0))
    assert journal.rounds == []
    assert journal.stopped_by == (MaxRoundsRule(0),)


def test_experimenter_ends_in_the_journals_last_state(seed_data):
    experimenter = Experimenter(CountingModel(), FirstRows())
    journal = run_campaign(
        experimenter, LinearEnvironment(), seed_data, 3,
        random_seed=0, termination_rule=MaxRoundsRule(2),
    )
    assert experimenter.surrogate_model is journal.rounds[-1].surrogate_model
