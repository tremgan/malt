"""Termination rules: composition, attribution, and each rule's behaviour."""

from __future__ import annotations

import pickle
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from conftest import CountingModel, LinearEnvironment, RandomPick

from malt.active_learning.actors import Experimenter
from malt.active_learning.loop import LabJournal, run_loop
from malt.active_learning.termination import (
    AllOf,
    AlwaysStopRule,
    AnyOf,
    MaxRoundsRule,
    NeverStopRule,
    RandomStopRule,
    TerminationRule,
    UnreliableFitRule,
)


def journal_at(k: int, random_seed: int = 0) -> LabJournal:
    """A stand-in journal with `k` completed rounds — enough for count-based rules."""
    return cast(LabJournal, SimpleNamespace(rounds=[None] * k, random_seed=random_seed))


Always, Never = AlwaysStopRule(), NeverStopRule()


# Composition


@pytest.mark.parametrize(
    ("rule", "expected"),
    [
        (Always | Never, True),
        (Never | Always, True),
        (Never | Never, False),
        (Always & Always, True),
        (Always & Never, False),
        (Never & Always, False),
        # & binds tighter than |
        (Never & Always | Always, True),
        (Always | Always & Never, True),
        (Never & (Always | Always), False),
    ],
)
def test_composition_truth_table(rule, expected):
    assert rule(journal_at(0)) is expected


@pytest.mark.parametrize("k", [0, 2, 5])
def test_always_and_never_are_identities(k):
    rule = MaxRoundsRule(2)
    assert (NeverStopRule() | rule)(journal_at(k)) == rule(journal_at(k))
    assert (AlwaysStopRule() & rule)(journal_at(k)) == rule(journal_at(k))
    assert (NeverStopRule() | rule).fired(journal_at(k)) == rule.fired(journal_at(k))


def test_composition_returns_new_rules():
    rule = MaxRoundsRule(3) | MaxRoundsRule(5)
    assert isinstance(rule, AnyOf)
    assert isinstance(MaxRoundsRule(3) & MaxRoundsRule(5), AllOf)
    assert rule.rules == (MaxRoundsRule(3), MaxRoundsRule(5))


@pytest.mark.parametrize(
    "compose",
    [
        lambda r: r | (lambda journal: True),
        lambda r: (lambda journal: True) | r,
        lambda r: r & (lambda journal: True),
        lambda r: (lambda journal: True) & r,
        lambda r: r | 5,
    ],
)
def test_only_rules_compose_with_rules(compose):
    with pytest.raises(TypeError):
        compose(MaxRoundsRule(3))


def test_base_class_is_abstract():
    with pytest.raises(TypeError):
        TerminationRule()  # pyright: ignore[reportAbstractUsage] — the point of the test


def test_rules_are_values():
    rule = (MaxRoundsRule(9) & UnreliableFitRule()) | RandomStopRule(0.1)
    assert rule == (MaxRoundsRule(9) & UnreliableFitRule()) | RandomStopRule(0.1)
    assert hash(rule) == hash((MaxRoundsRule(9) & UnreliableFitRule()) | RandomStopRule(0.1))
    assert pickle.loads(pickle.dumps(rule)) == rule


# Attribution: which parts fired


def test_fired_on_a_leaf():
    assert MaxRoundsRule(3).fired(journal_at(3)) == (MaxRoundsRule(3),)
    assert MaxRoundsRule(3).fired(journal_at(2)) == ()


def test_fired_any_of_reports_every_part_that_fired():
    rule = MaxRoundsRule(2) | MaxRoundsRule(3) | MaxRoundsRule(9)
    assert rule.fired(journal_at(3)) == (MaxRoundsRule(2), MaxRoundsRule(3))


def test_fired_all_of_reports_nothing_unless_all_fire():
    rule = MaxRoundsRule(2) & MaxRoundsRule(5)
    assert rule.fired(journal_at(3)) == ()
    assert rule.fired(journal_at(5)) == (MaxRoundsRule(2), MaxRoundsRule(5))


def test_fired_nested():
    rule = (MaxRoundsRule(1) & MaxRoundsRule(2)) | MaxRoundsRule(9)
    assert rule.fired(journal_at(2)) == (MaxRoundsRule(1), MaxRoundsRule(2))


# Each rule, through the real loop


def loop(seed_data, rule, *, model=None, random_seed=0, n=1):
    experimenter = Experimenter(model or CountingModel(), RandomPick())
    return run_loop(
        experimenter, LinearEnvironment(), seed_data, n,
        random_seed=random_seed, termination_rule=rule,
    )


def test_stopped_by_names_the_rule_that_ended_the_loop(seed_data):
    rule = UnreliableFitRule() | MaxRoundsRule(4)
    capped = loop(seed_data, rule, n=3)
    failed = loop(seed_data, rule, model=CountingModel(fail_at=9), n=3)
    assert (len(capped.rounds), capped.stopped_by) == (4, (MaxRoundsRule(4),))
    assert (len(failed.rounds), failed.stopped_by) == (2, (UnreliableFitRule(),))


def test_stopped_by_reports_simultaneous_rules(seed_data):
    # 3 seed + 3 rounds of 3 = 12 points: unreliable exactly when the cap hits.
    journal = loop(seed_data, UnreliableFitRule() | MaxRoundsRule(3),
                   model=CountingModel(fail_at=12), n=3)
    assert journal.stopped_by == (UnreliableFitRule(), MaxRoundsRule(3))


def test_unreliable_seed_fit_stops_before_proposing(seed_data):
    journal = loop(seed_data, UnreliableFitRule() | MaxRoundsRule(10),
                   model=CountingModel(fail_at=3))
    assert journal.rounds == []
    assert journal.stopped_by == (UnreliableFitRule(),)


def test_random_stop_is_reproducible_and_pure(seed_data):
    rule = RandomStopRule(0.3) | MaxRoundsRule(50)
    a = loop(seed_data, rule, random_seed=7)
    b = loop(seed_data, rule, random_seed=7)
    assert len(a.rounds) == len(b.rounds)
    # Re-evaluating on the finished journal agrees with what stopped the loop.
    assert rule.fired(a) == a.stopped_by


@pytest.mark.parametrize(("p", "rounds"), [(0.0, 5), (1.0, 0)])
def test_random_stop_extremes(seed_data, p, rounds):
    assert len(loop(seed_data, RandomStopRule(p) | MaxRoundsRule(5)).rounds) == rounds


def test_random_stop_length_is_geometric():
    rule = RandomStopRule(0.25)

    def length(seed: int) -> int:
        k = 0
        while not rule(journal_at(k, random_seed=seed)):
            k += 1
        return k

    lengths = np.array([length(s) for s in range(4000)])
    # Mean (1-p)/p = 3; standard error ~0.055, so 0.25 is ~4.5 SE.
    assert abs(lengths.mean() - 3.0) < 0.25


@pytest.mark.parametrize("p", [-0.1, 1.5, float("nan")])
def test_random_stop_rejects_invalid_p(p):
    with pytest.raises(ValueError):
        RandomStopRule(p)
