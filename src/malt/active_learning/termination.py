"""Termination rules: when an active-learning loop should stop.

Rules compose with `&` and `|`, each returning a new rule, so a stopping
criterion reads as it would be said aloud:

    TargetRule(...) | MaxRoundsRule(20)          # hit the target, or give up
    (A & B) | MaxRoundsRule(20)

Only rules compose with rules — `&` with anything else is a `TypeError` — so
every part of a composed rule is a named, inspectable object. That is what
lets `fired` report which parts stopped the loop. `run_loop` takes a single
rule; combine several into one before passing it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    # Annotation only: `loop` imports this module for its default rule.
    from malt.active_learning.loop import LabJournal

__all__ = [
    "AllOf",
    "AlwaysStopRule",
    "AnyOf",
    "MaxRoundsRule",
    "NeverStopRule",
    "RandomStopRule",
    "TerminationRule",
    "UnreliableFitRule",
]


class TerminationRule(ABC):
    """Decides, before each round, whether the loop should stop.

    A rule is a pure function of the journal: anything it needs (rounds run,
    budget spent, best y, the current surrogate) is derivable from it, so a
    rule never has to carry state of its own — only its configuration.

    Subclasses implement `__call__` and inherit `&`, `|` and `fired`.
    """

    __slots__ = ()

    @abstractmethod
    def __call__(self, journal: LabJournal) -> bool: ...

    def fired(self, journal: LabJournal) -> tuple[TerminationRule, ...]:
        """The leaf rules responsible for stopping, or `()` if this rule doesn't fire.

        Because a rule is a pure function of the journal, calling this on the
        final journal gives the same answer the rule gave when the loop stopped.
        """
        return (self,) if self(journal) else ()

    def __and__(self, other: TerminationRule) -> AllOf:
        if not isinstance(other, TerminationRule):
            return NotImplemented
        return AllOf((self, other))

    def __or__(self, other: TerminationRule) -> AnyOf:
        if not isinstance(other, TerminationRule):
            return NotImplemented
        return AnyOf((self, other))


@dataclass(frozen=True, slots=True)
class AllOf(TerminationRule):
    """Stop when every rule says stop."""

    rules: tuple[TerminationRule, ...]

    def __call__(self, journal: LabJournal) -> bool:
        return all(rule(journal) for rule in self.rules)

    def fired(self, journal: LabJournal) -> tuple[TerminationRule, ...]:
        if not self(journal):
            return ()
        return tuple(leaf for rule in self.rules for leaf in rule.fired(journal))


@dataclass(frozen=True, slots=True)
class AnyOf(TerminationRule):
    """Stop when any rule says stop."""

    rules: tuple[TerminationRule, ...]

    def __call__(self, journal: LabJournal) -> bool:
        return any(rule(journal) for rule in self.rules)

    def fired(self, journal: LabJournal) -> tuple[TerminationRule, ...]:
        return tuple(leaf for rule in self.rules for leaf in rule.fired(journal))


"""Library of termination rules for active learning loops.
vvv"""


@dataclass(frozen=True, slots=True)
class AlwaysStopRule(TerminationRule):
    """Always stop — the identity for `&`. Alone, the loop runs no rounds."""

    def __call__(self, journal: LabJournal) -> bool:
        return True


@dataclass(frozen=True, slots=True)
class NeverStopRule(TerminationRule):
    """Never stop — the identity for `|`. Alone, the loop runs forever."""

    def __call__(self, journal: LabJournal) -> bool:
        return False

@dataclass(frozen=True, slots=True)
class RandomStopRule(TerminationRule):
    """Stop with probability `p` before each round, so campaign length is geometric.

    The coin is not drawn from a live generator: a rule must be a pure function
    of the journal, or `fired` could not reconstruct why the loop stopped. Each
    flip is instead seeded by the campaign's `random_seed` and the number of
    rounds run, which makes it reproducible and independent of the loop's own
    random streams.
    """

    p: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {self.p}")

    def __call__(self, journal: LabJournal) -> bool:
        flip = np.random.default_rng([journal.random_seed, len(journal.rounds)])
        return bool(flip.random() < self.p)


@dataclass(frozen=True, slots=True)
class MaxRoundsRule(TerminationRule):
    """Stop after `k` rounds, not counting the seed."""

    k: int

    def __call__(self, journal: LabJournal) -> bool:
        return len(journal.rounds) >= self.k


@dataclass(frozen=True, slots=True)
class UnreliableFitRule(TerminationRule):
    """Stop when the current surrogate can't be trusted to propose from.

    The loop proposes from whatever model it holds; a fit that failed its
    convergence gate would otherwise pick the next experiments from unmixed
    draws. Checked before every round, so an unreliable seed fit stops the loop
    before anything is proposed. The failed model stays in the journal for
    diagnosis.
    """

    def __call__(self, journal: LabJournal) -> bool:
        current = (
            journal.rounds[-1].surrogate_model
            if journal.rounds
            else journal.seed.surrogate_model
        )
        return not current.reliable


