"""Termination rules: when an active-learning loop should stop.

`run_loop` takes a sequence of rules and stops as soon as any one fires, so
rules compose by listing them — a target criterion is capped by adding
`MaxRoundsRule` alongside it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    # Annotation only: `loop` imports this module for its default rule.
    from malt.active_learning.loop import LabJournal

__all__ = ["MaxRoundsRule", "TerminationRule"]


class TerminationRule(Protocol):
    """Decides, before each round, whether the loop should stop.

    A rule is a pure function of the journal: anything it needs (rounds run,
    budget spent, best y, the current surrogate) is derivable from it, so a
    rule never has to carry state of its own — only its configuration. A plain
    `lambda journal: ...` satisfies this too, for one-off rules.
    """

    def __call__(self, journal: LabJournal) -> bool: ...


@dataclass(frozen=True, slots=True)
class MaxRoundsRule:
    """Stop after `k` rounds, not counting the seed."""

    k: int

    def __call__(self, journal: LabJournal) -> bool:
        return len(journal.rounds) >= self.k
