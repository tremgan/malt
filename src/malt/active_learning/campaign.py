"""The active-learning campaign: an experimenter acting on an environment, round by round.

`run_campaign` is the only place randomness enters. It takes one `random_seed`,
records it in the `LabJournal`, and spawns independent streams from it — one
for the experimenter, one for the environment, one for round IDs. Keeping the environment on its
own stream is what makes benchmark arms comparable: two experimenters run with
the same seed see the same observation noise, however much randomness their
models and acquisition rules consume.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from malt.active_learning.actors import (
    Acquisition,
    Environment,
    Experimenter,
    SurrogateModel,
)
from malt.active_learning.termination import MaxRoundsRule, TerminationRule

__all__ = [
    "LabJournal",
    "RoundEntry",
    "Seed",
    "campaign_step",
    "run_campaign",
]


@dataclass(frozen=True, slots=True)
class Seed:
    """The experiments a campaign starts from, before any active learning.

    Not a round: the seed is chosen outside the campaign (a house design, historical
    data) and is shared across the experimenters a benchmark compares.
    """

    data: pd.DataFrame
    surrogate_model: SurrogateModel  # the prior conditioned on `data`


@dataclass(frozen=True, slots=True)
class RoundEntry:
    """One round of the campaign, with the model once it finished.

    The model that *proposed* this round's batch is the previous round's (or the
    seed's); storing the model after the round makes the journal's last entry
    the campaign's current state.
    """

    id: uuid.UUID # id for the entry but also for the batch of experiments it represents -> used for batch effect modelling
    data: pd.DataFrame 
    surrogate_model: SurrogateModel  # after conditioning on `data`


@dataclass(slots=True)
class LabJournal:
    """"Remember kids, the only difference between screwing around and science
    is writing it down."

    Everything needed to replay a campaign, and to rerun a synthetic one.

    Replay needs the starting experimenter, the seed and every round. Rerunning
    additionally needs `random_seed` — recorded here, but handed to the actors
    by `run_campaign`, not by the journal.
    """

    random_seed: int
    batch_size: int
    prior: SurrogateModel  # the surrogate before any data
    acquisition: Acquisition  # stateless, so the same rule every round
    seed: Seed
    termination_rule: TerminationRule
    rounds: list[RoundEntry] = field(default_factory=list)
    # The leaf rules that stopped the campaign; set by `run_campaign` once it ends.
    stopped_by: tuple[TerminationRule, ...] = ()

    def log(self, entry: RoundEntry) -> None:
        """Append a completed round."""
        self.rounds.append(entry)


def campaign_step(
    experimenter: Experimenter,
    environment: Environment,
    n: int,
    experimenter_rng: np.random.Generator,
    environment_rng: np.random.Generator,
    id_rng: np.random.Generator,
) -> RoundEntry:
    """One round of the active-learning campaign.

    1. The experimenter proposes `n` design points.
    2. The environment is queried at those points.
    3. The experimenter conditions its surrogate model on the new observations.

    Returns the round: its ID, the new observations — the proposed factor
    columns joined to whatever the environment returned — and the model
    after it.
    """
    # Drawn from its own stream, not uuid4(), so a rerun with the same seed
    # reproduces the IDs along with everything else.
    round_id = uuid.UUID(int=int.from_bytes(id_rng.bytes(16)), version=4)
    x = experimenter.propose(n, experimenter_rng)
    observations = environment.query(x, environment_rng)
    # Positional, not index-aligned: `x` usually keeps its candidate-grid row
    # labels, which an index join would misalign against a fresh 0..n-1 index.
    # `set_axis` raises on a length mismatch; `join` on a column name clash.
    data = x.join(observations.set_axis(x.index))
    experimenter.observe(data, experimenter_rng)
    return RoundEntry(round_id, data, experimenter.surrogate_model)


def run_campaign(
    experimenter: Experimenter,
    environment: Environment,
    seed_data: pd.DataFrame,
    n: int,
    *,
    random_seed: int,
    termination_rule: TerminationRule = MaxRoundsRule(10),
) -> LabJournal:
    """Condition on the seed, then run `campaign_step` with batches of `n` until
    `termination_rule` fires.

    The rule is checked before every round, including the first. Combine
    several with `&` and `|` — a target criterion is capped with
    `target | MaxRoundsRule(k)`.
    """
    # Independent streams, not two generators on the same seed — those would
    # produce identical sequences and correlate choices with observation noise.
    experimenter_rng, environment_rng, id_rng = np.random.default_rng(
        random_seed
    ).spawn(3)

    prior, acquisition = experimenter.surrogate_model, experimenter.acquisition
    seed_data = seed_data.copy()
    experimenter.observe(seed_data, experimenter_rng)
    journal = LabJournal(
        random_seed=random_seed,
        batch_size=n,
        prior=prior,
        acquisition=acquisition,
        seed=Seed(seed_data, experimenter.surrogate_model),
        termination_rule=termination_rule,
    )

    while not termination_rule(journal):
        journal.log(
            campaign_step(
                experimenter=experimenter,
                environment=environment,
                n=n,
                experimenter_rng=experimenter_rng,
                environment_rng=environment_rng,
                id_rng=id_rng,
            )
        )
    journal.stopped_by = termination_rule.fired(journal)
    return journal
