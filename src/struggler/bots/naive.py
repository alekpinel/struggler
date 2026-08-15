"""Trivial baseline bots: reference implementations of the `Player` protocol."""

from __future__ import annotations

import random
from typing import Sequence

from struggler.engine import Action, Observation
from struggler.engine.player import Event


class FirstLegalPlayer:
    """Always picks the first legal option. Deterministic baseline."""

    def choose_action(self, observation: Observation, history: Sequence[Event]) -> Action:
        return observation.pending_decision.options[0]


class RandomPlayer:
    """Picks uniformly among legal options, using its own seeded RNG.

    Deliberately never touches the engine's RNG — a bot's choices must
    not perturb (or depend on) the engine's own dice sequence.
    """

    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def choose_action(self, observation: Observation, history: Sequence[Event]) -> Action:
        return self._rng.choice(observation.pending_decision.options)
