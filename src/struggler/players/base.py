"""The Player interface: how any decision-maker plugs into a game.

A `Player` is deliberately a structural `Protocol`, not a base class to
subclass — writing a bot means writing any object with a matching
`choose_action` method, no inheritance required.

A player only ever sees an `Observation` (mandate #4: the per-player
observation function is the sole sanctioned view of the game), and only
ever returns one `Action` drawn from `observation.pending_decision.options`
(mandate #2: atomic decisions). `Side.CHANCE` decisions never reach a
`Player` at all — see `struggler.runner.play_game`.
"""

from __future__ import annotations

from typing import Protocol

from struggler.types import Action, Observation


class Player(Protocol):
    def choose_action(self, observation: Observation) -> Action:
        """Pick one action from `observation.pending_decision.options`."""
        ...
