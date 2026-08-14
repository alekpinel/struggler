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

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from struggler.engine.types import Action, Decision, Observation, Side


@dataclass(frozen=True)
class Event:
    """One resolved decision: what was decided, by whom, and its public
    aftermath (DEFCON/VP/turn track, the space race and military ops
    tracks). Only public state — safe to show to either player regardless
    of whose decision it was. Tracks are recorded as their totals right
    after the decision resolved, not as deltas, so a player can read a
    result off a single `Event` without diffing against the previous one.
    """

    actor: Side
    decision: Decision
    action: Action
    defcon: int
    vp: int
    turn: int
    action_round: int
    space_race: Mapping[str, int] = field(default_factory=dict)
    military_ops: Mapping[str, int] = field(default_factory=dict)
    # Populated only when `action.payload` names a country (place_influence,
    # coup/realignment targets, ...): that country's influence and controller
    # right after the decision resolved. Both are public board state, so
    # carrying them here is safe regardless of which side acted.
    country: str | None = None
    country_influence: Mapping[str, int] = field(default_factory=dict)
    country_control: str | None = None


class Player(Protocol):
    def choose_action(self, observation: Observation, history: Sequence[Event]) -> Action:
        """Pick one action from `observation.pending_decision.options`.

        `history` holds every `Event` resolved so far, oldest first —
        including the opponent's moves and CHANCE rolls made since this
        player was last consulted. Bots are free to ignore it.
        """
        ...
