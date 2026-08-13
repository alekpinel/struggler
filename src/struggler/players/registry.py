"""Bot registry: name -> factory, so picking a bot is a one-line config
change and adding a new one is a one-line addition -- no changes to the
engine, `runner.play_game`, or the CLI required.

To add a bot: write any object with a `choose_action(observation, history)
-> Action` method (see `players.base.Player` -- structural, no subclassing
needed) and register a factory for it below.
"""

from __future__ import annotations

from typing import Callable

from struggler.players.base import Player
from struggler.players.bots import FirstLegalPlayer, RandomPlayer
from struggler.players.greedy import GreedyPlayer
from struggler.players.human import HumanPlayer

PlayerFactory = Callable[..., Player]

PLAYER_REGISTRY: dict[str, PlayerFactory] = {
    "human": lambda seed=0: HumanPlayer(),
    "random": lambda seed=0: RandomPlayer(seed=seed),
    "first": lambda seed=0: FirstLegalPlayer(),
    "greedy": lambda seed=0: GreedyPlayer(),
}


def build_player(kind: str, *, seed: int = 0) -> Player:
    """Instantiate the registered bot named `kind`. Raises `ValueError`
    (listing the available names) for an unknown one."""
    try:
        factory = PLAYER_REGISTRY[kind]
    except KeyError:
        available = ", ".join(sorted(PLAYER_REGISTRY))
        raise ValueError(f"unknown player kind: {kind!r} (available: {available})") from None
    return factory(seed=seed)
