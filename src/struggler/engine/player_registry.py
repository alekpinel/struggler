"""Dynamic bot registry: name -> factory, populated by self-registration.

Each bot decorates its own factory with `@register("name")` in its own
module (see `struggler.bots.greedy`/`struggler.bots.naive`). Nothing here
enumerates the available bots -- whoever builds players (`main.py`, tests)
imports the specific bot modules it wants available before calling
`build_player`, and that import is what makes the name resolvable. This
keeps adding a bot a one-file change: write any object with a
`choose_action(observation, history) -> Action` method (see `Player` --
structural, no subclassing needed) and decorate a factory for it.

`"human"` is registered right here, since `HumanPlayer` already lives in
this package -- the engine stays self-sufficient for a human-vs-human game
with no dependency on `struggler.bots`.
"""

from __future__ import annotations

from typing import Callable

from .human import HumanPlayer
from .player import Player

PlayerFactory = Callable[..., Player]

_FACTORIES: dict[str, PlayerFactory] = {}


def register(name: str) -> Callable[[PlayerFactory], PlayerFactory]:
    """Class/function decorator: makes `factory` available as `name`."""

    def decorator(factory: PlayerFactory) -> PlayerFactory:
        _FACTORIES[name] = factory
        return factory

    return decorator


def available() -> tuple[str, ...]:
    """Names registered so far -- reflects whatever bot modules have been
    imported, not the full set of bots that exist in `struggler.bots`."""
    return tuple(sorted(_FACTORIES))


def build_player(kind: str, *, seed: int = 0) -> Player:
    """Instantiate the registered bot named `kind`. Raises `ValueError`
    (listing the names registered so far) for an unknown one -- which
    includes a bot that exists but whose module was never imported."""
    try:
        factory = _FACTORIES[kind]
    except KeyError:
        available = ", ".join(sorted(_FACTORIES))
        raise ValueError(f"unknown player kind: {kind!r} (available: {available})") from None
    return factory(seed=seed)


register("human")(lambda seed=0: HumanPlayer())
