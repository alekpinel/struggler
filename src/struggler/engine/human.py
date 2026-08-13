"""A console `Player`: prompts a human for each decision via stdin/stdout."""

from __future__ import annotations

from typing import Sequence

from .player import Event
from .types import Action, Observation


def _format_action(action: Action) -> str:
    return f"{action.kind.value} {dict(action.payload)}"


def _format_event(event: Event) -> str:
    return (
        f"  {event.actor.value}: {_format_action(event.action)}"
        f" (DEFCON {event.defcon}, VP {event.vp}, turn {event.turn}.{event.action_round},"
        f" space race US={event.space_race.get('US', 0)}/USSR={event.space_race.get('USSR', 0)},"
        f" military ops US={event.military_ops.get('US', 0)}/USSR={event.military_ops.get('USSR', 0)})"
    )


class HumanPlayer:
    """Prints the current decision and reads a choice from the console.

    Between prompts it also shows what happened since this player was last
    asked (the opponent's moves, chance rolls) and offers on-demand "board"
    and "history" commands, so a human doesn't have to track state by hand.

    Only ever returns an `Action` object taken verbatim from
    `observation.pending_decision.options` — never a hand-built payload —
    so the engine's legality check always passes.
    """

    def __init__(self) -> None:
        self._last_seen = 0

    def choose_action(self, observation: Observation, history: Sequence[Event]) -> Action:
        new_events = history[self._last_seen :]
        self._last_seen = len(history)
        if new_events:
            print(f"\nSince your last turn ({len(new_events)} event(s)):")
            for event in new_events:
                print(_format_event(event))

        decision = observation.pending_decision
        print(
            f"\n[{observation.side.value}] turn {observation.turn}, "
            f"action round {observation.action_round}, "
            f"DEFCON {observation.defcon}, VP {observation.vp}"
        )
        print(f"Decision: {decision.kind.value} (context: {dict(decision.context)})")
        for i, action in enumerate(decision.options):
            print(f"  [{i}] {_format_action(action)}")

        while True:
            raw = input(
                f"Choose an option [0-{len(decision.options) - 1}], "
                "or 'b' for board / 'h' for full history: "
            ).strip().lower()
            if raw in ("b", "board"):
                _print_board(observation)
                continue
            if raw in ("h", "history"):
                _print_history(history)
                continue
            if raw.isdigit() and int(raw) < len(decision.options):
                return decision.options[int(raw)]
            print("Invalid choice, try again.")


def _print_board(observation: Observation) -> None:
    print(f"\nBoard (DEFCON {observation.defcon}, VP {observation.vp}, phase {observation.phase}):")
    for country, infl in sorted(observation.influence.items()):
        us, ussr = infl.get("US", 0), infl.get("USSR", 0)
        if us or ussr:
            print(f"  {country}: US={us} USSR={ussr}")
    print(
        f"Space race: US={observation.space_race.get('US', 0)} "
        f"USSR={observation.space_race.get('USSR', 0)}"
    )
    print(
        f"Military ops: US={observation.military_ops.get('US', 0)} "
        f"USSR={observation.military_ops.get('USSR', 0)}"
    )
    if observation.turn_effects:
        print(f"Turn effects: {dict(observation.turn_effects)}")
    if observation.game_effects:
        print(f"Game effects: {dict(observation.game_effects)}")
    print(
        f"Hand: {list(observation.hand)} "
        f"(opponent holds {observation.opponent_hand_size} card(s))"
    )


def _print_history(history: Sequence[Event]) -> None:
    if not history:
        print("\nNo moves yet.")
        return
    print(f"\nFull history ({len(history)} event(s)):")
    for event in history:
        print(_format_event(event))
