"""OperatorConsolePlayer: the console interface for physical-mode games.

In physical mode (see `Engine.new_game(physical_mode=..., physical_side=...)`
and docs/BOTS.md) one seat is a real human playing
the physical board game. The operator — the person at the table running
this tool — is the single source of truth for everything the engine can't
know on its own: both hands' card-by-card dealing, every dice roll (both
sides'), and the physical side's own moves, typed in after they happen on
the real board. `runner.play_game` routes all of that to one
`OperatorConsolePlayer` instance, registered under both `players[physical_side]`
and `players[Side.CHANCE]`. The bot/AI side's own strategic decisions are
untouched — they still go to its own ordinary `Player`.

Reuses `HumanPlayer`'s prompt/format helpers (`human.py`) rather than
duplicating them; only the "too many options for a numbered list" case
(mandate #2's documented physical-mode exception) needs its own free-text
matching.
"""

from __future__ import annotations

from typing import Sequence

from struggler.engine.cards import load_cards
from struggler.engine.human import _format_action, _format_event, _print_board, _print_history
from struggler.engine.player import Event, Player
from struggler.engine.types import Action, DecisionKind, Observation

_CARDS = load_cards()

# Above this many options, a numbered list is unusable at a console; switch
# to free-text matching by the card's printed number or a name substring.
_FREE_TEXT_THRESHOLD = 15


def _match_key(action: Action) -> str:
    """The single payload value an option is matched against: a card id for
    every physical-mode wide-option decision (DEAL_CARD, ACTION_ROUND_PLAY,
    HEADLINE_PLAY, RANDOM_DISCARD, QUAGMIRE_DISCARD, HELD_CARD_DISCARD,
    hand-sourced EVENT_CHOICE, ...), or a non-card keyword like "refuse" —
    each of these decisions carries exactly one payload key."""
    return next(iter(action.payload.values()))


def _display_label(value: str) -> str:
    card = _CARDS.get(value)
    if card is None:
        return value  # a non-card keyword, e.g. "refuse"/"none"/"stop"
    return f"#{card.number} {card.name} (Ops {card.ops})"


def _matches(raw: str, value: str) -> bool:
    if raw == value.lower():
        return True
    card = _CARDS.get(value)
    if card is None:
        return False
    return raw == str(card.number) or raw in card.name.lower().replace("_", " ")


def _print_event_summary(card_id: str) -> None:
    card = _CARDS.get(card_id)
    if card is None:
        print(f"\nNo event text for {card_id!r} (not a card).")
        return
    summary = card.event_summary or "not implemented (playing it as event is a no-op discard)"
    print(f"\n#{card.number} {card.name} (Ops {card.ops}, {card.side.value}) event: {summary}")


class OperatorConsolePlayer:
    """Console `Player` for the operator of a physical-mode game.

    Only ever returns an `Action` object taken verbatim from
    `observation.pending_decision.options` — same contract as `HumanPlayer`.
    """

    def __init__(self) -> None:
        self._last_seen = 0

    def choose_action(self, observation: Observation, history: Sequence[Event]) -> Action:
        new_events = history[self._last_seen :]
        self._last_seen = len(history)
        if new_events:
            print(f"\nSince you were last asked ({len(new_events)} event(s)):")
            for event in new_events:
                print(_format_event(event))

        decision = observation.pending_decision
        print(
            f"\n[operator] turn {observation.turn}, action round {observation.action_round}, "
            f"DEFCON {observation.defcon}, VP {observation.vp}"
        )
        print(
            f"Decision: {decision.kind.value} for {decision.actor.value} "
            f"(context: {dict(decision.context)})"
        )

        if len(decision.options) > _FREE_TEXT_THRESHOLD:
            return self._prompt_free_text(decision.options, observation, history)
        return self._prompt_numbered(decision.options, observation, history)

    def _prompt_numbered(
        self, options: tuple[Action, ...], observation: Observation, history: Sequence[Event]
    ) -> Action:
        card = observation.pending_decision.context.get("card")
        for i, action in enumerate(options):
            print(f"  [{i}] {_format_action(action)}")
        while True:
            prompt = f"Choose an option [0-{len(options) - 1}], or 'b' for board / 'h' for full history"
            if card is not None:
                prompt += " / 'e' for this card's event text"
            raw = input(prompt + ": ").strip().lower()
            if raw in ("b", "board"):
                _print_board(observation)
                continue
            if raw in ("h", "history"):
                _print_history(history)
                continue
            if raw in ("e", "event") and card is not None:
                _print_event_summary(card)
                continue
            if raw.isdigit() and int(raw) < len(options):
                return options[int(raw)]
            print("Invalid choice, try again.")

    def _prompt_free_text(
        self, options: tuple[Action, ...], observation: Observation, history: Sequence[Event]
    ) -> Action:
        print(
            f"({len(options)} option(s) — too many to list. Enter the physical "
            "card's printed number, or part of its name.)"
        )
        while True:
            raw = input(
                "Which card (number or name), or 'b' for board / 'h' for full history: "
            ).strip().lower()
            if raw in ("b", "board"):
                _print_board(observation)
                continue
            if raw in ("h", "history"):
                _print_history(history)
                continue
            matches = [a for a in options if _matches(raw, _match_key(a))]
            if len(matches) == 1:
                return matches[0]
            if not matches:
                print("No match, try again.")
                continue
            print(f"{len(matches)} matches, be more specific:")
            for a in matches[:10]:
                print(f"  {_display_label(_match_key(a))}")


class BotHeadlineAnnouncer:
    """Wraps the bot side's `Player` in physical mode so its Headline pick
    is printed the moment it's chosen.

    Without this, a bot's `HEADLINE_PLAY` only becomes visible to the
    operator via `OperatorConsolePlayer`'s "since you were last asked"
    recap -- and that recap only fires the next time the operator is
    prompted for *anything*, which is often well after the reveal step the
    operator needs to perform on the physical board right now (both
    `HEADLINE_PLAY` events are buffered together until the second pick is
    locked in; see docs/BOTS.md). Announcing it immediately here is what
    actually lets the operator place the bot's card at reveal time.

    Every other bot decision already surfaces this way in time to act on
    it, since the operator is prompted again before the physical board
    needs to reflect it -- only the Headline reveal is time-sensitive
    enough to need an out-of-band print.
    """

    def __init__(self, inner: Player) -> None:
        self._inner = inner

    def choose_action(self, observation: Observation, history: Sequence[Event]) -> Action:
        action = self._inner.choose_action(observation, history)
        if observation.pending_decision.kind is DecisionKind.HEADLINE_PLAY:
            cid = action.payload["card"]
            print(
                f"\n[bot headline] {_display_label(cid)} -- play this as the "
                "bot's Headline on the physical board."
            )
        return action
