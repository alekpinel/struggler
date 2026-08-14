"""Builds the two kinds of prompt content `LLMPlayer` needs: a static
system prompt (game-invariant, built once) and a per-call user turn
(what's new since the last real LLM call, the current board state, and
the live decision to act on).
"""

from __future__ import annotations

import json
from typing import Sequence

from struggler.bots.llm.event_summaries import EVENT_MECHANICAL_SUMMARIES
from struggler.bots.llm.schema import PAYLOAD_KEY_BY_KIND, PLAYER_FACING_KINDS
from struggler.engine import Action, Decision, Observation
from struggler.engine.cards import load_cards
from struggler.engine.data_loader import load_json
from struggler.engine.player import Event
from struggler.engine.rules import RULES


def _payload_catalog_text() -> str:
    lines = ["Decision kind -> the payload key you must fill in for that kind's step:"]
    for kind in PLAYER_FACING_KINDS:
        key = PAYLOAD_KEY_BY_KIND.get(kind)
        if key is None:  # EVENT_RESUME: always single-option, never actually reaches you
            continue
        lines.append(f"  - {kind.value}: payload.{key}")
    return "\n".join(lines)


def _cards_text() -> str:
    cards = load_cards()
    lines = []
    for cid, card in sorted(cards.items(), key=lambda kv: kv[1].number):
        mechanic = EVENT_MECHANICAL_SUMMARIES.get(cid)
        event_text = mechanic if mechanic else "not implemented (playing it as 'event' is a no-op discard)"
        lines.append(
            f"  {cid} (#{card.number}): ops={card.ops} side={card.side.value} "
            f"period={card.period.value} scoring={card.scoring} "
            f"remove_after_event={card.remove_after_event} | event: {event_text}"
        )
    return "\n".join(lines)


_system_prompt_cache: str | None = None


def build_system_prompt() -> str:
    """The static part of the conversation: hard constraints, the payload
    catalog, and every card's mechanical facts. Depends on nothing
    per-call, so it's built once and cached."""
    global _system_prompt_cache
    if _system_prompt_cache is not None:
        return _system_prompt_cache

    countries = load_json("countries.json")

    parts = [
        "You are playing Twilight Struggle (GMT Games) as one seat in a "
        "deterministic rules engine. Follow these hard constraints:",
        "1. You are always shown the LIVE legal options for the current "
        "decision. Never invent an action outside what's offered -- describe "
        "your intended choice (country/card/mode/etc.) and it will be matched "
        "against the real options for you.",
        "2. Hidden information is simply absent from what you're shown (the "
        "opponent's hand contents, the draw pile's order/contents). Never ask "
        "about it or assume a value for it -- reason only from what's given.",
        "3. A game turn decomposes into many small decisions (an 'atomic "
        "action space' -- e.g. each point of Influence placed is its own "
        "decision). You may predict several of your own upcoming decisions in "
        "one response to save round-trips, but STOP predicting the moment the "
        "next decision could depend on a dice roll's outcome, the opponent's "
        "choice, or anything else not yet resolved. A short or single-step "
        "plan is always safe; guessing wrong on a longer plan just costs one "
        "extra request later, it never breaks anything.",
        "4. Dice are rolled by the engine's own seeded RNG before you're ever "
        "consulted again -- you never roll dice yourself and never see a "
        "decision asking you to.",
        _payload_catalog_text(),
        f"Rules constants (JSON):\n{json.dumps(RULES, indent=2)}",
        f"Countries (JSON):\n{json.dumps(countries, indent=2)}",
        f"Cards (mechanical facts only -- no printed flavor/event text is "
        f"available to you, only what's summarized below):\n{_cards_text()}",
    ]
    _system_prompt_cache = "\n\n".join(parts)
    return _system_prompt_cache


def _decision_to_text(decision: Decision) -> str:
    def option_text(action: Action) -> str:
        return json.dumps(dict(action.payload))

    options = "\n".join(f"    - {option_text(a)}" for a in decision.options)
    return (
        f"Pending decision id={decision.id} kind={decision.kind.value} "
        f"context={dict(decision.context)}\n  Live options:\n{options}"
    )


def _observation_to_text(observation: Observation) -> str:
    nonzero_influence = {
        cid: infl
        for cid, infl in observation.influence.items()
        if infl.get("US") or infl.get("USSR")
    }
    payload = {
        "side": observation.side.value,
        "phase": observation.phase,
        "defcon": observation.defcon,
        "vp": observation.vp,
        "turn": observation.turn,
        "action_round": observation.action_round,
        "influence": nonzero_influence,
        "hand": list(observation.hand),
        "opponent_hand_size": observation.opponent_hand_size,
        "draw_pile_size": observation.draw_pile_size,
        "discard_pile": list(observation.discard_pile),
        "removed_cards": list(observation.removed_cards),
        "china_card_owner": observation.china_card_owner.value,
        "china_card_available": observation.china_card_available,
        "space_race": dict(observation.space_race),
        "military_ops": dict(observation.military_ops),
        "turn_effects": dict(observation.turn_effects),
        "game_effects": dict(observation.game_effects),
    }
    return json.dumps(payload, indent=2)


def _event_to_text(event: Event) -> str:
    payload: dict[str, object] = {
        "actor": event.actor.value,
        "decision_kind": event.decision.kind.value,
        "action": dict(event.action.payload),
        "defcon": event.defcon,
        "vp": event.vp,
        "turn": event.turn,
        "action_round": event.action_round,
    }
    if event.country is not None:
        payload["country"] = event.country
        payload["country_influence"] = dict(event.country_influence)
        payload["country_control"] = event.country_control
    return json.dumps(payload)


def build_user_turn(
    observation: Observation, decision: Decision, new_events: Sequence[Event]
) -> str:
    """The per-call, non-static part of the conversation: what happened
    since the last time this bot actually consulted the LLM, the current
    board state, and the live decision to act on."""
    parts = []
    if new_events:
        events_text = "\n".join(f"  - {_event_to_text(e)}" for e in new_events)
        parts.append(f"Since your last request ({len(new_events)} event(s)):\n{events_text}")
    parts.append(f"Current observation:\n{_observation_to_text(observation)}")
    parts.append(_decision_to_text(decision))
    parts.append(
        "Respond with a decision_plan: the first step MUST match the pending "
        "decision above (same kind, and a payload identifying one of the "
        "listed live options). Further steps are your own prediction of what "
        "you'll be asked next, per constraint 3 above."
    )
    return "\n\n".join(parts)
