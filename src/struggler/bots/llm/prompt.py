"""Builds the two kinds of prompt content `LLMPlayer` needs: a static
system prompt (game-invariant, built once) and a per-call user turn
(what's new since the last real LLM call, the current board state, and
the live decision to act on).
"""

from __future__ import annotations

import json
from typing import Sequence

from struggler.bots.llm.event_summaries import EVENT_MECHANICAL_SUMMARIES
from struggler.bots.llm.rules_primer import RULES_PRIMER
from struggler.bots.llm.schema import PAYLOAD_KEY_BY_KIND, PLAYER_FACING_KINDS
from struggler.engine import Action, Decision, DecisionKind, Observation
from struggler.engine.cards import load_cards
from struggler.engine.data_loader import load_json
from struggler.engine.player import Event
from struggler.engine.rules import RULES

# One-line semantic meaning per player-facing DecisionKind -- what the
# decision actually represents, not just its payload shape. Source of truth
# is the inline comments already next to each DecisionKind member in
# engine/types.py; kept here by hand since Python enums don't expose
# adjacent source comments at runtime.
_DECISION_KIND_MEANING: dict[DecisionKind, str] = {
    DecisionKind.PLACE_INFLUENCE: "place one Influence point in the given country (one atomic point per decision)",
    DecisionKind.COUP_TARGET: "pick which country to attempt a Coup against",
    DecisionKind.REALIGNMENT_TARGET: "pick which country to attempt a Realignment roll against",
    DecisionKind.HEADLINE_PLAY: "pick which card from your hand to headline this turn",
    DecisionKind.ACTION_ROUND_PLAY: "pick which card to play for this action round",
    DecisionKind.PLAY_MODE: (
        "choose how to use the card just played -- normally a choice between "
        "'ops' (spend its Ops), 'event' (fire its Event), and 'space_race' "
        "(discard it for a Space Race advance attempt, if eligible); "
        "'un_intervention' additionally appears only while holding the UN "
        "Intervention card"
    ),
    DecisionKind.OPS_TYPE: "choose how to spend this card's Ops",
    DecisionKind.EVENT_OPS_ORDER: "the opponent's card Event was triggered by your Ops play -- choose whether it resolves before or after your Ops",
    DecisionKind.WAR_TARGET: "pick the target country for a 'war' Event whose attacker chooses the target",
    DecisionKind.EVENT_INFLUENCE: "an Event-driven Influence placement/removal step -- pick the country",
    DecisionKind.EVENT_CHOICE: "pick one of this Event's branching sub-options",
    DecisionKind.QUAGMIRE_DISCARD: "discard an Ops-2+ card to try to break free from Bear Trap/Quagmire",
    DecisionKind.HELD_CARD_DISCARD: "optionally discard your Held Card at end of turn (Space Race box 6 ability)",
}


def _payload_catalog_text() -> str:
    lines = ["Decision kind -> what it means, and the payload key you must fill in for that kind's step:"]
    for kind in PLAYER_FACING_KINDS:
        key = PAYLOAD_KEY_BY_KIND.get(kind)
        if key is None:  # EVENT_RESUME: always single-option, never actually reaches you
            continue
        meaning = _DECISION_KIND_MEANING[kind]
        lines.append(f"  - {kind.value}: {meaning} -- payload.{key}")
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


_STRATEGIC_GUIDANCE = "\n".join(
    [
        "STRATEGIC GUIDANCE (heuristics worth weighing, not hard rules -- "
        "unlike the RULES PRIMER above, following these is a judgment call, "
        "not a legality requirement):",
        "  - Coups are not automatically dangerous: DEFCON only drops to "
        "the CURRENT value minus 1 per Coup attempt, so at DEFCON 3+, "
        "coup-ing a low-Stability (1-2) country is usually a strong, low-risk "
        "play -- don't default to Influence out of reflexive DEFCON "
        "caution; weigh a Coup explicitly every time you spend Ops.",
        "  - Space Race is a real alternative use for a card, not a "
        "fallback: if a card's Ops aren't needed for a strong Influence/"
        "Coup/Realignment play this turn, or its Event would help your "
        "OPPONENT, consider discarding it for a Space Race attempt instead "
        "of defaulting to Ops. Remember: playing an opponent's card for "
        "Ops does NOT stop their Event from firing (only its order changes) "
        "-- Space Race is the only mode that actually denies it, so if "
        "avoiding that Event is the whole point, choose Space Race, not Ops.",
        "  - Read `vp` number -- a NEGATIVE vp is a big USSR lead (approaching "
        "USSR's automatic win at -20); a large POSITIVE vp is a big US lead."
        "  - The Mid War deck (Central America, Southeast Asia, Africa, "
        "and South America Scoring, among other cards) only enters play "
        "starting turn 4 -- Asia, Europe, and Middle East Scoring are the "
        "only Scoring cards available turns 1-3. Weight your early "
        "Influence investment toward the regions already in play, and "
        "start shifting toward the Mid War regions as turn 4 approaches.",
        "  - The full board (every country, including ones at 0-0 "
        "Influence) is in every observation you're given -- actively check "
        "it for empty Battlegrounds you can already reach (adjacent to "
        "your home space or your existing Influence) before assuming a "
        "region is settled. An uncontested Battleground is often the "
        "cheapest VP on the board, and in Europe specifically, controlling "
        "every country wins the game outright the instant Europe is "
        "scored (see VICTORY above) -- don't leave that path unexplored "
        "just because the opponent hasn't shown up there.",
        "  - Diminishing returns on stacking: Ops spent adding Influence "
        "to a country you already Control, beyond what's needed to keep "
        "that Control safe from a single Coup/Realignment swing, usually "
        "produce less value than the same Ops spent expanding to a new "
        "country, on a Coup, or on a Space Race attempt.",
        "  - Track your own running Military Operations total against the "
        "current DEFCON over the course of a turn (both are in every "
        "observation). Falling short at end of turn hands the opponent 1 "
        "VP per point of shortfall -- an easy accident if every Ops spend "
        "defaults to Influence and none to Coups or war Events.",
    ]
)

_system_prompt_cache: str | None = None


def build_system_prompt() -> str:
    """The static part of the conversation: hard constraints, the payload
    catalog, and every card's mechanical facts. Depends on nothing
    per-call, so it's built once and cached."""
    global _system_prompt_cache
    if _system_prompt_cache is not None:
        return _system_prompt_cache

    countries = load_json("countries.json")
    # The raw file carries dev-facing provenance/confidence notes
    # (_disclaimer, _confirmed_against_physical_board, _uncertain,
    # _setup_influence_note) that are not meant for the model to reason
    # about -- only the game-facing keys are included in the dump.
    countries_for_model = {
        key: countries[key] for key in ("superpowers", "setup_influence", "countries")
    }

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
        RULES_PRIMER,
        _STRATEGIC_GUIDANCE,
        _payload_catalog_text(),
        f"Rules constants (JSON):\n{json.dumps(RULES, indent=2)}",
        "Countries (JSON). Each entry has: region, subregion (nullable), "
        "stability (used in the Control/Coup/Realignment formulas above), "
        "battleground (true/false, used in regional scoring), and "
        "adjacent_to (country ids this country connects to for placement "
        "reachability, Coup/Realignment eligibility bonuses, and event "
        "targeting). \"US\" and \"USSR\" also appear as adjacency pseudo-"
        "nodes representing each superpower's home space -- a country "
        "adjacent to one of them is always a legal Influence-placement "
        f"target for that side:\n{json.dumps(countries_for_model, indent=2)}",
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
    payload = {
        "side": observation.side.value,
        "phase": observation.phase,
        "defcon": observation.defcon,
        "vp": observation.vp,
        "turn": observation.turn,
        "action_round": observation.action_round,
        "influence": dict(observation.influence),
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
