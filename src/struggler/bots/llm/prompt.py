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
from struggler.engine import Action, Decision, DecisionKind, Observation, Side
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


_COMMON_GUIDANCE = [
    "STRATEGIC GUIDANCE (heuristics, not rules):",
    "  - Use events to open regions or make drastic changes, if not, usually Ops are a better use of cards.",
    "  - Only Battleground coups degrade DEFCON. Non-BG coups are free tempo and still count as Military Ops.",
    "  - 1 or 2 Stability battleground are cheap to coup and that's usually the best option if you have the opportunity.",
    "  - Never trigger a DEFCON-degrading Event on your own AR at DEFCON 2. Same for opponent Events that hand them Ops.",
    "  - Plan every turn to space one card: one with a nasty rival event but without high ops that makes it not worth it using as Ops.",
    "  - Deck reshuffles on turns 3 and 7. From turn 7 on, discarding is removal.",
    "  - Resolve the opponent Event first, then spend the Ops to repair it.",
    "  - Military Ops requirement = DEFCON number; 1 VP to the opponent per point short. Coups and war Events count, Realignments don't.",
    "  - `vp` negative = USSR lead (auto-win at -20); positive = US lead (+20).",
    "  - Presence/Domination/Control are step functions. One Influence that crosses a threshold beats three that don't.",
    "  - Breaking control without taking it wastes the round.",
    "  - Don't overstack a country you control past one coup's worth of margin.",
    "  - Scan 0-0 countries for reachable empty Battlegrounds. Cheapest VP on the board.",
    "  - Controlling all of Europe wins outright when Europe Scoring is played.",
    "  - Turns 1-3 only Asia, Europe and Middle East Scoring exist. Mid War regions arrive on turn 4.",
]

_USSR_GUIDANCE = [
    "USSR-SPECIFIC GUIDANCE:",
    "  - Standard initial influence: 4 Poland, 1 East Germany, 1 Yugoslavia. Controls both, has access to Yugoslavia.",
    "  - You act first every round: take the turn's Battleground coup and lock DEFCON at 2 before the US can.",
    "  - Final Scoring favors the US. aim to win by Mid War or a turn-8 Wargames.",
    "  - Turn 1 AR1 is coup Iran or play for Italy whatever is weaker.",
    "  - Coup big. A weak coup the US can reverse is worse than none.",
    "  - Best turn-1 headlines: Red Scare/Purge, Suez Crisis, Arab-Israeli War, Socialist Governments, Vietnam Revolts.",
    "  - Suez (or a won Arab-Israeli War) plus a good Iran coup erases the US from the Middle East.",
    "  - Early targets: Greece/Turkey, Egypt and Libya via Nasser, Jordan or Lebanon, then east into Pakistan and India.",
    "  - Don't put 1 Op into Pakistan at DEFCON 4+. You can't coup back and you hand the US a target.",
    "  - China Card is your 4 Ops (5 if every Op goes to Asia). Holding it lets you hold an extra card -- your insurance against DEFCON-suicide hands.",
    "  - You have no natural access to the Americas. Don't invest there without an access Event.",
]

_US_GUIDANCE = [
    "US-SPECIFIC GUIDANCE:",
    "  - Standard initial influence: 4 West Germany, 3 Italy. Controls both.",
    "  - Play the long game: Final Scoring favors you. Survive the Early War.",
    "  - Losing one region is fine; losing all of them isn't. Cut losses where Ops can't repair and wait for an Event.",
    "  - Military Ops is your weak spot. Plan a non-Battleground coup (Colombia, Syria) just to meet the DEFCON requirement if you couldn't coup before.",
    "  - Turn 1: Jordan or Lebanon to shield Israel, Egypt toward Libya before Nasser, Malaysia toward Thailand, Greece for access.",
    "  - Counter-coup Iran only if their coup was weak. Prefer holding the turn's last coup over the second-to-last.",
    "  - Stay out of South Korea until Korean War is spent or you hold Japan. Triggering Korean War yourself early is usually right.",
    "  - Trigger USSR starred Events while they're cheap: Korean War, Nasser, Warsaw Pact, Comecon, Blockade.",
    "  - AR7 (AR6 on turns 1-3): make a play the USSR must answer on their first round next turn.",
    "  - NATO only works after Marshall Plan or Warsaw Pact. Check before relying on it.",
    "  - Never play NORAD or NATO for the Event.",
]


def _strategic_guidance_text(side: Side) -> str:
    lines = list(_COMMON_GUIDANCE)
    lines += _USSR_GUIDANCE if side is Side.USSR else _US_GUIDANCE
    return "\n".join(lines)


_system_prompt_cache: dict[Side, str] = {}


def build_system_prompt(side: Side) -> str:
    """The static part of the conversation: hard constraints, the payload
    catalog, and every card's mechanical facts, plus strategic guidance for
    the given `side` only (its own-side guidance never leaks to the other
    seat). Depends on nothing else per-call, so it's built once per side and
    cached."""
    cached = _system_prompt_cache.get(side)
    if cached is not None:
        return cached

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
        _strategic_guidance_text(side),
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
    text = "\n\n".join(parts)
    _system_prompt_cache[side] = text
    return text


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
