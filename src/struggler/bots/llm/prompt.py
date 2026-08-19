"""Builds the two kinds of prompt content `LLMPlayer` needs: a static
system prompt (game-invariant, built once) and a per-call user turn
(what's new since the last real LLM call, the current board state, and
the live decision to act on).
"""

from __future__ import annotations

import json
from typing import Sequence

from struggler.bots.llm import card_playbook
from struggler.bots.llm.board_report import build_board_report
from struggler.bots.llm.rules_primer import RULES_PRIMER
from struggler.bots.llm.schema import PAYLOAD_KEY_BY_KIND, PLAYER_FACING_KINDS
from struggler.engine import Action, Decision, DecisionKind, Observation, Side
from struggler.engine.types import CardSide
from struggler.engine.cards import action_rounds, load_cards
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
        event_text = card.event_summary or "not implemented (playing it as 'event' is a no-op discard)"
        lines.append(
            f"  {cid} (#{card.number}): ops={card.ops} side={card.side.value} "
            f"period={card.period.value} scoring={card.scoring} "
            f"remove_after_event={card.remove_after_event} | event: {event_text}"
        )
    return "\n".join(lines)


_BATTLEGROUND_DOCTRINE = [
    "BATTLEGROUND DOCTRINE (the board report's 'BATTLEGROUND PRIORITIES' "
    "section is the live version of this -- read it every decision):",
    "  - VP comes from Battlegrounds. Presence/Domination/Control all hinge on "
    "who Controls more of them, and each one Controlled is +1 VP on top of the "
    "tier, +1 more if it is adjacent to the enemy superpower.",
    "  - Never leave a Battleground you have Influence in uncontrolled. A "
    "half-built Battleground scores nothing, is the target Truman Doctrine and "
    "Independent Reds are looking for, and is cheap for the opponent to finish.",
    "  - If the opponent breaks your Control of a Battleground, RETAKE IT on "
    "your next action round. A broken Battleground is a swing of at least 2 VP "
    "in every scoring of that region, and it only gets more expensive once they "
    "Control it (placement cost doubles).",
    "  - Finishing a contested Battleground beats opening a new "
    "non-Battleground, every time. Romania, Hungary and Greece do not win games; "
    "Poland, East Germany, Iran and Egypt do.",
    "  - Before spending Ops anywhere, check the RETAKE and AT RISK lists in the "
    "board report. Only spend outside them when nothing there is affordable.",
    "  - Influence that crosses no threshold buys nothing. 'you need +N' in the "
    "board report is the only number that matters: spend N, or spend elsewhere.",
]

_COMMON_GUIDANCE = [
    "STRATEGIC GUIDANCE (heuristics, not rules):",
    "  - Use events to open regions or make drastic changes, if not, usually Ops are a better use of cards.",
    "  - At DEFCON 3 or higher, a legal Battleground coup is your default best play.",
    "  - Reserve non-Battleground coups for when DEFCON is already at 2"
    "  - 1 or 2 Stability battlegrounds are especially cheap to coup, don't coup countries with more stability.",
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
    "  - A Scoring card in your hand (or one you know is still coming, e.g. the mandatory "
    "Asia/Europe/Middle East cards on turns 1-3) tells you exactly where to spend the turn's Ops: "
    "that region is this turn's priority over opening new fronts elsewhere. It must be played the "
    "turn it's drawn, and your opponent can never play it out of your hand, so there's no race to "
    "beat them to it -- but there's also no reason to sit on it once you're not gaining more by "
    "waiting. Improve the region first if there's still cheap ground to gain, then play the card as "
    "soon as your position there is locked in. That's often your last action round, but don't treat "
    "'last AR no matter what' as the goal -- holding it past the point of diminishing returns only "
    "exposes the region to an opponent Event flipping it back before you cash in.",
    "  - If one region has been scored recently, it's less likely to reapear, focus on regions that will score in the near future.",
    "  - Your own cards and NEUTRAL cards never fire an opponent Event when you play "
    "them for Ops. Space Racing one 'so the opponent doesn't get the event' is a "
    "misread -- only the OPPONENT's own cards carry that risk.",
    "  - One Space Race attempt per turn (two only from the box the rules name). The "
    "board report states how many you have left: check it BEFORE choosing a card to "
    "discard there, because the card is committed before the mode is.",
    "  - A big NEUTRAL Ops card (Nuclear Test Ban, Red Scare/Purge, ABM Treaty) is worth "
    "more as Ops or a coup than as a headline or a Space Race discard.",
    "  - Answer the opponent, not just your own plan. Every opponent placement listed "
    "under OPPONENT ACTIVITY is a claim you either answer now or concede.",
    "  - Meet the Military Operations requirement every turn. It equals DEFCON, it is "
    "checked at end of turn, and every point short is 1 VP to your opponent -- a "
    "non-Battleground coup pays it without touching DEFCON.",
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
    "  - Early targets: Egypt and Libya via Nasser, then east into Pakistan and India, reach to Thailand.",
    "  - Don't put 1 Op into Pakistan at DEFCON 4+. You can't coup back and you hand the US a target.",
    "  - China Card is your 4 Ops (5 if every Op goes to Asia). Holding it lets you hold an extra card -- your insurance against DEFCON-suicide hands.",
    "  - You have no natural access to the Americas. Don't invest there without an access Event.",
    "  - Don't take Romania with ops, you will get it with the card Romanian Adbication.",
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
    lines = list(_BATTLEGROUND_DOCTRINE)
    lines.append("")
    lines += _COMMON_GUIDANCE
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
        "Countries (JSON). Each entry has: region, subregion (null, a single "
        "name, or -- for Austria and Finland, which the rules count as both "
        "Western and Eastern Europe -- a list of two names), "
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


def _effective_ops(card, observation: Observation, side: Side) -> int:
    """This side's Ops value for `card` under the turn's active modifiers.

    An estimate of the engine's own `_effective_ops`, not a second source of
    truth: it exists so the hand dossier can say "3 Ops (2 after Red Scare)"
    instead of making the model rediscover the modifier every decision. The
    live `Decision.options` remain what is actually legal.
    """
    ops = card.ops
    effects = observation.turn_effects
    if effects.get("containment") and side is Side.US:
        ops += 1
    if effects.get("brezhnev") and side is Side.USSR:
        ops += 1
    if effects.get("red_scare") == side.value:
        ops -= 1
    return max(1, ops)


def _space_race_note(card, observation: Observation, effective_ops: int) -> str:
    """Whether this card could be sent to the Space Race right now, and if
    not, why not. The card is committed one decision BEFORE the mode is
    chosen, so "can I actually Space Race this?" has to be answerable while
    picking the card, not after."""
    side = observation.side
    pos = observation.space_race.get(side.value, 0)
    if pos >= RULES["space_race_max_box"]:
        return "space race: NO (already on the last box)"
    used = observation.space_race_attempts.get(side.value, 0)
    allowed = 2 if pos >= RULES["space_race_two_attempts_from_box"] else 1
    if used >= allowed:
        return f"space race: NO ({used}/{allowed} attempts already used this turn)"
    required = RULES["space_race_boxes"][str(pos + 1)]["ops"]
    if effective_ops < required:
        return f"space race: NO (next box needs {required} effective Ops, this card has {effective_ops})"
    return f"space race: yes ({allowed - used} attempt(s) left this turn)"


def _hand_text(observation: Observation) -> str:
    """The dossier for the cards actually in hand: mechanical facts, this
    turn's real Ops value, whether a Space Race play is even available, and
    the playbook's advice for this seat. The full card catalog in the system
    prompt covers every card that exists; this covers the handful the next
    decision is actually about."""
    side = observation.side
    cards = load_cards()
    lines = [
        "YOUR HAND (mechanical facts, this turn's effective Ops, and advice for "
        "your seat). 'star' means the card is removed from the game once its "
        "Event fires:"
    ]
    if not observation.hand:
        lines.append("  (empty)")
    for cid in observation.hand:
        card = cards.get(cid)
        if card is None:  # a physical-mode placeholder, or unknown id
            lines.append(f"  {cid}: (no card data)")
            continue
        if card.scoring:
            # A Scoring card has no Ops and no Space Race use at all, so the
            # Ops/Space-Race columns would only be noise -- or worse, an Ops
            # figure it does not have.
            lines.append(
                f"  {cid}: SCORING CARD -- no Ops and no Space Race use; it must be "
                "played this turn, for its Event"
            )
        else:
            ops = _effective_ops(card, observation, side)
            ops_text = f"{ops} effective Ops" + (
                f" (printed {card.ops})" if ops != card.ops else ""
            )
            owner = {
                CardSide.NEUTRAL: "NEUTRAL event (never fires for either side on an Ops play)",
                CardSide(side.value): "YOUR event",
                CardSide(side.opponent.value): (
                    "OPPONENT'S event (fires even when you play it for Ops)"
                ),
            }[card.side]
            header = f"  {cid}: {ops_text} | {owner}"
            if card.remove_after_event:
                header += " | star (removed after its Event)"
            header += f" | {_space_race_note(card, observation, ops)}"
            lines.append(header)
        if not card.scoring:
            lines.append(
                f"      event: {card.event_summary or 'not implemented (event play is a no-op discard)'}"
            )
        advice = card_playbook.advice_for(cid, side)
        if card.scoring:
            advice = f"{card_playbook.scoring_card_rule()} {advice}" if advice else card_playbook.scoring_card_rule()
        if advice:
            lines.append(f"      advice: {advice}")
    return "\n".join(lines)


def _cards_in_play_text(observation: Observation) -> str:
    return (
        "CARDS IN PLAY: China Card held by "
        f"{observation.china_card_owner.value} "
        f"({'face up, playable' if observation.china_card_available else 'face down, not playable this turn'})"
        f"\n  Discard pile ({len(observation.discard_pile)}): "
        f"{', '.join(observation.discard_pile) or 'empty'}"
        f"\n  Removed from the game ({len(observation.removed_cards)}): "
        f"{', '.join(observation.removed_cards) or 'none'}"
    )


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


def _events_text(new_events: Sequence[Event]) -> str | None:
    if not new_events:
        return None
    events_text = "\n".join(f"  - {_event_to_text(e)}" for e in new_events)
    return f"Since your last request ({len(new_events)} event(s)):\n{events_text}"


def _situation_text(
    observation: Observation, new_events: Sequence[Event], history: Sequence[Event] = ()
) -> list[str]:
    """The shared body of every user turn: what just happened, the derived
    board reading, the hand dossier, and what's left in the deck. Identical
    for a decision call and a turn-planning call -- the model reasons from
    one picture of the game, not two.

    `history` is the whole game's events so far, not just `new_events` (the
    delta since the last call) -- `build_board_report` needs the full game
    to answer "when was this region last scored?"."""
    parts = []
    events_text = _events_text(new_events)
    if events_text:
        parts.append(events_text)
    parts.append(build_board_report(observation, new_events, history))
    parts.append(_hand_text(observation))
    parts.append(_cards_in_play_text(observation))
    return parts


def build_history_entry(new_events: Sequence[Event]) -> str:
    """What a user turn actually leaves behind in the persisted conversation,
    once the live call it was sent for is done.

    `_situation_text`'s board report / hand dossier / cards-in-play are a
    snapshot of one instant -- true for the live call that carried them,
    stale (and pure token cost) for every later call that would otherwise
    resend it verbatim forever. The event delta is the only part of a user
    turn that stays true for the rest of the game, so it's the only part
    that gets persisted; every live call recomputes the rest fresh from the
    current `Observation` instead of trusting an old copy.
    """
    return _events_text(new_events) or "(no new events since your last request)"


def build_turn_plan_request(
    observation: Observation, new_events: Sequence[Event], history: Sequence[Event] = ()
) -> str:
    """The user turn for the once-per-game-turn planning call. No decision is
    pending as far as this call is concerned -- it produces intent only, which
    every decision in the turn is then made against."""
    parts = _situation_text(observation, new_events, history)
    total_ars = action_rounds(observation.turn)
    remaining_ars = max(1, total_ars - observation.action_round + 1)
    parts.append(
        f"This is the start of YOUR turn {observation.turn}. Before you make any "
        "decision, plan the whole turn: respond with a turn_plan.\n"
        f"  - ROUND BUDGET: this turn has {total_ars} action round(s) in total; you "
        f"are at action round {observation.action_round}, so you have {remaining_ars} "
        "action round(s) left in which to play a card (one card per action round, "
        "the China Card counts as one of them if you play it). You cannot play more "
        f"cards than that this turn -- if your hand holds more than {remaining_ars} "
        "non-scoring cards, the rest must be marked intended_use='hold' in card_plan, "
        "not scheduled into a round that doesn't exist.\n"
        "  - Work through the board report first: which regions can still be won, "
        "which Battlegrounds are contested, what the opponent took last turn.\n"
        "  - Decide your region_focus for this turn, in priority order: any region "
        "whose Scoring card is in your hand comes first (it must be played this "
        "turn regardless of anything else). If you hold no Scoring card, prioritize "
        "the region(s) with the oldest 'last scored' turn in the regional scoring "
        "status above -- 'never' outranks every turn number. A region scored "
        "recently is unlikely to be scored again soon, so Ops spent there are "
        "usually wasted this turn.\n"
        "  - Assign every card in your hand a use. A card with no plan is a card "
        "played badly later. Set each card_plan 'order' to when you intend to play "
        f"it: 0 if it's this turn's headline, 1, 2, 3... up to {remaining_ars} for "
        "the sequence you'll spend your remaining action rounds in, or -1 for a card "
        "you intend to hold rather than play this turn. Two cards played this turn "
        "must not share the same order.\n"
        "  - Any Scoring card in hand must be played this turn: decide which action "
        "round, and what has to change in that region first.\n"
        "  - Name the Ops that meet the Military Operations requirement (equal to "
        "DEFCON) -- being short is a guaranteed VP payment.\n"
        "  - List the Battlegrounds you must hold or retake.\n"
        "  - Write contingencies for the opponent plays that would break the plan.\n"
        "You are not choosing an action now. Nothing in this plan is executed "
        "directly; it is the intent you will be held to for the rest of the turn."
    )
    return "\n\n".join(parts)


def build_user_turn(
    observation: Observation,
    decision: Decision,
    new_events: Sequence[Event],
    turn_plan_text: str | None = None,
    history: Sequence[Event] = (),
) -> str:
    """The per-call, non-static part of the conversation: what happened
    since the last time this bot actually consulted the LLM, the derived
    board reading, this turn's standing plan, and the live decision to act
    on."""
    parts = _situation_text(observation, new_events, history)
    if turn_plan_text:
        parts.append(turn_plan_text)
    parts.append(_decision_to_text(decision))
    parts.append(
        "Respond with a decision_plan: the first step MUST match the pending "
        "decision above (same kind, and a payload identifying one of the "
        "listed live options). Further steps are your own prediction of what "
        "you'll be asked next, per constraint 3 above. State in the "
        "justification how this serves your plan for the turn -- and if it "
        "departs from the plan, say why the board changed."
    )
    return "\n\n".join(parts)
