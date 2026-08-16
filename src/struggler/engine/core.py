from __future__ import annotations

import copy
import random

from struggler.engine.board import Board
from struggler.engine.cards import action_rounds, cards_entering, hand_limit, load_cards
from struggler.engine.events import EVENTS
from struggler.engine.rules import RULES
from struggler.engine.types import (
    Action,
    Card,
    Decision,
    DecisionKind,
    Observation,
    Period,
    Region,
    ScoringTier,
    Side,
    Subregion,
)

_DEFAULT_MIN_DEFCON = 1

# Every coup attempt, in any region, degrades DEFCON by 1, regardless of
# success. Realignment is subject to the same COUP_MIN_DEFCON restriction above (8.1.5 restricts "Coup or
# Realignment rolls" identically) — enforced in _usable_coup_realign_target.
SCORING_CARD_REGION: dict[str, Region] = {
    "Asia_Scoring": Region.ASIA,
    "Europe_Scoring": Region.EUROPE,
    "Middle_East_Scoring": Region.MIDDLE_EAST,
    "Central_America_Scoring": Region.CENTRAL_AMERICA,
    "Africa_Scoring": Region.AFRICA,
    "South_America_Scoring": Region.SOUTH_AMERICA,
}


class Engine:
    def __init__(self, seed: int, board: Board | None = None) -> None:
        self.board = board if board is not None else Board()
        self.defcon = 5
        self.vp = 0  # US-positive: >0 favors US, <0 favors USSR (matches score_region)
        self.turn = 1
        self.action_round = 1

        self._seed = seed
        self._rng = random.Random(seed)
        self._decision_stack: list[Decision] = []
        self._next_decision_id = 0
        self._winner: Side | None = None
        self._game_over_reason: str | None = None
        self.cards: dict[str, Card] = load_cards()
        self.phase = "idle"  # idle | headline | action_rounds | complete
        self.include_optional = False
        self.draw_pile: list[str] = []
        self.discard_pile: list[str] = []
        self.removed_cards: list[str] = []
        self.hands: dict[str, list[str]] = {"US": [], "USSR": []}
        self.china_card_owner = "USSR"
        self.china_card_available = True  # face-up: playable by its owner this turn
        self.space_race: dict[str, int] = {"US": 0, "USSR": 0}
        self.space_race_attempts: dict[str, int] = {"US": 0, "USSR": 0}  # this turn
        self.military_ops: dict[str, int] = {"US": 0, "USSR": 0}
        self._ars_played = 0  # completed action-round plays this turn (both sides)
        self._headline: dict[str, str | None] = {"US": None, "USSR": None}
        # Headline resolution is stack-driven so an event fired at the headline
        # can enqueue sub-decisions (e.g. a war's CHANCE roll) that must drain
        # before the second headline card resolves. `_headline_resolving` marks
        # that both cards are chosen and the frozen `_headline_pending` order
        # ([side, cid] pairs, higher Ops first) is being worked through.
        self._headline_resolving = False
        self._headline_pending: list[list[str]] = []

        # -- Card-event state --------------------------------------------
        # `events_enabled` gates the whole event layer: False reproduces M2
        # (every card is Ops-only, no event ever fires). `turn_effects` holds
        # persistent per-turn modifiers set by events (e.g. Containment) and is
        # cleared at end of turn; its values are JSON primitives (mandate #5).
        self.events_enabled = False
        self.turn_effects: dict[str, object] = {}
        # Our Man in Tehran reveals the top few draw-pile cards to the US one at
        # a time; the not-yet-decided cards wait here (serialized state, mandate
        # #5) but are deliberately *not* surfaced by observe(), so only the card
        # currently being decided is exposed — the rest of the draw order stays
        # hidden (mandate #4).
        self._our_man_queue: list[str] = []
        self._our_man_kept: list[str] = []
        # Persistent effects that last for the rest of the *game* (not just the
        # turn): NATO and similar "USSR may no longer coup/realign X" locks, and
        # the flags they depend on. Never cleared at end of turn. JSON-native
        # values only (mandate #5).
        self.game_effects: dict[str, object] = {}
        # Rule 6.1.1: within one Operations spend, every placed point must be
        # adjacent to friendly markers that were in place at the *start* of
        # the phasing player's Action Round -- not markers placed earlier in
        # the same spend. Frozen the moment a placement chain begins (see
        # _maybe_push_place_influence / _maybe_push_bonus_influence), reused
        # for every point in that chain, and cleared once it ends. None
        # outside of an Ops-driven placement chain.
        self._ops_round_snapshot: dict[str, dict[str, int]] | None = None

    # -- public API -------------------------------------------------------

    @property
    def pending_decision(self) -> Decision | None:
        return self._decision_stack[-1] if self._decision_stack else None

    def legal_actions(self) -> tuple[Action, ...]:
        decision = self.pending_decision
        return decision.options if decision is not None else ()

    def step(self, action: Action) -> None:
        if self.is_terminal:
            raise RuntimeError("cannot step: game has ended")
        decision = self.pending_decision
        if decision is None:
            raise RuntimeError("cannot step: no pending decision")
        if action not in decision.options:
            raise ValueError(f"illegal action {action!r} for decision {decision!r}")
        self._decision_stack.pop()
        self._dispatch(decision, action)
        self._advance()

    def observe(self, player: Side) -> Observation:
        if player not in (Side.US, Side.USSR):
            raise ValueError("observe() is only valid for Side.US or Side.USSR")
        opponent = player.opponent
        return Observation(
            side=player,
            phase=self.phase,
            defcon=self.defcon,
            vp=self.vp,
            turn=self.turn,
            action_round=self.action_round,
            influence=copy.deepcopy(self.board.influence),
            pending_decision=self.pending_decision,
            # Own hand in full; the opponent's hand only as a count (mandate
            # #4). The draw pile is a count too — its order never leaks.
            hand=tuple(self.hands[player.value]),
            opponent_hand_size=len(self.hands[opponent.value]),
            draw_pile_size=len(self.draw_pile),
            discard_pile=tuple(self.discard_pile),
            removed_cards=tuple(self.removed_cards),
            china_card_owner=Side(self.china_card_owner),
            china_card_available=self.china_card_available,
            space_race=dict(self.space_race),
            military_ops=dict(self.military_ops),
            # Public M3 modifiers only (e.g. NATO, Containment) — the
            # in-progress secret headline pick lives on `self._headline`
            # and is never surfaced here.
            turn_effects=copy.deepcopy(self.turn_effects),
            game_effects=copy.deepcopy(self.game_effects),
        )

    @property
    def is_terminal(self) -> bool:
        # A finished game either has a winner or ended in a draw (phase
        # 'complete' with no winner); both leave pending_decision None.
        return self._winner is not None or self.phase == "complete"

    @property
    def winner(self) -> Side | None:
        return self._winner

    def serialize(self) -> dict:
        return {
            "seed": self._seed,
            "rng_state": _encode_rng_state(self._rng.getstate()),
            "board": self.board.serialize(),
            "defcon": self.defcon,
            "vp": self.vp,
            "turn": self.turn,
            "action_round": self.action_round,
            "next_decision_id": self._next_decision_id,
            "decision_stack": [_encode_decision(d) for d in self._decision_stack],
            "winner": self._winner.value if self._winner is not None else None,
            "game_over_reason": self._game_over_reason,
            # -- M2 full-game state --
            "phase": self.phase,
            "include_optional": self.include_optional,
            "draw_pile": list(self.draw_pile),
            "discard_pile": list(self.discard_pile),
            "removed_cards": list(self.removed_cards),
            "hands": {side: list(cards) for side, cards in self.hands.items()},
            "china_card_owner": self.china_card_owner,
            "china_card_available": self.china_card_available,
            "space_race": dict(self.space_race),
            "space_race_attempts": dict(self.space_race_attempts),
            "military_ops": dict(self.military_ops),
            "ars_played": self._ars_played,
            "headline": dict(self._headline),
            "headline_resolving": self._headline_resolving,
            "headline_pending": [list(pair) for pair in self._headline_pending],
            "events_enabled": self.events_enabled,
            "turn_effects": dict(self.turn_effects),
            "game_effects": dict(self.game_effects),
            "our_man_queue": list(self._our_man_queue),
            "our_man_kept": list(self._our_man_kept),
            "ops_round_snapshot": (
                copy.deepcopy(self._ops_round_snapshot)
                if self._ops_round_snapshot is not None
                else None
            ),
        }

    @classmethod
    def deserialize(cls, data: dict) -> "Engine":
        engine = cls(seed=data["seed"])
        engine._rng.setstate(_decode_rng_state(data["rng_state"]))
        engine.board.load_influence(data["board"])
        engine.defcon = data["defcon"]
        engine.vp = data["vp"]
        engine.turn = data["turn"]
        engine.action_round = data["action_round"]
        engine._next_decision_id = data["next_decision_id"]
        engine._decision_stack = [_decode_decision(d) for d in data["decision_stack"]]
        engine._winner = Side(data["winner"]) if data["winner"] is not None else None
        engine._game_over_reason = data["game_over_reason"]
        # -- M2 full-game state (absent in pre-M2 logs: fall back to M1 sandbox) --
        engine.phase = data.get("phase", "idle")
        engine.include_optional = data.get("include_optional", False)
        engine.draw_pile = list(data.get("draw_pile", []))
        engine.discard_pile = list(data.get("discard_pile", []))
        engine.removed_cards = list(data.get("removed_cards", []))
        hands = data.get("hands", {"US": [], "USSR": []})
        engine.hands = {side: list(cards) for side, cards in hands.items()}
        engine.china_card_owner = data.get("china_card_owner", "USSR")
        engine.china_card_available = data.get("china_card_available", True)
        engine.space_race = dict(data.get("space_race", {"US": 0, "USSR": 0}))
        engine.space_race_attempts = dict(
            data.get("space_race_attempts", {"US": 0, "USSR": 0})
        )
        engine.military_ops = dict(data.get("military_ops", {"US": 0, "USSR": 0}))
        engine._ars_played = data.get("ars_played", 0)
        engine._headline = dict(data.get("headline", {"US": None, "USSR": None}))
        engine._headline_resolving = data.get("headline_resolving", False)
        engine._headline_pending = [list(pair) for pair in data.get("headline_pending", [])]
        engine.events_enabled = data.get("events_enabled", False)
        engine.turn_effects = dict(data.get("turn_effects", {}))
        engine.game_effects = dict(data.get("game_effects", {}))
        engine._our_man_queue = list(data.get("our_man_queue", []))
        engine._our_man_kept = list(data.get("our_man_kept", []))
        snapshot = data.get("ops_round_snapshot")
        engine._ops_round_snapshot = copy.deepcopy(snapshot) if snapshot is not None else None
        return engine

    # -- M1 test-harness entry points (no cards yet) -----------------------

    def begin_influence_operations(self, side: Side, ops: int) -> None:
        if ops <= 0:
            raise ValueError("ops must be positive")
        self._maybe_push_place_influence(side, ops)

    def begin_coup(self, side: Side, ops: int, bonus: str | None = None) -> None:
        if ops <= 0:
            raise ValueError("ops must be positive")
        options = self._coup_target_options(side)
        if not options:
            return
        self._push(side, DecisionKind.COUP_TARGET, options, {"ops": ops, "bonus": bonus})

    def begin_realignment_operations(self, side: Side, ops: int) -> None:
        if ops <= 0:
            raise ValueError("ops must be positive")
        self._maybe_push_realignment_target(side, card_ops=ops, spent=0)

    # -- M2: full-game entry point -----------------------------------------

    @classmethod
    def new_game(
        cls,
        seed: int,
        include_optional: bool = True,
        board: Board | None = None,
        events: bool = True,
    ) -> "Engine":
        """Start a complete game: build the Early War deck, deal opening
        hands, and push the first (USSR) headline decision.

        Opening setup runs first: printed at-start influence is applied, then
        the USSR places 6 additional Influence in Eastern Europe and the US 7
        in Western Europe (as ordinary placement decisions), before the turn-1
        headline.
        """
        engine = cls(seed=seed, board=board)
        engine.include_optional = include_optional
        engine.events_enabled = events
        engine.china_card_owner = "USSR"
        engine.china_card_available = True
        engine.turn = 1
        engine._start_turn()  # build Early War deck, deal (phase set to headline)
        engine._begin_setup()  # apply/choose opening influence (phase -> setup)
        engine._advance()      # runs once setup completes: push the first headline
        return engine

    # -- M2: turn director --------------------------------------------------

    def _advance(self) -> None:
        """Push the next top-level decision whenever the stack drains.

        The decision stack holds only genuine pending choices; between them
        (start of an action round, headline resolution, end of turn) this
        director decides what happens next. It is a no-op in the M1 sandbox
        (phase 'idle') and once the game is over ('complete').
        """
        # 'setup' self-drives through its own placement handler (it always
        # leaves a decision pending until it flips the phase to 'headline'),
        # so the director only takes over from the headline onward.
        if self.phase in ("idle", "complete", "setup"):
            return
        while not self._decision_stack and not self.is_terminal:
            self._advance_once()
            if self.phase in ("idle", "complete", "setup"):
                return

    def _advance_once(self) -> None:
        if self.phase == "headline":
            if not self._headline_resolving:
                if self._headline["USSR"] is None:
                    self._push_headline(Side.USSR)
                    return
                if self._headline["US"] is None:
                    self._push_headline(Side.US)
                    return
                # Both cards are chosen: freeze the resolution order and clear
                # the picks so they live in exactly one place from here on.
                order = self._headline_resolution_order()
                order = self._apply_defectors_headline(order)
                self._headline_pending = order
                self._headline = {"US": None, "USSR": None}
                self._headline_resolving = True
            if self._headline_pending:
                # Resolve one headline card. If its event enqueues sub-decisions
                # they land on the stack; _advance stops until they drain, then
                # returns here for the next card (that is the interrupt order).
                side_str, cid = self._headline_pending.pop(0)
                self._resolve_headline_card(Side(side_str), cid)
                return
            self._headline_resolving = False
            self._begin_action_rounds()
            return

        if self.phase == "action_rounds":
            total = self._total_action_rounds()
            if self._ars_played >= total:
                self._end_of_turn()
                return
            idx = self._ars_played  # 0-based play index within the turn
            side = self._side_for_play_index(idx)
            self.action_round = idx // 2 + 1
            self._ars_played += 1
            if self.turn_effects.get("cuban_missile_crisis") == side.value:
                # "May defuse at any point in the turn": offered fresh at the
                # start of every one of this side's action rounds until they
                # take it or the turn ends -- a free choice, not a spent round.
                self._push_cmc_defuse_offer(side)
            else:
                self._dispatch_action_round(side)
            return

    # -- M2: turn boundaries ------------------------------------------------

    def _start_turn(self) -> None:
        """Add the period's cards to the deck (as the war escalates), deal
        both hands up to the limit, and enter the headline phase."""
        if self.turn == 1:
            self._add_period_to_deck(Period.EARLY_WAR)
        elif self.turn == 4:
            self._add_period_to_deck(Period.MID_WAR)
        elif self.turn == 8:
            self._add_period_to_deck(Period.LATE_WAR)
        self._deal_to_limit()
        self.phase = "headline"

    def _end_of_turn(self) -> None:
        # Required military operations: a side that spent fewer military Ops
        # (coups) than the current DEFCON hands the deficit to its opponent.
        for side in (Side.US, Side.USSR):
            deficit = self.defcon - self.military_ops[side.value]
            if deficit > 0:
                self._award_vp(side.opponent, deficit)
                if self.is_terminal:
                    return
        # We Will Bury You: the USSR scores 3 VP at the end of the turn unless
        # the US cancelled it (by playing UN Intervention, see _handle_play_mode).
        if self.turn_effects.get("we_will_bury_you"):
            self._award_vp(Side.USSR, 3)
            if self.is_terminal:
                return
        # DEFCON recovers by one at the end of every turn.
        self._change_defcon(+1, caused_by=Side.US)
        # A China Card passed this turn becomes available to its new owner.
        self.china_card_available = True
        # Reset per-turn accounting.
        self.military_ops = {"US": 0, "USSR": 0}
        self.space_race_attempts = {"US": 0, "USSR": 0}
        self._headline = {"US": None, "USSR": None}
        # Persistent per-turn event modifiers (Containment, Red Scare, ...) last
        # only "for the remainder of the turn"; they lapse here.
        self.turn_effects = {}

        if self._push_held_card_discard():
            return  # HELD_CARD_DISCARD pending; _advance_past_turn_boundary resumes it
        self._advance_past_turn_boundary()

    def _advance_past_turn_boundary(self) -> None:
        if self.turn >= 10:
            self._finish_game()
            return
        self.turn += 1
        self._start_turn()

    def _push_held_card_discard(self) -> bool:
        """Space Race box 6 (Eagle/Bear has Landed): its sole holder may
        discard their Held Card before the next deal, instead of carrying it
        into next turn. Returns True iff a decision was pushed."""
        holder = self.game_effects.get("space_race_discard_holder")
        if holder is None:
            return False
        side = Side(holder)
        hand = self.hands[side.value]
        if not hand:
            return False
        options = tuple(
            Action(DecisionKind.HELD_CARD_DISCARD, {"card": cid}) for cid in hand
        ) + (Action(DecisionKind.HELD_CARD_DISCARD, {"card": "none"}),)
        self._push(side, DecisionKind.HELD_CARD_DISCARD, options, {})
        return True

    def _handle_held_card_discard(self, decision: Decision, action: Action) -> None:
        cid = action.payload["card"]
        if cid != "none":
            self._file_card(decision.actor, cid, fired=False)
        self._advance_past_turn_boundary()

    def _finish_game(self) -> None:
        """End the game after turn 10: final-score every region, then decide
        on total VP (a 0 VP board is a draw).

        Final scoring reuses the same board mechanic as the scoring cards, so
        every region contributes its Presence/Domination/Control tier once.
        """
        for region in Region:
            self._change_vp_by(self._score_region_net(region))
            if self.is_terminal:  # a VP-20 swing or Europe control ends it here
                return
        if self.vp > 0:
            self._win(Side.US, "final_vp")
        elif self.vp < 0:
            self._win(Side.USSR, "final_vp")
        else:
            self.phase = "complete"  # draw: terminal with no winner
            self._decision_stack.clear()

    def _begin_action_rounds(self) -> None:
        self.phase = "action_rounds"
        self._ars_played = 0
        self.action_round = 1

    def _extra_action_round_sides(self) -> tuple[Side, ...]:
        """Sides granted an extra Action Round this turn, beyond the normal
        alternating rounds, in the order those rounds are played: North Sea
        Oil grants the US one for this turn only; Space Race box 8 (Space
        Station) grants its sole holder one every turn for as long as it
        holds the ability (6.4.3-6.4.4)."""
        extra: list[Side] = []
        if self.turn_effects.get("north_sea_oil_extra"):
            extra.append(Side.US)
        holder = self.game_effects.get("space_race_extra_round_holder")
        if holder is not None:
            extra.append(Side(holder))
        return tuple(extra)

    def _total_action_rounds(self) -> int:
        """Total card plays this turn across both sides: normally 2*N (N per
        side), plus one per currently-held extra-round source."""
        return 2 * action_rounds(self.turn) + len(self._extra_action_round_sides())

    def _side_for_play_index(self, idx: int) -> Side:
        """Whose play the 0-based `idx` is. The base rounds alternate USSR,
        US, USSR, ...; any extra rounds beyond the base go to
        _extra_action_round_sides(), in order."""
        base = 2 * action_rounds(self.turn)
        if idx < base:
            return Side.USSR if idx % 2 == 0 else Side.US
        return self._extra_action_round_sides()[idx - base]

    # -- M2: opening setup --------------------------------------------------

    def _begin_setup(self) -> None:
        """Apply printed at-start influence, then hand control to the players
        for the additional Eastern/Western Europe placement."""
        for side_str, influence in self.board.setup_influence.items():
            for cid, amount in influence.items():
                self.board.influence[cid][side_str] += amount
        self.phase = "setup"
        self._push_setup_influence(Side.USSR, Subregion.EASTERN_EUROPE)

    def _push_setup_influence(self, side: Side, subregion: Subregion) -> None:
        remaining = RULES["setup_additional"][subregion.name]["amount"]
        self._push_setup_influence_remaining(side, subregion, remaining)

    def _push_setup_influence_remaining(
        self, side: Side, subregion: Subregion, remaining: int
    ) -> None:
        # Setup placement is free within the region (reachability does not
        # apply), so every country in the subregion is a legal target.
        options = tuple(
            Action(DecisionKind.PLACE_INFLUENCE, {"country": cid})
            for cid, info in self.board.countries.items()
            if info.subregion is subregion
        )
        self._push(
            side,
            DecisionKind.PLACE_INFLUENCE,
            options,
            {"setup": True, "side": side.value, "subregion": subregion.value,
             "remaining": remaining},
        )

    # -- M2: deck operations (shuffle via the injected RNG, mandate #3) -----

    def _add_period_to_deck(self, period: Period) -> None:
        entering = cards_entering(self.cards, period, self.include_optional)
        self.draw_pile.extend(entering)
        self._rng.shuffle(self.draw_pile)

    def _deal_to_limit(self) -> None:
        limit = hand_limit(self.turn)
        order = ("USSR", "US")  # USSR is the first player; VERIFY deal order
        while any(len(self.hands[s]) < limit for s in order):
            progressed = False
            for s in order:
                if len(self.hands[s]) < limit:
                    card = self._draw_card()
                    if card is None:
                        return  # draw + discard exhausted (should not happen)
                    self.hands[s].append(card)
                    progressed = True
            if not progressed:
                return

    def _draw_card(self) -> str | None:
        if not self.draw_pile:
            self._reshuffle_discard_into_draw()
        if not self.draw_pile:
            return None
        return self.draw_pile.pop()

    def _reshuffle_discard_into_draw(self) -> None:
        # Removed cards stay out of the game; only the discard is recycled.
        self.draw_pile = self.discard_pile
        self.discard_pile = []
        self._rng.shuffle(self.draw_pile)

    # -- M2: headline phase -------------------------------------------------

    def _push_headline(self, side: Side) -> None:
        # The China Card cannot be headlined; scoring cards can.
        options = tuple(
            Action(DecisionKind.HEADLINE_PLAY, {"card": cid})
            for cid in self.hands[side.value]
        )
        if options:
            self._push(side, DecisionKind.HEADLINE_PLAY, options, {})

    def _handle_headline_play(self, decision: Decision, action: Action) -> None:
        side = decision.actor
        cid = action.payload["card"]
        self.hands[side.value].remove(cid)
        self._headline[side.value] = cid

    def _headline_resolution_order(self) -> list[list[str]]:
        """Freeze the order the two headlined cards resolve in: higher Ops
        first, ties US-first. Returns [side_value, card_id] pairs. (In M2 only
        scoring events act, so order is cosmetic; with events on it decides
        which event — and its interrupts — happens first.)"""
        picks = {s: self._headline[s.value] for s in (Side.US, Side.USSR)}
        order = sorted(
            (Side.US, Side.USSR),
            key=lambda s: (-self.cards[picks[s]].ops, s is not Side.US),
        )
        return [[s.value, picks[s]] for s in order]

    def _apply_defectors_headline(self, order: list[list[str]]) -> list[list[str]]:
        """Defectors, headlined by the US, cancels the USSR's headline — the
        USSR card is discarded without resolving (Defectors always acts first,
        so this happens before any card in `order` resolves). If the USSR ever
        headlines Defectors (only reachable via Missile Envy handing it over),
        the US instead gains 1 VP. Only meaningful with the event layer on;
        without it both headlines are no-op discards anyway."""
        if not self.events_enabled:
            return order
        picks = {side_str: cid for side_str, cid in order}
        if picks.get("US") == "Defectors":
            ussr_card = picks.get("USSR")
            if ussr_card is not None:
                self._file_card(Side.USSR, ussr_card, fired=False)
                order = [pair for pair in order if pair[0] != "USSR"]
        elif picks.get("USSR") == "Defectors":
            self._award_vp(Side.US, 1)
        return order

    def _maybe_flower_power(self, side: Side, cid: str) -> None:
        """Flower Power: the USSR scores 2 VP each time the US plays a war card
        (for its Event or Operations), until An Evil Empire cancels it."""
        if side is Side.US and cid in RULES["war_cards"] and self.game_effects.get("flower_power"):
            self._award_vp(Side.USSR, 2)

    def _resolve_headline_card(self, side: Side, cid: str) -> None:
        """Resolve one headlined card for its owner. A scoring card scores; with
        events on, a card with an implemented event fires it (and may enqueue
        sub-decisions); otherwise it is a no-op discard (M2 behavior)."""
        self._maybe_flower_power(side, cid)
        if self.is_terminal:
            return
        card = self.cards[cid]
        if card.scoring:
            self._resolve_scoring_card(cid)
            self._file_card(side, cid, fired=True)
        elif self.events_enabled and self._has_event(cid):
            self._file_card(side, cid, fired=True)
            self._fire_event(side, cid)
        else:
            self._file_card(side, cid, fired=False)

    # -- M2: action round: pick a card, then how to use it ------------------

    def _dispatch_action_round(self, side: Side) -> None:
        """`side`'s next action round: a trap step if it's caught in one,
        otherwise its ordinary card play."""
        trap_key = self._trap_key_for(side)
        if trap_key is not None:
            self._push_trap_step(side, trap_key)
        else:
            self._push_action_round_play(side)

    def _push_cmc_defuse_offer(self, side: Side) -> None:
        """Cuban Missile Crisis: `side` may remove 2 Influence from Cuba
        (USSR) or West Germany/Turkey (US, its choice) to lift the
        Coup-attempt ban for the rest of the turn. Offered as a free choice
        -- it never costs the action round that follows it."""
        candidates = ["Cuba"] if side is Side.USSR else ["West_Germany", "Turkey"]
        eligible = [c for c in candidates if self.board.influence[c][side.value] >= 2]
        if not eligible:
            self._dispatch_action_round(side)
            return
        self.push_event_choice(
            "Cuban_Missile_Crisis_defuse", side, tuple(eligible) + ("skip",)
        )

    def _push_action_round_play(self, side: Side) -> None:
        hand = self.hands[side.value]
        scoring_in_hand = [cid for cid in hand if self.cards[cid].scoring]
        # A scoring card may not be held past the end of the turn. Once a side
        # has as many scoring cards as it has action rounds left, every
        # remaining round must spend one (the China Card is not offered then).
        must_play_scoring = bool(scoring_in_hand) and len(
            scoring_in_hand
        ) >= self._remaining_action_rounds(side)

        # Missile Envy: "next round your opponent must use this card for
        # Operations" -- the scoring deadline still takes priority if both
        # apply at once, since carrying a scoring card past end of turn isn't
        # legal at all.
        forced_missile_envy = (
            not must_play_scoring
            and self.game_effects.get("missile_envy_forced") == side.value
            and "Missile_Envy" in hand
        )

        if forced_missile_envy:
            playable = ["Missile_Envy"]
        else:
            playable = scoring_in_hand if must_play_scoring else list(hand)
        options = [
            Action(DecisionKind.ACTION_ROUND_PLAY, {"card": cid}) for cid in playable
        ]
        if (
            not must_play_scoring
            and not forced_missile_envy
            and side.value == self.china_card_owner
            and self.china_card_available
        ):
            options.append(Action(DecisionKind.ACTION_ROUND_PLAY, {"card": RULES["china_card_id"]}))
        if options:
            self._push(side, DecisionKind.ACTION_ROUND_PLAY, tuple(options), {})

    def _handle_missile_envy_forced_play(self, side: Side, cid: str) -> None:
        self.game_effects.pop("missile_envy_forced", None)
        self._file_card(side, cid, fired=False)
        self._push_ops_type(side, self._effective_ops(side, self.cards[cid]))

    def _remaining_action_rounds(self, side: Side) -> int:
        """Action rounds `side` still has this turn, including the one now
        being set up. (`side` is always the side whose play is current.)

        Counted from the current play index onward so it stays correct with an
        asymmetric extra round in play (North Sea Oil)."""
        total = self._total_action_rounds()
        start = self._ars_played - 1  # current 0-based play index within the turn
        if start < 0:
            return 0
        return sum(1 for i in range(start, total) if self._side_for_play_index(i) is side)

    def _handle_action_round_play(self, decision: Decision, action: Action) -> None:
        side = decision.actor
        cid = action.payload["card"]
        if cid == "Missile_Envy" and self.game_effects.get("missile_envy_forced") == side.value:
            self._handle_missile_envy_forced_play(side, cid)
            return
        self.push_full_card_play(side, cid)

    def push_full_card_play(self, side: Side, cid: str) -> None:
        """Offer `side` the ordinary Event/Ops/Space-Race choice for `cid`,
        exactly as if it were their action-round card play. Used both for a
        normal action round and for Grain Sales to Soviets' "play the card"
        outcome (a card taken from the opponent's hand is played in full,
        not just for a fixed Ops amount) -- the card need not already be in
        `side`'s hand; `_file_card` tolerates that."""
        modes = self._play_modes(side, cid)
        options = tuple(Action(DecisionKind.PLAY_MODE, {"mode": m}) for m in modes)
        self._push(side, DecisionKind.PLAY_MODE, options, {"card": cid})

    def _play_modes(self, side: Side, cid: str) -> tuple[str, ...]:
        card = self.cards[cid]
        if card.scoring:
            return ("event",)  # a scoring card can only be played as its event
        modes = ["ops"]
        # The event-vs-ops choice is enumerated per the M2 spec even though no
        # non-scoring event fires yet (choosing it is a no-op discard). The
        # China Card has no event, so it is Ops-only.
        if cid != RULES["china_card_id"]:
            modes.append("event")
        if self._can_space_race(side, card):
            modes.append("space_race")
        # UN Intervention: if this is an opponent's (implemented, eligible) event
        # card and the player is holding UN Intervention, they may play the card
        # for Ops with its event cancelled (discarding UN Intervention).
        if (
            self.events_enabled
            and cid != RULES["un_intervention_id"]
            and self._is_opponent_event(side, card)
            and self._has_event(cid)
            and EVENTS[cid].eligible(self, side)
            and RULES["un_intervention_id"] in self.hands[side.value]
        ):
            modes.append("un_intervention")
        return tuple(modes)

    def _handle_play_mode(self, decision: Decision, action: Action) -> None:
        side = decision.actor
        cid = decision.context["card"]
        card = self.cards[cid]
        mode = action.payload["mode"]

        if mode in ("event", "ops", "un_intervention"):
            self._maybe_flower_power(side, cid)
            if self.is_terminal:
                return

        if mode == "event":
            if card.scoring:
                self._resolve_scoring_card(cid)
                self._file_card(side, cid, fired=True)
            elif self.events_enabled and self._has_event(cid):
                # Playing a card for its (implemented) event: it fires now and
                # the card leaves play (removed if remove_after_event).
                self._file_card(side, cid, fired=True)
                self._fire_event(side, cid)
            else:
                # M2 behavior: an unfired/unimplemented event is a no-op discard.
                self._file_card(side, cid, fired=False)
            return

        if mode == "un_intervention":
            # Cancel the opponent card's event; use it purely for its Ops. UN
            # Intervention itself is spent to the discard pile. Playing it also
            # defuses We Will Bury You's end-of-turn VP for the US.
            if side is Side.US:
                self.turn_effects.pop("we_will_bury_you", None)
            self.hands[side.value].remove(RULES["un_intervention_id"])
            self.discard_pile.append(RULES["un_intervention_id"])
            self._file_card(side, cid, fired=False)  # event cancelled: normal discard
            self._push_ops_type(side, self._effective_ops(side, card))
            return

        if mode == "space_race":
            self._file_card(side, cid, fired=False)
            self.space_race_attempts[side.value] += 1
            roll = self._roll_d6()
            self._push(
                Side.CHANCE,
                DecisionKind.SPACE_RACE_ROLL,
                (Action(DecisionKind.SPACE_RACE_ROLL, {"value": roll}),),
                {"side": side.value},
            )
            return

        # mode == "ops": the card's Ops value drives an influence/coup/realign.
        ops = self._effective_ops(side, card)
        if (
            self.events_enabled
            and self._is_opponent_event(side, card)
            and self._has_event(cid)
            and EVENTS[cid].eligible(self, side)
        ):
            # An opponent's event also fires when their card is played for Ops;
            # the phasing player picks the order (event first or Ops first).
            self._file_card(side, cid, fired=True)
            self._push(
                side,
                DecisionKind.EVENT_OPS_ORDER,
                (
                    Action(DecisionKind.EVENT_OPS_ORDER, {"order": "event_first"}),
                    Action(DecisionKind.EVENT_OPS_ORDER, {"order": "ops_first"}),
                ),
                {"side": side.value, "card": cid, "ops": ops},
            )
            return
        self._file_card(side, cid, fired=False)  # China Card passes here
        self._push_ops_type(side, ops, china=(cid == RULES["china_card_id"]))

    def _push_ops_type(
        self, side: Side, ops: int, china: bool = False, allow_coup: bool = True
    ) -> None:
        self._push(
            side, DecisionKind.OPS_TYPE, self._ops_type_options(side, ops, allow_coup),
            {"side": side.value, "ops": ops, "bonus": self._ops_bonus_region(side, china)},
        )

    def _ops_bonus_region(self, side: Side, china: bool) -> str | None:
        """The region a play earns its "+1 Op if all Ops used here" bonus in:
        "asia" for the China Card, "se_asia" for a USSR play while Vietnam
        Revolts is in effect this turn, else None. (China takes precedence; the
        rare China+Vietnam stack is not modeled.)"""
        if china:
            return "asia"
        if side is Side.USSR and self.turn_effects.get("vietnam_revolts"):
            return "se_asia"
        return None

    def _in_bonus_region(self, cid: str, bonus: str) -> bool:
        info = self.board.countries[cid]
        if bonus == "asia":
            return info.region is Region.ASIA
        if bonus == "se_asia":
            return info.subregion is Subregion.SOUTHEAST_ASIA
        return False

    def _ops_type_options(
        self, side: Side, ops: int, allow_coup: bool = True
    ) -> tuple[Action, ...]:
        types = []
        if self._place_influence_options(side, ops):
            types.append("influence")
        if allow_coup and self._coup_target_options(side):
            types.append("coup")
        if self._realignment_target_options(side):
            types.append("realignment")
        return tuple(Action(DecisionKind.OPS_TYPE, {"type": t}) for t in types)

    def _handle_ops_type(self, decision: Decision, action: Action) -> None:
        side = Side(decision.context["side"])
        ops = decision.context["ops"]
        bonus = decision.context.get("bonus")
        ops_type = action.payload["type"]
        if ops_type == "influence":
            if bonus:
                # A region-bonus play's +1 applies only if every Op is spent in
                # that region; the placement step enforces the all-or-nothing
                # rule (China Card -> Asia, Vietnam Revolts -> Southeast Asia).
                self._maybe_push_bonus_influence(side, ops, 0, 0, bonus)
            else:
                self._maybe_push_place_influence(side, ops)
        elif ops_type == "coup":
            # Coups count toward the turn's required military operations. A
            # region-bonus coup gets its +1 only against a target in that region
            # (resolved at target selection, in _handle_coup_target).
            self.military_ops[side.value] += ops
            self.begin_coup(side, ops, bonus=bonus)
        else:  # realignment
            # Region-bonus play (China Card -> Asia, Vietnam Revolts -> SE
            # Asia): one extra Realignment attempt, all-or-nothing, if every
            # attempt this Ops-spend targets a country inside the bonus
            # region — mirrors _maybe_push_bonus_influence's rule.
            self._maybe_push_realignment_target(side, card_ops=ops, spent=0, bonus=bonus)

    # -- M2: space race -----------------------------------------------------

    def _space_attempts_allowed(self, side: Side) -> int:
        if self.space_race[side.value] >= RULES["space_race_two_attempts_from_box"]:
            return 2
        return 1

    def _can_space_race(self, side: Side, card: Card) -> bool:
        pos = self.space_race[side.value]
        if pos >= RULES["space_race_max_box"]:
            return False
        if self._effective_ops(side, card) < RULES["space_race_boxes"][str(pos + 1)]["ops"]:
            return False
        return self.space_race_attempts[side.value] < self._space_attempts_allowed(side)

    def _handle_space_race_roll(self, decision: Decision, action: Action) -> None:
        side = Side(decision.context["side"])
        roll = action.payload["value"]
        next_box = self.space_race[side.value] + 1
        if roll <= RULES["space_race_boxes"][str(next_box)]["roll_max"]:
            self.advance_space_race_box(side)

    def advance_space_race_box(self, side: Side, award_vp: bool = True) -> None:
        """Move `side` one box up the Space Race track and award that box's VP
        (first vs second to reach it). Shared by a successful attempt roll and
        by events that advance the marker directly (e.g. Captured Nazi
        Scientist). No-op at the top of the track. `award_vp=False` still
        moves the marker and grants/cancels the box's ability, but withholds
        VP -- One Small Step's "get VP for the second step only"."""
        if self.space_race[side.value] >= RULES["space_race_max_box"]:
            return
        next_box = self.space_race[side.value] + 1
        box = RULES["space_race_boxes"][str(next_box)]
        first = self.space_race[side.opponent.value] < next_box
        self.space_race[side.value] = next_box
        vp = box["vp_first"] if first else box["vp_second"]
        if vp and award_vp:
            self._award_vp(side, vp)
        self._update_space_race_ability(side, next_box, first)

    def _update_space_race_ability(self, side: Side, box: int, first: bool) -> None:
        """6.4.4: a Space Race special ability is granted only to the first
        side to reach its box, and is cancelled outright (not transferred)
        the instant the second side also reaches it. Only the abilities
        modeled as a held flag are keyed here: box 6 (may discard the Held
        Card) and box 8 (an extra Action Round). Box 2's double-attempt
        ability is instead a direct position check (_space_attempts_allowed)
        and box 4's headline-order perk remains unmodeled."""
        key = RULES["space_race_ability_keys"].get(str(box))
        if key is None:
            return
        if first:
            self.game_effects[key] = side.value
        else:
            self.game_effects.pop(key, None)

    # -- M3: card events ----------------------------------------------------

    def _has_event(self, cid: str) -> bool:
        return cid in EVENTS

    def _is_opponent_event(self, side: Side, card: Card) -> bool:
        """Whether `card`'s event belongs to `side`'s opponent (so playing it
        for Ops also fires the event). NEUTRAL cards never trigger this way."""
        if card.side.value not in ("US", "USSR"):
            return False
        return Side(card.side.value) is side.opponent

    def _effective_ops(self, side: Side, card: Card) -> int:
        """The card's Ops value for `side` after persistent per-turn modifiers
        (Containment/Brezhnev +1, Red Scare -1). Never below 1."""
        ops = card.ops
        if self.turn_effects.get("containment") and side is Side.US:
            ops += 1
        if self.turn_effects.get("brezhnev") and side is Side.USSR:
            ops += 1
        if self.turn_effects.get("red_scare") == side.value:
            ops -= 1
        return max(1, ops)

    def _fire_event(self, side: Side, cid: str) -> None:
        """Resolve `cid`'s event for the phasing `side`. Unimplemented events
        fizzle (a no-op discard already happened at the call site); an
        implemented event whose precondition is unmet (e.g. NATO before
        Marshall Plan/Warsaw Pact) also does nothing."""
        ev = EVENTS.get(cid)
        if ev is not None and ev.eligible(self, side):
            ev.resolve(self, side)

    def _usable_coup_realign_target(
        self, attacker: Side, cid: str, for_coup: bool = True
    ) -> bool:
        """Whether `attacker` may attempt a Coup/Realignment against `cid`.

        Requires the opponent to hold at least 1 Influence there (6.2.1 /
        6.3.1) and the current DEFCON to allow it in that region (8.1.5,
        which restricts Coup *and* Realignment alike). Beyond that,
        persistent effects may lock the USSR out further (NATO protects
        US-controlled Europe; the US/Japan pact protects Japan; The Reformer
        bars USSR *coups* in Europe). NATO's lock is lifted per-country by
        De Gaulle (France) and Willy Brandt (West Germany)."""
        info = self.board.countries[cid]
        if self.board.influence[cid][attacker.opponent.value] <= 0:
            return False
        min_defcon = RULES["coup_min_defcon"].get(info.region.name, _DEFAULT_MIN_DEFCON)
        if self.defcon < min_defcon:
            return False
        if attacker is not Side.USSR:
            return True
        ge = self.game_effects
        region = info.region
        if ge.get("us_japan_pact") and cid == "Japan":
            return False
        if for_coup and ge.get("reformer") and region is Region.EUROPE:
            return False
        if self._nato_protects(cid):
            return False
        return True

    def _nato_protects(self, cid: str) -> bool:
        """Whether NATO currently shields `cid` from the USSR: "USA-controlled
        countries in Europe are protected from CCCP Coup attempts, CCCP
        Realignments, Brush War." Lifted per-country by De Gaulle (France) and
        Willy Brandt (West Germany)."""
        ge = self.game_effects
        info = self.board.countries[cid]
        if not (
            ge.get("nato")
            and info.region is Region.EUROPE
            and self.board.control(cid) is Side.US
        ):
            return False
        if cid == "France" and ge.get("degaulle_france"):
            return False
        if cid == "West_Germany" and ge.get("willy_brandt"):
            return False
        return True

    def _handle_event_ops_order(self, decision: Decision, action: Action) -> None:
        side = Side(decision.context["side"])
        cid = decision.context["card"]
        ops = decision.context["ops"]
        if action.payload["order"] == "event_first":
            depth = len(self._decision_stack)
            self._fire_event(side, cid)
            if len(self._decision_stack) > depth:
                # The event enqueued sub-decisions; run the Ops after they drain
                # by slipping a resume marker underneath them.
                self._decision_stack.insert(
                    depth,
                    self._new_decision(
                        side, DecisionKind.EVENT_RESUME,
                        (Action(DecisionKind.EVENT_RESUME, {}),),
                        {"what": "ops", "side": side.value, "ops": ops},
                    ),
                )
            elif not self.is_terminal:
                self._push_ops_type(side, ops)
        else:  # ops_first: Ops resolve, then the event fires afterward.
            depth = len(self._decision_stack)
            self._push_ops_type(side, ops)
            self._decision_stack.insert(
                depth,
                self._new_decision(
                    side, DecisionKind.EVENT_RESUME,
                    (Action(DecisionKind.EVENT_RESUME, {}),),
                    {"what": "event", "side": side.value, "card": cid},
                ),
            )

    def _handle_event_resume(self, decision: Decision, action: Action) -> None:
        ctx = decision.context
        side = Side(ctx["side"])
        if self.is_terminal:
            return
        if ctx["what"] == "ops":
            self._push_ops_type(side, ctx["ops"])
        else:  # "event"
            self._fire_event(side, ctx["card"])

    # -- M3: player-choice event steps (tier 2) -----------------------------
    #
    # An event that lets a player distribute influence enqueues its own
    # decisions through one generic, fully serializable step type
    # (EVENT_INFLUENCE): place, remove, or remove-all, one country at a time,
    # for `remaining` steps, honouring a per-country cap and control filters.
    # The step re-pushes itself until `remaining` hits 0 or no legal target is
    # left — so it lives on the same decision stack as everything else and is
    # hosted correctly inside a headline or an opponent's Ops play. A branch
    # ("remove OR add", e.g. Warsaw Pact) is a single EVENT_CHOICE first.

    def push_event_influence(
        self,
        event: str,
        op: str,                       # "place" | "remove"
        choose_side: Side,             # who makes the choices (the beneficiary)
        inf_side: Side,                # whose influence is placed/removed
        remaining: int,
        candidates: list[str],
        cap: int | None = None,
        whole: bool = False,           # remove: clear ALL inf_side in the country
        requires_uncontrolled: bool = False,
        exclude_controlled_by: Side | None = None,
        amount: int = 1,               # influence moved per selected country
    ) -> None:
        """Begin a player-choice influence sequence, if it has any legal step.

        Each selected country moves `amount` Influence (default 1); East European
        Unrest uses `amount=2` in the Late War, one selection per country."""
        context = {
            "event": event,
            "op": op,
            "choose_side": choose_side.value,
            "inf_side": inf_side.value,
            "remaining": remaining,
            "candidates": list(candidates),
            "cap": cap,
            "whole": whole,
            "requires_uncontrolled": requires_uncontrolled,
            "exclude_controlled_by": (
                exclude_controlled_by.value if exclude_controlled_by is not None else None
            ),
            "amount": amount,
            "placed": {},
        }
        self._maybe_push_event_influence(context)

    def _maybe_push_event_influence(self, context: dict) -> None:
        if context["remaining"] <= 0:
            return
        options = self._event_influence_options(context)
        if not options:
            return
        self._push(
            Side(context["choose_side"]), DecisionKind.EVENT_INFLUENCE, options, context
        )

    def _event_influence_options(self, context: dict) -> tuple[Action, ...]:
        inf_side = context["inf_side"]
        cap = context["cap"]
        whole = context["whole"]
        placed = context["placed"]
        exclude = context["exclude_controlled_by"]
        options = []
        for cid in context["candidates"]:
            used = placed.get(cid, 0)
            if context["op"] == "place":
                if cap is not None and used >= cap:
                    continue
                if exclude is not None and self.board.control(cid) is Side(exclude):
                    continue
            else:  # remove
                if self.board.influence[cid][inf_side] <= 0:
                    continue
                if whole:
                    if used >= 1:
                        continue
                elif cap is not None and used >= cap:
                    continue
            if context["requires_uncontrolled"] and self.board.control(cid) is not None:
                continue
            options.append(Action(DecisionKind.EVENT_INFLUENCE, {"country": cid}))
        return tuple(options)

    def _handle_event_influence(self, decision: Decision, action: Action) -> None:
        context = dict(decision.context)
        cid = action.payload["country"]
        inf_side = context["inf_side"]
        amount = context.get("amount", 1)
        if context["op"] == "place":
            self.board.influence[cid][inf_side] += amount
        elif context["whole"]:
            self.board.influence[cid][inf_side] = 0
        else:
            self.board.influence[cid][inf_side] = max(
                0, self.board.influence[cid][inf_side] - amount
            )
        placed = dict(context["placed"])
        placed[cid] = placed.get(cid, 0) + 1
        context["placed"] = placed
        context["remaining"] -= 1
        self._maybe_push_event_influence(context)

    def push_event_choice(
        self,
        event: str,
        choose_side: Side,
        choices: tuple[str, ...],
        extra: dict | None = None,
    ) -> None:
        """Offer a player a branch within an event (routed by events.py). `extra`
        merges extra JSON-native keys into the decision context, so a router can
        carry running state (e.g. Ask Not's discard count, Grain Sales' card)."""
        options = tuple(
            Action(DecisionKind.EVENT_CHOICE, {"choice": c}) for c in choices
        )
        context = {"event": event, "choose_side": choose_side.value}
        if extra:
            context.update(extra)
        self._push(choose_side, DecisionKind.EVENT_CHOICE, options, context)

    def _handle_event_choice(self, decision: Decision, action: Action) -> None:
        from struggler.engine.events import CHOICE_ROUTERS

        event = decision.context["event"]
        side = Side(decision.context["choose_side"])
        CHOICE_ROUTERS[event](self, side, action.payload["choice"], decision.context)

    # -- M3: influence / control helpers used by events ---------------------

    def add_influence(self, country: str, side: Side, amount: int) -> None:
        self.board.influence[country][side.value] += amount

    def remove_influence(self, country: str, side: Side, amount: int) -> None:
        current = self.board.influence[country][side.value]
        self.board.influence[country][side.value] = max(0, current - amount)

    def remove_all_influence(self, country: str, side: Side) -> None:
        self.board.influence[country][side.value] = 0

    def gain_control(self, country: str, side: Side) -> None:
        """Remove all opponent Influence in `country` and give `side` enough of
        its own for Control ("adds sufficient Influence for Control")."""
        self.board.influence[country][side.opponent.value] = 0
        stability = self.board.countries[country].stability
        if self.board.influence[country][side.value] < stability:
            self.board.influence[country][side.value] = stability

    # -- M3: events that grant "conduct Operations" -------------------------

    def push_event_operations(self, side: Side, ops: int, allow_coup: bool = True) -> None:
        """An event that has its beneficiary conduct `ops` Operations (CIA
        Created, Lone Gunman, ABM Treaty, ...). `allow_coup=False` restricts
        the spend to Influence/Realignment only (Glasnost, KAL-007's printed
        text names only those two, never Coup)."""
        if ops > 0:
            self._push_ops_type(side, ops, allow_coup=allow_coup)

    def set_defcon(self, level: int, caused_by: Side) -> None:
        """Set DEFCON to `level` (How I Learned to Stop Worrying, ...). Routed
        through _change_defcon so the DEFCON-1 loss condition still fires and
        the caller is blamed for it."""
        self._change_defcon(level - self.defcon, caused_by=caused_by)

    # -- M3: forced random discard from a hidden hand (a CHANCE decision) ----
    #
    # The discard is drawn from the seeded RNG and exposed as a CHANCE decision
    # with a *single* option — the drawn card (which is about to become public
    # in the discard pile). The rest of the hand is never enumerated, so no
    # hidden card leaks (mandate #4), and the log stays replayable (mandate #3).

    def push_random_discard(self, owner: Side, purpose: str, count: int = 1) -> None:
        hand = self.hands[owner.value]
        if count <= 0 or not hand:
            return
        card = hand[self._rng.randrange(len(hand))]
        self._push(
            Side.CHANCE,
            DecisionKind.RANDOM_DISCARD,
            (Action(DecisionKind.RANDOM_DISCARD, {"card": card}),),
            {"owner": owner.value, "purpose": purpose, "count": count},
        )

    def _handle_random_discard(self, decision: Decision, action: Action) -> None:
        ctx = decision.context
        owner = Side(ctx["owner"])
        card = action.payload["card"]
        if ctx["purpose"] == "five_year_plan":
            # A discarded USSR-associated event fires (even against the USSR's
            # own interest); anything else is just discarded.
            info = self.cards[card]
            if not info.scoring and info.side.value == owner.value and self._has_event(card):
                self._file_card(owner, card, fired=True)
                self._fire_event(owner, card)
            else:
                self._file_card(owner, card, fired=False)
        elif ctx["purpose"] == "grain_sales":
            # The revealed card is not filed yet: the opponent (US) decides to
            # take it (use its Ops, then discard) or return it (use Grain Sales'
            # own 2 Ops). It stays in the USSR hand until then.
            self.push_event_choice(
                "Grain_Sales_to_Soviets", owner.opponent, ("take", "return"),
                extra={"card": card},
            )
        else:  # plain forced discard (Terrorism), possibly repeated
            self._file_card(owner, card, fired=False)
            self.push_random_discard(owner, ctx["purpose"], ctx["count"] - 1)

    def draw_cards_to_hand(self, side: Side, n: int) -> None:
        """Draw `n` cards from the deck into `side`'s hand (Ask Not's redraw)."""
        for _ in range(n):
            card = self._draw_card()
            if card is None:
                break
            self.hands[side.value].append(card)

    # -- M3: a two-die "both roll, higher wins" contest ---------------------
    #
    # Both sides roll (from the seeded RNG, logged as a single CHANCE option),
    # a per-side modifier is added, and the winner takes `vp`. By default ties
    # reroll (Olympic Games); Summit's printed text says "do not reroll ties",
    # so `reroll_ties=False` makes a tie a wash instead: no VP, no resolver.
    # An optional per-event follow-up (events.CONTEST_RESOLVERS) runs after.

    def push_dice_contest(
        self,
        event: str,
        sponsor: Side,
        sponsor_mod: int,
        defender_mod: int,
        vp: int,
        reroll_ties: bool = True,
    ) -> None:
        s_roll, d_roll = self._roll_d6(), self._roll_d6()
        self._push(
            Side.CHANCE, DecisionKind.CONTEST_ROLL,
            (Action(DecisionKind.CONTEST_ROLL,
                    {"sponsor_roll": s_roll, "defender_roll": d_roll}),),
            {"event": event, "sponsor": sponsor.value,
             "sponsor_mod": sponsor_mod, "defender_mod": defender_mod, "vp": vp,
             "reroll_ties": reroll_ties},
        )

    def _handle_contest_roll(self, decision: Decision, action: Action) -> None:
        from struggler.engine.events import CONTEST_RESOLVERS

        ctx = decision.context
        sponsor = Side(ctx["sponsor"])
        s_total = action.payload["sponsor_roll"] + ctx["sponsor_mod"]
        d_total = action.payload["defender_roll"] + ctx["defender_mod"]
        if s_total == d_total:
            if ctx.get("reroll_ties", True):
                self.push_dice_contest(
                    ctx["event"], sponsor, ctx["sponsor_mod"], ctx["defender_mod"],
                    ctx["vp"], reroll_ties=True,
                )
            return  # not rerolling: a tie is a wash -- no VP, no resolver
        winner = sponsor if s_total > d_total else sponsor.opponent
        if ctx["vp"]:
            self._award_vp(winner, ctx["vp"])
        if self.is_terminal:
            return
        resolver = CONTEST_RESOLVERS.get(ctx["event"])
        if resolver is not None:
            resolver(self, sponsor, winner)

    def _regions_dominated(self, side: Side) -> int:
        """How many regions `side` Dominates or Controls (Summit's modifier)."""
        return sum(
            1
            for region in Region
            if self.board.region_tier(side, region)
            in (ScoringTier.DOMINATION, ScoringTier.CONTROL)
        )

    # -- M3: the "war" family (seeded CHANCE roll) --------------------------

    def push_war_target_choice(
        self,
        card_id: str,
        attacker: Side,
        candidates: list[str],
        win_from: int,
        vp: int,
        military_ops: int,
        count_target_control: bool = True,
    ) -> None:
        """A war whose attacker chooses the target (Brush War, Indo-Pakistani
        War, Iran-Iraq War). Resolves to begin_war once the target is picked."""
        options = tuple(
            Action(DecisionKind.WAR_TARGET, {"country": c}) for c in candidates
        )
        if not options:
            return
        self._push(
            attacker, DecisionKind.WAR_TARGET, options,
            {
                "card": card_id, "attacker": attacker.value, "win_from": win_from,
                "vp": vp, "military_ops": military_ops,
                "count_target_control": count_target_control,
            },
        )

    def _handle_war_target(self, decision: Decision, action: Action) -> None:
        ctx = decision.context
        self.begin_war(
            card_id=ctx["card"],
            attacker=Side(ctx["attacker"]),
            target=action.payload["country"],
            win_from=ctx["win_from"],
            vp=ctx["vp"],
            military_ops=ctx["military_ops"],
            count_target_control=ctx["count_target_control"],
        )

    def begin_war(
        self,
        card_id: str,
        attacker: Side,
        target: str,
        win_from: int,
        vp: int,
        military_ops: int,
        count_target_control: bool,
    ) -> None:
        """Start a war event: it always counts toward the attacker's required
        military operations, then a logged CHANCE roll decides the outcome."""
        self.military_ops[attacker.value] += military_ops
        roll = self._roll_d6()
        self._push(
            Side.CHANCE,
            DecisionKind.WAR_ROLL,
            (Action(DecisionKind.WAR_ROLL, {"value": roll}),),
            {
                "card": card_id,
                "attacker": attacker.value,
                "target": target,
                "win_from": win_from,
                "vp": vp,
                "count_target_control": count_target_control,
            },
        )

    def _handle_war_roll(self, decision: Decision, action: Action) -> None:
        ctx = decision.context
        attacker = Side(ctx["attacker"])
        defender = attacker.opponent
        target = ctx["target"]
        roll = action.payload["value"]

        # -1 per defender-controlled country adjacent to the target, plus the
        # target itself when the war counts it (e.g. Arab-Israeli War).
        penalty = sum(
            1 for n in self.board.neighbors(target) if self.board.control(n) is defender
        )
        if ctx["count_target_control"] and self.board.control(target) is defender:
            penalty += 1

        if roll - penalty >= ctx["win_from"]:
            self._award_vp(attacker, ctx["vp"])
            if self.is_terminal:
                return
            # Seize the target: the attacker takes over all defender Influence.
            seized = self.board.influence[target][defender.value]
            self.board.influence[target][defender.value] = 0
            self.board.influence[target][attacker.value] += seized

    # -- M2: scoring & VP ---------------------------------------------------

    def _resolve_scoring_card(self, cid: str) -> None:
        if cid == "Southeast_Asia_Scoring":
            net = self._score_southeast_asia()
        else:
            net = self._score_region_net(SCORING_CARD_REGION[cid])
        self._change_vp_by(net)

    def _scoring_overrides(self, region: Region) -> tuple[frozenset[str], frozenset[str]]:
        """Per-scoring board adjustments set by events, as
        (extra_battlegrounds, ignored) for `region`.

        - Formosan Resolution: while active and the US controls Taiwan, Taiwan
          scores as a Battleground in Asia. (Persistent until the China Card is
          played; not consumed here.)
        - Shuttle Diplomacy: at the *next* scoring of the Middle East or Asia,
          one USSR-controlled Battleground is not counted; the effect is
          consumed (whichever region scores first).
        """
        extra_battlegrounds: set[str] = set()
        ignored: set[str] = set()
        if (
            region is Region.ASIA
            and self.game_effects.get("formosan_resolution")
            and self.board.control("Taiwan") is Side.US
        ):
            extra_battlegrounds.add("Taiwan")
        if region in (Region.MIDDLE_EAST, Region.ASIA) and self.game_effects.get(
            "shuttle_diplomacy"
        ):
            dropped = self._first_ussr_battleground(region)
            if dropped is not None:
                ignored.add(dropped)
            self.game_effects.pop("shuttle_diplomacy", None)  # consumed
        return frozenset(extra_battlegrounds), frozenset(ignored)

    def _first_ussr_battleground(self, region: Region) -> str | None:
        """A USSR-controlled Battleground in `region` (canonical order), or
        None. Shuttle Diplomacy drops exactly one from the USSR tally."""
        for cid, info in self.board.countries.items():
            if (
                info.region is region
                and info.battleground
                and self.board.control(cid) is Side.USSR
            ):
                return cid
        return None

    def _score_region_net(self, region: Region) -> int:
        # Controlling all of Europe when Europe is scored wins outright.
        if region is Region.EUROPE:
            controller = self.board.controls_all_of_europe()
            if controller is not None:
                self._win(controller, "europe_control")
                return 0
        extra_bg, ignored = self._scoring_overrides(region)
        presence, domination, control = RULES["scoring"][region.name]
        tier_value = {
            ScoringTier.NONE: 0,
            ScoringTier.PRESENCE: presence,
            ScoringTier.DOMINATION: domination,
        }

        def value_for(s: Side) -> int:
            tier = self.board.region_tier(s, region, extra_bg, ignored)
            if tier is ScoringTier.CONTROL:
                # Europe leaves its Control value undefined (full control is
                # the win handled above); approximate as Domination. VERIFY.
                base = control if control is not None else domination
            else:
                base = tier_value[tier]
            # 10.1.2: +1 VP per Battleground Controlled in the region, +1 VP
            # per country Controlled there adjacent to the enemy superpower.
            return base + self.board.region_bonus_vp(s, region, extra_bg, ignored)

        return value_for(Side.US) - value_for(Side.USSR)

    def _score_southeast_asia(self) -> int:
        # Physical card text: +2 VP for control of Thailand, +1 VP per other
        # controlled Southeast Asia country, netted US-positive.
        net = 0
        for cid, info in self.board.countries.items():
            if info.subregion is Subregion.SOUTHEAST_ASIA:
                value = 2 if cid == "Thailand" else 1
                controller = self.board.control(cid)
                if controller is Side.US:
                    net += value
                elif controller is Side.USSR:
                    net -= value
        return net

    def _award_vp(self, side: Side, amount: int) -> None:
        self._change_vp_by(amount if side is Side.US else -amount)

    def _change_vp_by(self, net: int) -> None:
        self.vp += net
        if self.vp >= RULES["vp_to_win"]:
            self._win(Side.US, "vp")
        elif self.vp <= -RULES["vp_to_win"]:
            self._win(Side.USSR, "vp")

    def _win(self, side: Side, reason: str) -> None:
        if not self.is_terminal:
            self._winner = side
            self._game_over_reason = reason
            self.phase = "complete"
            # The game is over: no decision is pending (mandate #1 — pending is
            # None iff the game ended). Any half-resolved Ops/event continuation
            # still queued (e.g. an EVENT_RESUME marker below a coup that just
            # hit DEFCON 1) is abandoned.
            self._decision_stack.clear()

    def _file_card(self, side: Side, cid: str, fired: bool) -> None:
        if cid == RULES["china_card_id"]:
            # The China Card is never discarded: it passes to the opponent
            # face-down and becomes available to them next turn. Formosan
            # Resolution's printed text nullifies it specifically "when USA
            # plays The China Card" -- not whenever it merely changes hands.
            if side is Side.US:
                self.game_effects.pop("formosan_resolution", None)
            self.china_card_owner = side.opponent.value
            self.china_card_available = False
            return
        if cid in self.hands[side.value]:
            self.hands[side.value].remove(cid)
        if fired and self.cards[cid].remove_after_event:
            self.removed_cards.append(cid)
        else:
            self.discard_pile.append(cid)

    # -- dispatch -----------------------------------------------------------

    def _dispatch(self, decision: Decision, action: Action) -> None:
        handler = {
            DecisionKind.PLACE_INFLUENCE: self._handle_place_influence,
            DecisionKind.COUP_TARGET: self._handle_coup_target,
            DecisionKind.COUP_ROLL: self._handle_coup_roll,
            DecisionKind.REALIGNMENT_TARGET: self._handle_realignment_target,
            DecisionKind.REALIGNMENT_ACTOR_ROLL: self._handle_realignment_actor_roll,
            DecisionKind.REALIGNMENT_OPPONENT_ROLL: self._handle_realignment_opponent_roll,
            DecisionKind.HEADLINE_PLAY: self._handle_headline_play,
            DecisionKind.ACTION_ROUND_PLAY: self._handle_action_round_play,
            DecisionKind.PLAY_MODE: self._handle_play_mode,
            DecisionKind.OPS_TYPE: self._handle_ops_type,
            DecisionKind.SPACE_RACE_ROLL: self._handle_space_race_roll,
            DecisionKind.EVENT_OPS_ORDER: self._handle_event_ops_order,
            DecisionKind.EVENT_RESUME: self._handle_event_resume,
            DecisionKind.WAR_ROLL: self._handle_war_roll,
            DecisionKind.WAR_TARGET: self._handle_war_target,
            DecisionKind.EVENT_INFLUENCE: self._handle_event_influence,
            DecisionKind.EVENT_CHOICE: self._handle_event_choice,
            DecisionKind.RANDOM_DISCARD: self._handle_random_discard,
            DecisionKind.CONTEST_ROLL: self._handle_contest_roll,
            DecisionKind.QUAGMIRE_DISCARD: self._handle_quagmire_discard,
            DecisionKind.QUAGMIRE_ROLL: self._handle_quagmire_roll,
            DecisionKind.HELD_CARD_DISCARD: self._handle_held_card_discard,
        }[decision.kind]
        handler(decision, action)

    def _new_decision(
        self, actor: Side, kind: DecisionKind, options: tuple[Action, ...], context: dict
    ) -> Decision:
        self._next_decision_id += 1
        return Decision(
            id=self._next_decision_id, actor=actor, kind=kind, options=options, context=context
        )

    def _push(
        self, actor: Side, kind: DecisionKind, options: tuple[Action, ...], context: dict
    ) -> None:
        self._decision_stack.append(self._new_decision(actor, kind, options, context))

    def _roll_d6(self) -> int:
        return self._rng.randint(1, 6)

    # -- influence placement --------------------------------------------------

    def _chernobyl_blocks(self, side: Side, cid: str) -> bool:
        """Chernobyl: the USSR may not add Influence via Operations to the
        designated region for the rest of the turn (events still may)."""
        return (
            side is Side.USSR
            and self.turn_effects.get("chernobyl") == self.board.countries[cid].region.value
        )

    def _place_influence_options(self, side: Side, ops_remaining: int) -> tuple[Action, ...]:
        snapshot = self._ops_round_snapshot
        options = []
        for cid in self.board.countries:
            if not self.board.is_reachable(side, cid, influence=snapshot):
                continue
            if self.board.influence_cost(side, cid) > ops_remaining:
                continue
            if self._chernobyl_blocks(side, cid):
                continue
            options.append(Action(DecisionKind.PLACE_INFLUENCE, {"country": cid}))
        return tuple(options)

    def _maybe_push_place_influence(self, side: Side, ops_remaining: int) -> None:
        if self.is_terminal or ops_remaining <= 0:
            self._ops_round_snapshot = None
            return
        if self._ops_round_snapshot is None:
            # First point of a fresh Operations spend: freeze the board as it
            # stands right now -- this *is* "the start of the Action Round"
            # for every subsequent point in this same spend (6.1.1).
            self._ops_round_snapshot = copy.deepcopy(self.board.influence)
        options = self._place_influence_options(side, ops_remaining)
        if not options:
            self._ops_round_snapshot = None
            return
        self._push(side, DecisionKind.PLACE_INFLUENCE, options, {"ops_remaining": ops_remaining})

    def _bonus_influence_options(
        self, side: Side, base: int, spent: int, non_bonus: int, bonus: str
    ) -> tuple[Action, ...]:
        """Legal placements for a region-bonus influence spend. The +1 bonus
        point is available only while nothing has been (and nothing would be)
        placed outside the bonus region: a placement of cost `c` is legal iff
        `spent + c <= base`, or all Ops (this one included) stay in the region
        and `spent + c <= base + 1`."""
        snapshot = self._ops_round_snapshot
        options = []
        for cid in self.board.countries:
            if not self.board.is_reachable(side, cid, influence=snapshot):
                continue
            if self._chernobyl_blocks(side, cid):
                continue
            cost = self.board.influence_cost(side, cid)
            in_region = self._in_bonus_region(cid, bonus)
            new_spent = spent + cost
            new_non_bonus = non_bonus + (0 if in_region else cost)
            if new_spent <= base or (new_non_bonus == 0 and new_spent <= base + 1):
                options.append(Action(DecisionKind.PLACE_INFLUENCE, {"country": cid}))
        return tuple(options)

    def _maybe_push_bonus_influence(
        self, side: Side, base: int, spent: int, non_bonus: int, bonus: str
    ) -> None:
        if self.is_terminal:
            self._ops_round_snapshot = None
            return
        if self._ops_round_snapshot is None:
            self._ops_round_snapshot = copy.deepcopy(self.board.influence)
        options = self._bonus_influence_options(side, base, spent, non_bonus, bonus)
        if not options:
            self._ops_round_snapshot = None
            return
        self._push(
            side, DecisionKind.PLACE_INFLUENCE, options,
            {"bonus": bonus, "base": base, "spent": spent, "non_bonus": non_bonus},
        )

    def _handle_place_influence(self, decision: Decision, action: Action) -> None:
        if decision.context.get("setup"):
            self._handle_setup_influence(decision, action)
            return
        side = decision.actor
        country = action.payload["country"]
        cost = self.board.influence_cost(side, country)
        self.board.influence[country][side.value] += 1
        bonus = decision.context.get("bonus")
        if bonus:
            in_region = self._in_bonus_region(country, bonus)
            self._maybe_push_bonus_influence(
                side,
                decision.context["base"],
                decision.context["spent"] + cost,
                decision.context["non_bonus"] + (0 if in_region else cost),
                bonus,
            )
            return
        self._maybe_push_place_influence(side, decision.context["ops_remaining"] - cost)

    def _handle_setup_influence(self, decision: Decision, action: Action) -> None:
        side = decision.actor
        # Setup placement always costs one point flat (no opponent doubling).
        self.board.influence[action.payload["country"]][side.value] += 1
        remaining = decision.context["remaining"] - 1
        subregion = Subregion(decision.context["subregion"])
        if remaining > 0:
            self._push_setup_influence_remaining(side, subregion, remaining)
        elif side is Side.USSR:
            # USSR's Eastern Europe done -> US places in Western Europe.
            self._push_setup_influence(Side.US, Subregion.WESTERN_EUROPE)
        else:
            self.phase = "headline"  # setup complete; _advance pushes headline

    # -- coup --------------------------------------------------------------

    def _coup_target_options(self, side: Side) -> tuple[Action, ...]:
        return tuple(
            Action(DecisionKind.COUP_TARGET, {"country": cid})
            for cid in self.board.countries
            if self._usable_coup_realign_target(side, cid)
        )

    def _handle_coup_target(self, decision: Decision, action: Action) -> None:
        side = decision.actor
        country = action.payload["country"]
        ops = decision.context["ops"]
        # Region-bonus play: +1 Op (and +1 military Op) when the coup target is
        # in the bonus region (China Card -> Asia, Vietnam Revolts -> SE Asia).
        bonus = decision.context.get("bonus")
        if bonus and self._in_bonus_region(country, bonus):
            ops += 1
            self.military_ops[side.value] += 1
        roll = self._roll_d6()
        self._push(
            Side.CHANCE,
            DecisionKind.COUP_ROLL,
            (Action(DecisionKind.COUP_ROLL, {"value": roll}),),
            {"side": side.value, "country": country, "ops": ops},
        )

    def _handle_coup_roll(self, decision: Decision, action: Action) -> None:
        side = Side(decision.context["side"])
        country = decision.context["country"]
        ops = decision.context["ops"]
        roll = action.payload["value"]
        info = self.board.countries[country]

        # Cuban Missile Crisis: any coup attempt by the flagged side this turn
        # is Global Thermonuclear War — that side loses immediately.
        if self.turn_effects.get("cuban_missile_crisis") == side.value:
            self._win(side.opponent, "cuban_missile_crisis")
            return

        opp_removed = 0
        margin = roll + ops - 2 * info.stability + self._coup_roll_modifier(side, info)
        if margin > 0:
            opponent = side.opponent
            opp_removed = min(margin, self.board.influence[country][opponent.value])
            self.board.influence[country][opponent.value] -= opp_removed
            leftover = margin - opp_removed
            self.board.influence[country][side.value] += leftover

        # Every coup attempt degrades DEFCON by 1 — except a US coup in a
        # Battleground while Nuclear Subs is in effect this turn.
        nuclear_subs = (
            side is Side.US
            and info.battleground
            and self.turn_effects.get("nuclear_subs")
        )
        if not nuclear_subs:
            self._change_defcon(-1, caused_by=side)

        # Yuri and Samantha: the USSR scores 1 VP for every US coup attempt,
        # for the rest of the game.
        if (
            side is Side.US
            and self.game_effects.get("yuri_samantha")
            and not self.is_terminal
        ):
            self._award_vp(Side.USSR, 1)

        # Che: a coup that removed opponent Influence grants ONE second free coup
        # against a *different* country in the same regions (len(used) < 2 caps
        # the chain at two attempts total).
        che = decision.context.get("che")
        if (
            che is not None
            and opp_removed > 0
            and not self.is_terminal
            and len(che["used"]) < 2
        ):
            self.push_che_coup(side, che["ops"], che["candidates"], used=che["used"])

    def _coup_roll_modifier(self, side: Side, info) -> int:
        """Per-turn additive modifiers to a coup roll: Latin American Death
        Squads (+1 for its player, -1 for the opponent, in the Americas) and
        SALT Negotiations (-1 for both sides, everywhere)."""
        mod = 0
        lads = self.turn_effects.get("la_death_squads")
        if lads and info.region in (Region.CENTRAL_AMERICA, Region.SOUTH_AMERICA):
            mod += 1 if side.value == lads else -1
        if self.turn_effects.get("salt"):
            mod -= 1
        return mod

    # -- M3: a free operation confined to a set of countries ----------------

    def push_free_coup_or_realign(
        self, side: Side, event: str, ops: int, countries: list[str],
        allow_realign: bool = True,
    ) -> None:
        """Offer `side` a free Coup or Realignment (or neither), restricted to
        `countries` (Junta's "free Coup/Realignment in the Americas"; Ortega
        Elected in Nicaragua's Coup-only variant sets `allow_realign=False`).
        Coup is offered only where DEFCON and the persistent locks allow it."""
        choices = ["none"]
        coupable = [
            cid for cid in countries
            if self._usable_coup_realign_target(side, cid)
        ]
        realignable = [
            cid for cid in countries
            if allow_realign and self._usable_coup_realign_target(side, cid, for_coup=False)
        ]
        if coupable:
            choices.append("coup")
        if realignable:
            choices.append("realign")
        if len(choices) == 1:  # only "none"
            return
        self._push(
            side, DecisionKind.EVENT_CHOICE,
            tuple(Action(DecisionKind.EVENT_CHOICE, {"choice": c}) for c in choices),
            {"event": event, "choose_side": side.value,
             "ops": ops, "countries": list(countries)},
        )

    def resolve_free_op_choice(
        self, side: Side, choice: str, ops: int, countries: list[str]
    ) -> None:
        """Continue a push_free_coup_or_realign branch once the player picks.

        A free Coup roll granted by an event does not count towards required
        Military Operations (8.2.5), so it never touches military_ops."""
        if choice == "coup":
            options = tuple(
                Action(DecisionKind.COUP_TARGET, {"country": cid})
                for cid in countries
                if self._usable_coup_realign_target(side, cid)
            )
            if options:
                self._push(side, DecisionKind.COUP_TARGET, options, {"ops": ops, "bonus": None})
        elif choice == "realign":
            options = tuple(
                Action(DecisionKind.REALIGNMENT_TARGET, {"country": cid})
                for cid in countries
                if self._usable_coup_realign_target(side, cid, for_coup=False)
            )
            if options:
                self._push(
                    side, DecisionKind.REALIGNMENT_TARGET, options,
                    {"card_ops": ops, "spent": 0, "bonus": None, "non_bonus": 0},
                )

    # -- M3: reclaim a card from the (public) discard pile ------------------

    def push_take_from_discard(self, side: Side, event: str) -> None:
        """Offer `side` a non-scoring card from the discard pile to take back to
        hand (SALT Negotiations, ...). A "none" option keeps it optional; if the
        discard has no non-scoring card, no decision is pushed."""
        choices = tuple(
            cid for cid in self.discard_pile if not self.cards[cid].scoring
        )
        if not choices:
            return
        self.push_event_choice(event, side, choices + ("none",))

    def play_card_from_discard(self, side: Side, cid: str) -> None:
        """Play `cid` — already pulled out of the discard pile — immediately for
        its event, on `side`'s behalf (Star Wars). Files it afterwards (removed
        if it is a remove-after-event card), exactly like a normal event play; an
        unimplemented event is a no-op discard."""
        card = self.cards[cid]
        if card.scoring:
            self._resolve_scoring_card(cid)
            if not self.is_terminal:
                self._file_card(side, cid, fired=True)
            return
        if self._has_event(cid) and EVENTS[cid].eligible(self, side):
            self._file_card(side, cid, fired=True)
            self._fire_event(side, cid)
        else:
            self._file_card(side, cid, fired=False)

    # -- M3: Che — a free coup with a conditional repeat --------------------

    def push_che_coup(
        self, side: Side, ops: int, candidates: list[str], used: tuple[str, ...] = ()
    ) -> None:
        """Offer `side` a free Coup (or a decline) against one of `candidates`,
        excluding any already couped this play (`used`). Reused for both the
        first Che attempt and the conditional second one."""
        used = list(used)
        targets = [
            cid
            for cid in candidates
            if cid not in used
            and self._usable_coup_realign_target(side, cid)
        ]
        if not targets:
            return
        self.push_event_choice(
            "Che", side, tuple(targets) + ("none",),
            extra={"che_ops": ops, "che_candidates": list(candidates), "che_used": used},
        )

    def begin_che_coup(
        self, side: Side, country: str, ops: int, candidates: list[str], used: list[str]
    ) -> None:
        """Resolve a chosen Che coup: a free Coup roll (8.2.5), so it does not
        move the Military Ops track. A logged CHANCE roll decides it; the
        COUP_ROLL context carries the `che` state so _handle_coup_roll can
        offer the second attempt if this one removes US Influence."""
        used = list(used) + [country]
        roll = self._roll_d6()
        self._push(
            Side.CHANCE,
            DecisionKind.COUP_ROLL,
            (Action(DecisionKind.COUP_ROLL, {"value": roll}),),
            {
                "side": side.value, "country": country, "ops": ops,
                "che": {"ops": ops, "candidates": list(candidates), "used": used},
            },
        )

    # -- M3: Missile Envy — take the opponent's top-Ops card and use it -----

    def missile_envy_take(self, taker: Side, cid: str) -> None:
        """`taker` takes `cid` from the giver and either uses it (a neutral card
        or one of the taker's own events → Ops-or-Event choice) or is forced to
        Ops only (a scoring card or the giver's own event).

        The card is left in the giver's hand until `missile_envy_use` actually
        resolves it, so it is never in limbo while the Ops-or-Event choice is
        pending (the same convention Grain Sales uses for its revealed card)."""
        card = self.cards[cid]
        ops_only = card.scoring or card.side.value == taker.opponent.value
        if ops_only:
            self.missile_envy_use(taker, cid, "ops")
        else:
            self.push_event_choice(
                "Missile_Envy_use", taker, ("ops", "event"), extra={"card": cid}
            )

    def missile_envy_use(self, taker: Side, cid: str, mode: str) -> None:
        """Resolve the taken card as `mode` for `taker`: fire its event, or
        conduct its Ops. The card leaves the giver's hand here and is filed
        (removed if it is a remove-after-event card whose event fired)."""
        giver = taker.opponent
        if cid in self.hands[giver.value]:
            self.hands[giver.value].remove(cid)
        card = self.cards[cid]
        if mode == "event":
            implemented = self._has_event(cid) and EVENTS[cid].eligible(self, taker)
            self._file_card(taker, cid, fired=implemented)
            if implemented:
                self._fire_event(taker, cid)
        else:  # ops
            self.discard_pile.append(cid)
            self.push_event_operations(taker, card.ops)

    # -- M3: Bear Trap / Quagmire — a persistent per-player operating lock --
    #
    # While trapped, `side` may not play cards normally. On each forthcoming
    # round: if it holds a card with Ops 2+, it must discard one, then rolls
    # to break free (5-6). Once it holds no such card, it instead plays every
    # scoring card still in hand (forced) and its remaining rounds *this
    # turn* are skipped outright -- but it still gets that turn's one escape
    # roll right away, not a silent skip. Bear Trap traps the USSR; Quagmire
    # traps the US -- independent of who actually plays the card, exactly
    # like Duck and Cover always favors the US regardless of who plays it.

    _TRAP_KEYS: dict[str, Side] = {"bear_trap": Side.USSR, "quagmire": Side.US}

    def _trap_key_for(self, side: Side) -> str | None:
        for key, trapped_side in self._TRAP_KEYS.items():
            if trapped_side is side and self.game_effects.get(key):
                return key
        return None

    def _push_trap_step(self, side: Side, key: str) -> None:
        if side.value in self.turn_effects.get("trap_skip", []):
            return  # already out of cards this turn: remaining rounds skip
        payable = [
            cid for cid in self.hands[side.value]
            if not self.cards[cid].scoring and self.cards[cid].ops >= 2
        ]
        if payable:
            options = tuple(
                Action(DecisionKind.QUAGMIRE_DISCARD, {"card": cid}) for cid in payable
            )
            self._push(side, DecisionKind.QUAGMIRE_DISCARD, options, {"key": key})
            return
        # No Ops-2+ card: forced to play every scoring card still held, then
        # every remaining round this turn is skipped -- but the escape roll
        # still happens now, this turn, not a silent forfeit.
        skip = self.turn_effects.setdefault("trap_skip", [])
        skip.append(side.value)
        for cid in list(self.hands[side.value]):
            if self.cards[cid].scoring:
                self._resolve_scoring_card(cid)
                self._file_card(side, cid, fired=True)
                if self.is_terminal:
                    return
        self._push_quagmire_roll(key)

    def _handle_quagmire_discard(self, decision: Decision, action: Action) -> None:
        side = decision.actor
        key = decision.context["key"]
        cid = action.payload["card"]
        self._file_card(side, cid, fired=False)
        self._push_quagmire_roll(key)

    def _push_quagmire_roll(self, key: str) -> None:
        roll = self._roll_d6()
        self._push(
            Side.CHANCE, DecisionKind.QUAGMIRE_ROLL,
            (Action(DecisionKind.QUAGMIRE_ROLL, {"value": roll}),), {"key": key},
        )

    def _handle_quagmire_roll(self, decision: Decision, action: Action) -> None:
        if action.payload["value"] >= 5:  # break free
            self.game_effects.pop(decision.context["key"], None)

    # -- realignment ---------------------------------------------------------

    def _realignment_target_options(self, side: Side) -> tuple[Action, ...]:
        return tuple(
            Action(DecisionKind.REALIGNMENT_TARGET, {"country": cid})
            for cid in self.board.countries
            if self._usable_coup_realign_target(side, cid, for_coup=False)
        )

    def _maybe_push_realignment_target(
        self,
        side: Side,
        card_ops: int,
        spent: int,
        bonus: str | None = None,
        non_bonus: int = 0,
    ) -> None:
        """Push the next Realignment-attempt decision, if any attempts are
        still available. `spent` attempts are always allowed up to
        `card_ops`; one extra attempt (`card_ops + 1`) is allowed on top of
        that only while `bonus` is active and every attempt so far has
        stayed inside the bonus region (`non_bonus == 0`) -- the same
        all-or-nothing rule `_maybe_push_bonus_influence` uses."""
        if self.is_terminal:
            return
        if not (spent < card_ops or (bonus and non_bonus == 0 and spent < card_ops + 1)):
            return
        options = self._realignment_target_options(side)
        if not options:
            return
        self._push(
            side,
            DecisionKind.REALIGNMENT_TARGET,
            options,
            {"card_ops": card_ops, "spent": spent, "bonus": bonus, "non_bonus": non_bonus},
        )

    def _handle_realignment_target(self, decision: Decision, action: Action) -> None:
        side = decision.actor
        country = action.payload["country"]
        roll = self._roll_d6()
        self._push(
            Side.CHANCE,
            DecisionKind.REALIGNMENT_ACTOR_ROLL,
            (Action(DecisionKind.REALIGNMENT_ACTOR_ROLL, {"value": roll}),),
            {
                "side": side.value,
                "country": country,
                "card_ops": decision.context["card_ops"],
                "spent": decision.context["spent"],
                "bonus": decision.context["bonus"],
                "non_bonus": decision.context["non_bonus"],
            },
        )

    def _handle_realignment_actor_roll(self, decision: Decision, action: Action) -> None:
        opp_roll = self._roll_d6()
        self._push(
            Side.CHANCE,
            DecisionKind.REALIGNMENT_OPPONENT_ROLL,
            (Action(DecisionKind.REALIGNMENT_OPPONENT_ROLL, {"value": opp_roll}),),
            {**decision.context, "actor_roll": action.payload["value"]},
        )

    def _handle_realignment_opponent_roll(self, decision: Decision, action: Action) -> None:
        side = Side(decision.context["side"])
        opponent = side.opponent
        country = decision.context["country"]
        card_ops = decision.context["card_ops"]
        spent = decision.context["spent"]
        bonus = decision.context["bonus"]
        non_bonus = decision.context["non_bonus"]
        actor_roll = decision.context["actor_roll"]
        opp_roll = action.payload["value"]

        # 6.2.2: the die roll is not modified by the Operations value of the
        # card spent — Ops only buys attempts (1 per point), unlike a Coup.
        actor_total = (
            actor_roll + self._realignment_bonus(side, country)
            + self._realignment_modifier(side)
        )
        opp_total = opp_roll + self._realignment_bonus(opponent, country)
        margin = actor_total - opp_total
        if margin > 0:
            removed = min(margin, self.board.influence[country][opponent.value])
            self.board.influence[country][opponent.value] -= removed
        elif margin < 0:
            # A losing realignment roll costs the acting side their own
            # influence in the target country, not just a wasted attempt.
            removed = min(-margin, self.board.influence[country][side.value])
            self.board.influence[country][side.value] -= removed

        if bonus and not self._in_bonus_region(country, bonus):
            non_bonus += 1
        self._maybe_push_realignment_target(side, card_ops, spent + 1, bonus, non_bonus)

    def _realignment_bonus(self, side: Side, country: str) -> int:
        bonus = 1 if self.board.is_adjacent(side.value, country) else 0
        bonus += sum(1 for n in self.board.neighbors(country) if self.board.control(n) is side)
        # 6.2.2's third modifier: +1 if this side already holds more
        # Influence in the target than their opponent does.
        if self.board.influence[country][side.value] > self.board.influence[country][side.opponent.value]:
            bonus += 1
        return bonus

    def _realignment_modifier(self, side: Side) -> int:
        """Per-turn additive modifier to the acting side's realignment roll
        (Iran-Contra Scandal: -1 to US realignment rolls this turn)."""
        if side is Side.US and self.turn_effects.get("iran_contra"):
            return -1
        return 0

    # -- shared -------------------------------------------------------------

    def _change_defcon(self, delta: int, caused_by: Side) -> None:
        before = self.defcon
        self.defcon = max(1, min(5, self.defcon + delta))
        if self.defcon == 1:
            self._win(caused_by.opponent, "defcon_1")
            return
        # NORAD: "If Canada is US-controlled", each time DEFCON MOVES to level
        # 2 the US adds 1 Influence to a country where it already has some.
        if (
            self.defcon == 2
            and before != 2
            and self.game_effects.get("norad")
            and self.board.control("Canada") is Side.US
        ):
            self._push_norad_influence()

    def _push_norad_influence(self) -> None:
        candidates = [
            cid for cid in self.board.countries if self.board.influence[cid]["US"] > 0
        ]
        if candidates:
            self.push_event_influence(
                event="NORAD", op="place", choose_side=Side.US, inf_side=Side.US,
                remaining=1, candidates=candidates,
            )


# -- serialization helpers ---------------------------------------------------


def _encode_action(a: Action) -> dict:
    return {"kind": a.kind.value, "payload": dict(a.payload)}


def _decode_action(d: dict) -> Action:
    return Action(kind=DecisionKind(d["kind"]), payload=d["payload"])


def _encode_decision(d: Decision) -> dict:
    return {
        "id": d.id,
        "actor": d.actor.value,
        "kind": d.kind.value,
        "options": [_encode_action(a) for a in d.options],
        "context": dict(d.context),
    }


def _decode_decision(d: dict) -> Decision:
    return Decision(
        id=d["id"],
        actor=Side(d["actor"]),
        kind=DecisionKind(d["kind"]),
        options=tuple(_decode_action(a) for a in d["options"]),
        context=d["context"],
    )


def _encode_rng_state(state: tuple) -> list:
    version, internal, gauss_next = state
    return [version, list(internal), gauss_next]


def _decode_rng_state(data: list) -> tuple:
    version, internal, gauss_next = data
    return (version, tuple(internal), gauss_next)
