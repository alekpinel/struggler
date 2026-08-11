"""The M1 engine: board mechanics only, no cards.

Implements the pending-decision stack (mandate #1), atomic Ops actions
(mandate #2), seeded RNG exposed as CHANCE decisions (mandate #3),
per-player observation (mandate #4), and flat serialization (mandate #5)
for: influence placement, control, region scoring, DEFCON, coups, and
realignment.

Since no cards exist yet, Ops points are granted directly via
`begin_influence_operations` / `begin_coup` / `begin_realignment_operations`
(per CLAUDE.md's M1 scope: "Ops-only actions are driven directly for
testing"). M2 will replace direct calls to these with a legitimate
PLAY_CARD decision that grants Ops through the card mechanism; the
decision-stack handlers below don't change.

Numeric constants below are confirmed against the physical game unless
marked UNCONFIRMED.
"""

from __future__ import annotations

import copy
import random

from struggler.board import SCORING, Board
from struggler.cards import action_rounds, cards_entering, hand_limit, load_cards
from struggler.types import (
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

# Minimum DEFCON level required to attempt a coup in a region; regions not
# listed have no restriction. Confirmed against the physical game.
COUP_MIN_DEFCON: dict[Region, int] = {
    Region.EUROPE: 5,
    Region.ASIA: 4,
    Region.MIDDLE_EAST: 3,
}
_DEFAULT_MIN_DEFCON = 1

# Every coup attempt, in any region, degrades DEFCON by 1, regardless of
# success. Confirmed against the physical game. Realignment is NOT subject
# to the COUP_MIN_DEFCON restriction above (only coups are) — this remains
# an unconfirmed assumption.

# VP required to win outright; the track runs to 20 in either direction.
VP_TO_WIN = 20

# Each regional scoring card maps to the region score_region() already
# computes (mandate: scoring is a board mechanic reused from M1, not a card
# "event"). Southeast Asia Scoring is a subregion-scoring card with different
# rules and is handled separately.
SCORING_CARD_REGION: dict[str, Region] = {
    "Asia_Scoring": Region.ASIA,
    "Europe_Scoring": Region.EUROPE,
    "Middle_East_Scoring": Region.MIDDLE_EAST,
    "Central_America_Scoring": Region.CENTRAL_AMERICA,
    "Africa_Scoring": Region.AFRICA,
    "South_America_Scoring": Region.SOUTH_AMERICA,
}

# The China Card starts face-up with the USSR.
CHINA_CARD_ID = "The_China_Card"

# Additional influence each side places by choice during setup, after the
# printed at-start influence: the USSR into Eastern Europe, the US into
# Western Europe. VERIFY the exact counts against the rulebook.
SETUP_ADDITIONAL = {
    Subregion.EASTERN_EUROPE: (Side.USSR, 6),
    Subregion.WESTERN_EUROPE: (Side.US, 7),
}

# Space Race track, boxes 1..8. Per box: minimum Ops the played card must be
# worth to attempt entry, the die roll needed (success iff d6 <= roll_max),
# and the VP awarded to the first / second superpower to reach the box.
#
# VERIFY: these numeric constants are best-effort from knowledge of the
# physical Space Race track and have NOT been reconfirmed line-by-line. The
# *mechanism* around them (attempt -> seeded CHANCE roll -> advance -> award)
# is the part M2 proves; only the numbers here are provisional. The
# functional perks some boxes grant (extra action round, headline-reveal
# advantage, opponent-must-discard) are deferred to a later increment.
SPACE_RACE_BOXES: dict[int, dict[str, int]] = {
    1: {"ops": 2, "roll_max": 3, "vp_first": 2, "vp_second": 1},
    2: {"ops": 2, "roll_max": 4, "vp_first": 0, "vp_second": 0},
    3: {"ops": 2, "roll_max": 3, "vp_first": 2, "vp_second": 0},
    4: {"ops": 2, "roll_max": 4, "vp_first": 0, "vp_second": 0},
    5: {"ops": 3, "roll_max": 3, "vp_first": 3, "vp_second": 1},
    6: {"ops": 3, "roll_max": 4, "vp_first": 0, "vp_second": 0},
    7: {"ops": 3, "roll_max": 3, "vp_first": 4, "vp_second": 2},
    8: {"ops": 4, "roll_max": 2, "vp_first": 2, "vp_second": 0},
}
SPACE_RACE_MAX_BOX = 8
# A side that has reached this box may make two Space Race attempts per turn
# instead of one. VERIFY exact box.
SPACE_RACE_TWO_ATTEMPTS_FROM_BOX = 2


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

        # -- M2 card / full-game state --------------------------------------
        # Defaults leave the engine in the M1 "sandbox" (phase="idle"): no
        # deck, no turn loop, begin_* entry points drive decisions directly.
        # A full game is started via Engine.new_game(), which sets phase and
        # populates the deck/hands below.
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
        return engine

    # -- M1 test-harness entry points (no cards yet) -----------------------

    def begin_influence_operations(self, side: Side, ops: int) -> None:
        if ops <= 0:
            raise ValueError("ops must be positive")
        self._maybe_push_place_influence(side, ops)

    def begin_coup(self, side: Side, ops: int) -> None:
        if ops <= 0:
            raise ValueError("ops must be positive")
        options = self._coup_target_options()
        if not options:
            return
        self._push(side, DecisionKind.COUP_TARGET, options, {"ops": ops})

    def begin_realignment_operations(self, side: Side, ops: int) -> None:
        if ops <= 0:
            raise ValueError("ops must be positive")
        self._maybe_push_realignment_target(side, card_ops=ops, attempts_remaining=ops)

    # -- M2: full-game entry point -----------------------------------------

    @classmethod
    def new_game(
        cls, seed: int, include_optional: bool = False, board: Board | None = None
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
            if self._headline["USSR"] is None:
                self._push_headline(Side.USSR)
                return
            if self._headline["US"] is None:
                self._push_headline(Side.US)
                return
            self._resolve_headline()
            if self.is_terminal:
                return
            self._begin_action_rounds()
            return

        if self.phase == "action_rounds":
            total = 2 * action_rounds(self.turn)
            if self._ars_played >= total:
                self._end_of_turn()
                return
            idx = self._ars_played  # 0-based play index within the turn
            side = Side.USSR if idx % 2 == 0 else Side.US
            self.action_round = idx // 2 + 1
            self._ars_played += 1
            self._push_action_round_play(side)
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
        # DEFCON recovers by one at the end of every turn.
        self._change_defcon(+1, caused_by=Side.US)
        # A China Card passed this turn becomes available to its new owner.
        self.china_card_available = True
        # Reset per-turn accounting.
        self.military_ops = {"US": 0, "USSR": 0}
        self.space_race_attempts = {"US": 0, "USSR": 0}
        self._headline = {"US": None, "USSR": None}

        if self.turn >= 10:
            self._finish_game()
            return
        self.turn += 1
        self._start_turn()

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

    def _begin_action_rounds(self) -> None:
        self.phase = "action_rounds"
        self._ars_played = 0
        self.action_round = 1

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
        remaining = SETUP_ADDITIONAL[subregion][1]
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

    def _resolve_headline(self) -> None:
        # Capture and clear the selections first so a headlined card lives in
        # exactly one place (its destination pile) once resolved, never also
        # lingering as a stale marker. Higher Ops resolves first; ties resolve
        # US-first (VERIFY). Order is cosmetic in M2 (only scoring events act).
        picks = {s: self._headline[s.value] for s in (Side.US, Side.USSR)}
        self._headline = {"US": None, "USSR": None}
        order = sorted(
            (Side.US, Side.USSR),
            key=lambda s: (-self.cards[picks[s]].ops, s is not Side.US),
        )
        for s in order:
            self._apply_card_event(s, picks[s])
            if self.is_terminal:
                return

    # -- M2: action round: pick a card, then how to use it ------------------

    def _push_action_round_play(self, side: Side) -> None:
        hand = self.hands[side.value]
        scoring_in_hand = [cid for cid in hand if self.cards[cid].scoring]
        # A scoring card may not be held past the end of the turn. Once a side
        # has as many scoring cards as it has action rounds left, every
        # remaining round must spend one (the China Card is not offered then).
        must_play_scoring = bool(scoring_in_hand) and len(
            scoring_in_hand
        ) >= self._remaining_action_rounds(side)

        playable = scoring_in_hand if must_play_scoring else list(hand)
        options = [
            Action(DecisionKind.ACTION_ROUND_PLAY, {"card": cid}) for cid in playable
        ]
        if (
            not must_play_scoring
            and side.value == self.china_card_owner
            and self.china_card_available
        ):
            options.append(Action(DecisionKind.ACTION_ROUND_PLAY, {"card": CHINA_CARD_ID}))
        if options:
            self._push(side, DecisionKind.ACTION_ROUND_PLAY, tuple(options), {})

    def _remaining_action_rounds(self, side: Side) -> int:
        """Action rounds `side` still has this turn, including the one now
        being set up. (`side` is always the side whose play is current.)"""
        total = 2 * action_rounds(self.turn)
        idx = self._ars_played - 1  # current 0-based play index within the turn
        if idx < 0 or idx >= total:
            return 0
        return (total - 1 - idx) // 2 + 1

    def _handle_action_round_play(self, decision: Decision, action: Action) -> None:
        side = decision.actor
        cid = action.payload["card"]
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
        if cid != CHINA_CARD_ID:
            modes.append("event")
        if self._can_space_race(side, card):
            modes.append("space_race")
        return tuple(modes)

    def _handle_play_mode(self, decision: Decision, action: Action) -> None:
        side = decision.actor
        cid = decision.context["card"]
        card = self.cards[cid]
        mode = action.payload["mode"]

        if mode == "event":
            if card.scoring:
                self._resolve_scoring_card(cid)
            self._file_card(side, cid, fired=card.scoring)
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
        ops = card.ops
        self._file_card(side, cid, fired=False)  # China Card passes here
        self._push(
            side, DecisionKind.OPS_TYPE, self._ops_type_options(side, ops),
            {"side": side.value, "ops": ops},
        )

    def _ops_type_options(self, side: Side, ops: int) -> tuple[Action, ...]:
        types = []
        if self._place_influence_options(side, ops):
            types.append("influence")
        if self._coup_target_options():
            types.append("coup")
        types.append("realignment")  # always has at least one legal target
        return tuple(Action(DecisionKind.OPS_TYPE, {"type": t}) for t in types)

    def _handle_ops_type(self, decision: Decision, action: Action) -> None:
        side = Side(decision.context["side"])
        ops = decision.context["ops"]
        ops_type = action.payload["type"]
        if ops_type == "influence":
            self._maybe_push_place_influence(side, ops)
        elif ops_type == "coup":
            # Coups count toward the turn's required military operations.
            self.military_ops[side.value] += ops
            self.begin_coup(side, ops)
        else:  # realignment
            self._maybe_push_realignment_target(side, card_ops=ops, attempts_remaining=ops)

    # -- M2: space race -----------------------------------------------------

    def _space_attempts_allowed(self, side: Side) -> int:
        if self.space_race[side.value] >= SPACE_RACE_TWO_ATTEMPTS_FROM_BOX:
            return 2
        return 1

    def _can_space_race(self, side: Side, card: Card) -> bool:
        pos = self.space_race[side.value]
        if pos >= SPACE_RACE_MAX_BOX:
            return False
        if card.ops < SPACE_RACE_BOXES[pos + 1]["ops"]:
            return False
        return self.space_race_attempts[side.value] < self._space_attempts_allowed(side)

    def _handle_space_race_roll(self, decision: Decision, action: Action) -> None:
        side = Side(decision.context["side"])
        roll = action.payload["value"]
        next_box = self.space_race[side.value] + 1
        box = SPACE_RACE_BOXES[next_box]
        if roll <= box["roll_max"]:
            self.space_race[side.value] = next_box
            first = self.space_race[side.opponent.value] < next_box
            vp = box["vp_first"] if first else box["vp_second"]
            if vp:
                self._award_vp(side, vp)

    # -- M2: scoring & VP ---------------------------------------------------

    def _apply_card_event(self, side: Side, cid: str) -> None:
        card = self.cards[cid]
        if card.scoring:
            self._resolve_scoring_card(cid)
        self._file_card(side, cid, fired=card.scoring)

    def _resolve_scoring_card(self, cid: str) -> None:
        if cid == "Southeast_Asia_Scoring":
            net = self._score_southeast_asia()
        else:
            net = self._score_region_net(SCORING_CARD_REGION[cid])
        self._change_vp_by(net)

    def _score_region_net(self, region: Region) -> int:
        # Controlling all of Europe when Europe is scored wins outright.
        if region is Region.EUROPE:
            controller = self.board.controls_all_of_europe()
            if controller is not None:
                self._win(controller, "europe_control")
                return 0
        presence, domination, control = SCORING[region]
        tier_value = {
            ScoringTier.NONE: 0,
            ScoringTier.PRESENCE: presence,
            ScoringTier.DOMINATION: domination,
        }

        def value_for(s: Side) -> int:
            tier = self.board.region_tier(s, region)
            if tier is ScoringTier.CONTROL:
                # Europe leaves its Control value undefined (full control is
                # the win handled above); approximate as Domination. VERIFY.
                return control if control is not None else domination
            return tier_value[tier]

        return value_for(Side.US) - value_for(Side.USSR)

    def _score_southeast_asia(self) -> int:
        # 1 VP per Southeast Asia country controlled, netted US-positive.
        # VERIFY: the printed card gives Thailand extra weight (not modeled).
        net = 0
        for cid, info in self.board.countries.items():
            if info.subregion is Subregion.SOUTHEAST_ASIA:
                controller = self.board.control(cid)
                if controller is Side.US:
                    net += 1
                elif controller is Side.USSR:
                    net -= 1
        return net

    def _award_vp(self, side: Side, amount: int) -> None:
        self._change_vp_by(amount if side is Side.US else -amount)

    def _change_vp_by(self, net: int) -> None:
        self.vp += net
        if self.vp >= VP_TO_WIN:
            self._win(Side.US, "vp")
        elif self.vp <= -VP_TO_WIN:
            self._win(Side.USSR, "vp")

    def _win(self, side: Side, reason: str) -> None:
        if not self.is_terminal:
            self._winner = side
            self._game_over_reason = reason
            self.phase = "complete"

    def _file_card(self, side: Side, cid: str, fired: bool) -> None:
        if cid == CHINA_CARD_ID:
            # The China Card is never discarded: it passes to the opponent
            # face-down and becomes available to them next turn.
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
        }[decision.kind]
        handler(decision, action)

    def _push(
        self, actor: Side, kind: DecisionKind, options: tuple[Action, ...], context: dict
    ) -> None:
        self._next_decision_id += 1
        self._decision_stack.append(
            Decision(id=self._next_decision_id, actor=actor, kind=kind, options=options, context=context)
        )

    def _roll_d6(self) -> int:
        return self._rng.randint(1, 6)

    # -- influence placement --------------------------------------------------

    def _place_influence_options(self, side: Side, ops_remaining: int) -> tuple[Action, ...]:
        options = []
        for cid in self.board.countries:
            if not self.board.is_reachable(side, cid):
                continue
            if self.board.influence_cost(side, cid) > ops_remaining:
                continue
            options.append(Action(DecisionKind.PLACE_INFLUENCE, {"country": cid}))
        return tuple(options)

    def _maybe_push_place_influence(self, side: Side, ops_remaining: int) -> None:
        if self.is_terminal or ops_remaining <= 0:
            return
        options = self._place_influence_options(side, ops_remaining)
        if not options:
            return
        self._push(side, DecisionKind.PLACE_INFLUENCE, options, {"ops_remaining": ops_remaining})

    def _handle_place_influence(self, decision: Decision, action: Action) -> None:
        if decision.context.get("setup"):
            self._handle_setup_influence(decision, action)
            return
        side = decision.actor
        country = action.payload["country"]
        cost = self.board.influence_cost(side, country)
        self.board.influence[country][side.value] += 1
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

    def _coup_target_options(self) -> tuple[Action, ...]:
        options = []
        for cid, info in self.board.countries.items():
            min_defcon = COUP_MIN_DEFCON.get(info.region, _DEFAULT_MIN_DEFCON)
            if self.defcon < min_defcon:
                continue
            options.append(Action(DecisionKind.COUP_TARGET, {"country": cid}))
        return tuple(options)

    def _handle_coup_target(self, decision: Decision, action: Action) -> None:
        side = decision.actor
        country = action.payload["country"]
        ops = decision.context["ops"]
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

        margin = roll + ops - 2 * info.stability
        if margin > 0:
            opponent = side.opponent
            removed = min(margin, self.board.influence[country][opponent.value])
            self.board.influence[country][opponent.value] -= removed
            leftover = margin - removed
            self.board.influence[country][side.value] += leftover

        # Every coup attempt, anywhere, degrades DEFCON by 1 regardless of
        # region or success.
        self._change_defcon(-1, caused_by=side)

    # -- realignment ---------------------------------------------------------

    def _maybe_push_realignment_target(
        self, side: Side, card_ops: int, attempts_remaining: int
    ) -> None:
        if self.is_terminal or attempts_remaining <= 0:
            return
        options = tuple(
            Action(DecisionKind.REALIGNMENT_TARGET, {"country": cid}) for cid in self.board.countries
        )
        self._push(
            side,
            DecisionKind.REALIGNMENT_TARGET,
            options,
            {"card_ops": card_ops, "attempts_remaining": attempts_remaining},
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
                "attempts_remaining": decision.context["attempts_remaining"],
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
        attempts_remaining = decision.context["attempts_remaining"]
        actor_roll = decision.context["actor_roll"]
        opp_roll = action.payload["value"]

        actor_total = actor_roll + card_ops + self._realignment_bonus(side, country)
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

        self._maybe_push_realignment_target(side, card_ops, attempts_remaining - 1)

    def _realignment_bonus(self, side: Side, country: str) -> int:
        bonus = 1 if self.board.is_adjacent(side.value, country) else 0
        bonus += sum(1 for n in self.board.neighbors(country) if self.board.control(n) is side)
        return bonus

    # -- shared -------------------------------------------------------------

    def _change_defcon(self, delta: int, caused_by: Side) -> None:
        self.defcon = max(1, min(5, self.defcon + delta))
        if self.defcon == 1 and not self.is_terminal:
            self._winner = caused_by.opponent
            self._game_over_reason = "defcon_1"
            self.phase = "complete"


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
