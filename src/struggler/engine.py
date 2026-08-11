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

Several numeric constants are flagged PROVISIONAL where cross-checking
against public sources gave inconsistent or uncertain answers. See the
docstring on each constant for exactly what's uncertain.
"""

from __future__ import annotations

import copy
import random

from struggler.board import Board
from struggler.types import Action, Decision, DecisionKind, Observation, Region, Side

# Minimum DEFCON level required to attempt a coup in a region; regions not
# listed have no restriction. PROVISIONAL — the existence of this rule and
# its general shape (lower DEFCON restricts coups in more "sensitive"
# regions) is well established, but these exact threshold numbers are not
# independently verified against the physical rulebook.
COUP_MIN_DEFCON: dict[Region, int] = {
    Region.EUROPE: 3,
    Region.ASIA: 4,
}
_DEFAULT_MIN_DEFCON = 1

# Every coup attempt against a country in Europe additionally degrades
# DEFCON by 1, regardless of success. Realignment is NOT subject to the
# COUP_MIN_DEFCON restriction above (only coups are) — this is an
# assumption, flagged alongside COUP_MIN_DEFCON as an area to verify.


class Engine:
    def __init__(self, seed: int, board: Board | None = None) -> None:
        self.board = board if board is not None else Board()
        self.defcon = 5
        self.vp = 0
        self.turn = 1
        self.action_round = 1

        self._seed = seed
        self._rng = random.Random(seed)
        self._decision_stack: list[Decision] = []
        self._next_decision_id = 0
        self._winner: Side | None = None
        self._game_over_reason: str | None = None

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

    def observe(self, player: Side) -> Observation:
        if player not in (Side.US, Side.USSR):
            raise ValueError("observe() is only valid for Side.US or Side.USSR")
        return Observation(
            side=player,
            defcon=self.defcon,
            vp=self.vp,
            turn=self.turn,
            action_round=self.action_round,
            influence=copy.deepcopy(self.board.influence),
            pending_decision=self.pending_decision,
        )

    @property
    def is_terminal(self) -> bool:
        return self._winner is not None

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

    # -- dispatch -----------------------------------------------------------

    def _dispatch(self, decision: Decision, action: Action) -> None:
        handler = {
            DecisionKind.PLACE_INFLUENCE: self._handle_place_influence,
            DecisionKind.COUP_TARGET: self._handle_coup_target,
            DecisionKind.COUP_ROLL: self._handle_coup_roll,
            DecisionKind.REALIGNMENT_TARGET: self._handle_realignment_target,
            DecisionKind.REALIGNMENT_ACTOR_ROLL: self._handle_realignment_actor_roll,
            DecisionKind.REALIGNMENT_OPPONENT_ROLL: self._handle_realignment_opponent_roll,
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
        side = decision.actor
        country = action.payload["country"]
        cost = self.board.influence_cost(side, country)
        self.board.influence[country][side.value] += 1
        self._check_auto_win()
        self._maybe_push_place_influence(side, decision.context["ops_remaining"] - cost)

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

        if info.region is Region.EUROPE:
            self._change_defcon(-1, caused_by=side)

        self._check_auto_win()

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

        self._check_auto_win()
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

    def _check_auto_win(self) -> None:
        if self.is_terminal:
            return
        side = self.board.controls_all_of_europe()
        if side is not None:
            self._winner = side
            self._game_over_reason = "europe_control"


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
