# Architecture

`struggler` is a state machine with a public API narrow enough that an
agent — human, scripted, or learned — can only interact with the game the
way a player at the table can.

The five mandates below are the reason this project exists. An
implementation that violates one of them is wrong, regardless of whether
it passes the tests.

## The five mandates

### 1. Pending-decision stack, not "one turn = one action"

At any point the engine has zero or more pending decisions, held on an
internal **stack** (not a plain FIFO queue) because card events
*interrupt*: resolving a decision can push new sub-decisions on top (e.g.
a `DUCK_AND_COVER` event triggers a "USSR: which country loses influence"
decision mid-resolution of something else), which must resolve before
control returns to the interrupted context.

- `pending_decision` always returns the top-of-stack frame.
- Resolving the top frame either pops back to the frame beneath it, or —
  if resolving it produced new sub-decisions — pushes those first.
- The engine never assumes whose decision is next; `Decision.actor` says
  so explicitly (`Side.US`, `Side.USSR`, or `Side.CHANCE`).

### 2. Atomic action space

Spending 4 Ops is four separate single-point-placement decisions, not one
action out of `C(countries, 4)`-many combinations. Target: **tens** of
legal options per decision, never thousands. If a legal-actions list is
ever in the hundreds, that decision is decomposed wrong and needs to be
broken down further.

**Exception**: physical mode's `DEAL_CARD` and the physical-hand-sourced
options on a few other kinds (`ACTION_ROUND_PLAY`/`HEADLINE_PLAY`/
`RANDOM_DISCARD`/`QUAGMIRE_DISCARD`/`HELD_CARD_DISCARD`, plus a few
`EVENT_CHOICE` candidate lists) may run into the hundreds early in a game.
This is narrowly scoped to decisions that only ever reach the physical-mode
operator console (see [BOTS.md](BOTS.md)) — never a bot/RL `Player`, which
is what this mandate exists to keep tractable for. The console presentation
layer never dumps a giant numbered menu (it matches free text against a
card's printed number or name instead); the `Decision.options`/
`legal_actions()` contract itself is unchanged — still the literal,
exhaustive, replay-log-faithful legal set.

### 3. Seeded, injectable RNG

`Engine(seed=...)` takes (or is handed) a seeded RNG. Same seed + same
action sequence → byte-identical resulting state, every time, on every
machine. Chance is never read from `random`/`os.urandom` directly anywhere
in the engine — always through the injected RNG.

Chance events (coup rolls, realignment rolls, headline-order ties, card
effects that roll dice) are **exposed as decisions** with
`actor=Side.CHANCE`, not resolved silently inside `step()`. The engine
draws the outcome from its own seeded RNG and calls `step()` with it
internally, so the roll still appears as an explicit, logged
`(Decision, Action)` pair — this is what makes replay logs a complete,
re-playable record of a game, dice included.

### 4. Per-player observation function

`observe(player)` is the *only* sanctioned way an agent sees the game. It
must never leak:

- the opponent's hand (card identities — count is public),
- the draw pile's order or the identity of undrawn cards,
- any other information hidden from that seat under the rules (e.g. an
  opponent's not-yet-revealed China Card possession is public in Twilight
  Struggle, but anything analogous in card effects must be modeled the same
  way — hidden fields simply absent from the returned view, never
  masked/zeroed in a way that leaks their existence via shape).

`observe()` is asymmetric by construction: `observe(Side.US)` and
`observe(Side.USSR)` are different objects, not the same object with a
"redact" flag.

### 5. Flat, serializable state

`GameState` is representable as a flat dict of JSON primitives (int, str,
bool, list, dict) with no custom encoder required. This is what makes
replay logs diffable, hashable, and greppable, and what makes
`serialize()`/`deserialize()` trivial to keep in sync — the wire format
*is* the internal shape, not a projection of it.

## Public API surface

```python
class Engine:
    def __init__(self, seed: int, ...): ...

    @property
    def pending_decision(self) -> Decision | None:
        """Top of the decision stack. None iff the game has ended."""

    def legal_actions(self) -> tuple[Action, ...]:
        """Legal actions for the CURRENT pending_decision only."""

    def step(self, action: Action) -> None:
        """
        Apply `action` to the current pending_decision, advance state,
        and update the decision stack (pop, or push interrupts).
        Raises if `action` is not in legal_actions().
        """

    def observe(self, player: Side) -> Observation:
        """Player-scoped view. See mandate #4."""

    def serialize(self) -> dict:
        """Flat, JSON-primitive dict. Full state, including RNG state."""

    @classmethod
    def deserialize(cls, data: dict) -> "Engine":
        """Inverse of serialize(); must round-trip exactly."""

    @property
    def is_terminal(self) -> bool: ...

    @property
    def winner(self) -> Side | None: ...
```

### Core types (dataclasses/enums)

Decisions and actions are typed dataclasses, not dicts — this is the
project's chosen tradeoff of type-safety and pattern-matchability
(important for both engine-internal logic and RL action-value tables) over
wire-nativeness. JSON-facing boundaries (`serialize()`, replay logs)
convert explicitly; nothing internal is a dict-in-disguise.

```python
class Side(Enum):
    US = "US"
    USSR = "USSR"
    CHANCE = "CHANCE"

class DecisionKind(Enum):
    PLACE_INFLUENCE = "place_influence"
    REALIGNMENT_ROLL = "realignment_roll"
    COUP_ROLL = "coup_roll"
    # ... see struggler.engine.types for the full set

@dataclass(frozen=True)
class Action:
    kind: DecisionKind
    payload: Mapping[str, Any]  # e.g. {"country": "Poland"}

@dataclass(frozen=True)
class Decision:
    id: int                       # monotonic, unique within a game
    actor: Side
    kind: DecisionKind
    options: tuple[Action, ...]   # == legal_actions() for this frame
    context: Mapping[str, Any]    # e.g. {"ops_remaining": 2}
```

`Decision.options` and `Engine.legal_actions()` return the same data;
`legal_actions()` exists as the ergonomic accessor.

## Reachability within one Operations spend (rule 6.1.1)

Placing Influence is atomic per point (mandate #2), but legality is *not*
re-derived from the live board after each point: "all markers must be
placed with, or adjacent to, friendly markers that were in place at the
start of the phasing player's Action Round" — a point placed earlier in the
same Ops spend does not itself unlock a further-away country later in that
same spend.

`Engine._ops_round_snapshot` freezes `board.influence` the moment an
Ops-driven placement chain begins (`_maybe_push_place_influence` /
`_maybe_push_bonus_influence`, threaded into `Board.is_reachable` via its
optional `influence` override), is reused for every point in that chain,
and clears once it ends. It round-trips through
`serialize()`/`deserialize()`, so a save/resume mid-chain cannot reopen the
chaining bug.

Event-driven placement (`push_event_influence`) is unaffected — rule
6.1.1's own exception excludes it, and it was never reachability-gated in
the first place, since its candidates are fixed lists rather than
adjacency-derived.

## The Ops-only toggle

`Engine.new_game(..., events=False)` runs the game with the card-event
layer switched off entirely: all 110 cards exist as data and are playable
for their Ops value, the headline phase, space race, China Card and DEFCON
degradation all function, but **no card event ever fires** — every card
play is Ops. `events=True` is the default.

The toggle exists because "a complete game is playable start to finish
through the public API alone, with no event mechanics involved" is a
property worth being able to test in isolation. `serialize()` carries
`events_enabled` alongside `turn_effects` and `game_effects`, so a saved
game round-trips its event state either way (mandate #5).
