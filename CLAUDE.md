# struggler

An API-first, deterministic rules engine for *Twilight Struggle* (GMT
Games, 2005), built so AI agents can be trained and evaluated against it.
Public, open source.

This file is the project contract. It defines the architecture, the
public API, and the milestones. It is binding: implementation work
should be checked against it, and any change to what's specified here
should update this file in the same change.

## Non-negotiable architectural mandates

These five points are the reason this project exists. Any
implementation that violates one of them is wrong, regardless of
whether it "passes the tests."

### 1. Pending-decision stack, not "one turn = one action"

The engine is a state machine. At any point it has zero or more
pending decisions, held on an internal **stack** (not a plain FIFO
queue) because card events *interrupt*: resolving a decision can push
new sub-decisions on top (e.g. a `DUCK_AND_COVER` event triggers a
"USSR: which country loses influence" decision mid-resolution of
something else), which must resolve before control returns to the
interrupted context.

- `pending_decision` always returns the top-of-stack frame.
- Resolving the top frame either pops back to the frame beneath it, or
  — if resolving it produced new sub-decisions — pushes those first.
- The engine never assumes whose decision is next; `Decision.actor`
  says so explicitly (`Side.US`, `Side.USSR`, or `Side.CHANCE`).

### 2. Atomic action space

Spending 4 Ops is four separate single-point-placement decisions, not
one action out of `C(countries, 4)`-many combinations. Target: **tens**
of legal options per decision, never thousands. If a legal-actions list
is ever in the hundreds, that decision is decomposed wrong and needs to
be broken down further.

### 3. Seeded, injectable RNG

`Engine(seed=...)` takes (or is handed) a seeded RNG. Same seed + same
action sequence → byte-identical resulting state, every time, on every
machine. Chance is never read from `random`/`os.urandom` directly
anywhere in the engine — always through the injected RNG.

Chance events (coup rolls, realignment rolls, headline-order ties,
future card effects that roll dice) are **exposed as decisions** with
`actor=Side.CHANCE`, not resolved silently inside `step()`. The engine
draws the outcome from its own seeded RNG and calls `step()` with it
internally, so the roll still appears as an explicit, logged
`(Decision, Action)` pair — this is what makes replay logs a complete,
re-playable record of a game, dice included.

### 4. Per-player observation function

`observe(player)` is the *only* sanctioned way an agent sees the game.
It must never leak:
- the opponent's hand (card identities — count is public),
- the draw pile's order or the identity of undrawn cards,
- any other information hidden from that seat under the rules (e.g. an
  opponent's not-yet-revealed China Card possession is public in TS,
  but anything analogous in future card effects must be modeled the
  same way — hidden fields simply absent from the returned view, never
  masked/zeroed in a way that leaks their existence via shape).

`observe()` is asymmetric by construction: `observe(Side.US)` and
`observe(Side.USSR)` are different objects, not the same object with a
"redact" flag.

### 5. Flat, serializable state

`GameState` is representable as a flat dict of JSON primitives (int,
str, bool, list, dict) with no custom encoder required. This is what
makes replay logs diffable, hashable, and greppable, and what makes
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
(important for both engine-internal logic and RL action-value tables)
over wire-nativeness. JSON-facing boundaries (`serialize()`, replay
logs) convert explicitly; nothing internal is a dict-in-disguise.

```python
class Side(Enum):
    US = "US"
    USSR = "USSR"
    CHANCE = "CHANCE"

class DecisionKind(Enum):
    PLACE_INFLUENCE = "place_influence"
    REALIGNMENT_ROLL = "realignment_roll"
    COUP_ROLL = "coup_roll"
    # ... extended per milestone; see M1-M3 below

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

## Card data policy

The reference repo (`glowsplint/twilight-struggle-py`) is a **data
source only**, and only for cross-checking — it currently has no
license (all rights reserved by default), so its files/code must not
be copied verbatim. The factual data we need — each card's ops value,
side (US/USSR/Neutral), deck (Early/Mid/Late War), and
remove-after-event flag — are facts about the published board game
itself, not that repo's expression of them. Re-enter this data
independently (source of truth: the physical card text / GMT's
published card list), store it as our own data file (e.g.
`data/cards.json`), and cite the physical game as the source in
comments/docs, not the reference repo.

Card *mechanics* (event text implementation) are out of scope until
M3, and when implemented, must be designed against mandates #1–#2
(decisions/actions), never adapted from the reference repo's
single-action model.

## Milestones

Each milestone is a hard gate: it must be fully correct and tested
before the next one starts.

### M1 — Board, no cards
Influence placement/removal, control determination, adjacency,
region scoring, DEFCON track (including DEFCON-1 loss condition),
coups, realignment rolls. No cards exist yet; "playing a card" isn't
modeled — Ops-only actions are driven directly for testing.
**Proves:** the decision stack, atomic placement, seeded RNG-as-decision,
and observation all work for the board mechanics that every later
milestone depends on.

### M2 — Full game, cards as Ops only, zero events fire
All 110 cards exist as data (see Card data policy) and can be played
for their Ops value; headline phase, space race, China Card, defcon
degradation from play all function. **No card event ever fires** —
every card play is forced/assumed Ops. This is the real proof the game
loop works end-to-end for a complete game.
**Proves:** a full game is playable start-to-finish through the public
API alone, with legal Ops-play decisions (event-vs-ops choice included,
even though event is never taken) correctly enumerated.

### M3 — Cards by mechanical difficulty
Introduce events in increasing order of implementation difficulty:
1. Pure state change (e.g. immediate, unconditional board effect).
2. Player-choice events (event enqueues a new decision for a player).
3. Persistent modifiers (event changes future legality/scoring, e.g.
   for the rest of the turn or game).
4. Rule-modifying events (event changes how other rules or cards
   themselves resolve).

Each tier is its own set of sub-milestones; a card is not "done" until
it has a replay-log regression test (see Testing strategy).

## Testing strategy

Testing is first-class, not an afterthought — this is a rules engine;
correctness *is* the product.

### Deterministic replay logs (primary strategy)
A replay log is a JSON file:
```json
{
  "seed": 12345,
  "actions": [ {"kind": "place_influence", "payload": {"country": "Poland"}}, ... ],
  "checkpoints": [ {"after_step": 10, "state": { ...full serialize() dict... } } ]
}
```
- Replaying `seed` + `actions` through a fresh `Engine` must reproduce
  every checkpoint's state exactly (`==` on the deserialized dict, not
  a hash — exact equality gives a diffable failure).
- Golden replay logs are checked into `tests/replays/` and grow with
  the project: each new mechanic (a coup, a realignment, later each
  card) gets at least one golden replay exercising it.
- Because chance is a logged `CHANCE` decision (mandate #3), replay
  logs are fully deterministic even through dice-driven mechanics —
  there is no separate "RNG trace" to keep in sync.

### Property-based invariant tests
Using `hypothesis` (or equivalent) to generate random *legal* action
sequences (always drawn from `legal_actions()`, never arbitrary
payloads) and assert invariants that must hold after every `step()`:
- DEFCON always in `[1, 5]`; game ends immediately at DEFCON 1.
- Influence values are never negative.
- `serialize()` → `deserialize()` → `serialize()` is idempotent.
- `observe(player)` never contains a key/value that reveals hidden
  information (checked structurally, not just spot-checked fields).
- `legal_actions()` is never empty while `pending_decision` is not
  `None` (the engine must never deadlock).

### Unit tests
Standard per-mechanic tests (e.g. "coup in a country at DEFCON 2 with
a effective defcon-adjustment card active behaves like X") for M1-era
board mechanics and each M3 card individually.

## Tooling and conventions

- **Python**: 3.12+.
- **Environment**: conda (`environment.yml` at repo root).
- **Tests**: `pytest`, plus `hypothesis` for property-based tests.
- **License**: MIT.
- **Language**: all code, comments, docstrings, and commit messages in
  English.
- **Layout**: `src/struggler/` package (src-layout to avoid accidental
  implicit imports of the working directory during tests); tests under
  `tests/`, golden replay logs under `tests/replays/`, card data under
  `data/`.
