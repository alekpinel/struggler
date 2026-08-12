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

#### M3 implementation status and framework

The event layer is **opt-in**, so it never regresses M2 (whose defining
proof is that *zero events fire*): `Engine.new_game(..., events=False)`
— the default — is the M2 game, byte-identical to before; `events=True`
turns the layer on. `serialize()` carries `events_enabled`,
`turn_effects` (per-turn modifiers) and `game_effects` (persistent
game-long effects), so a saved game round-trips its event state (mandate
#5; the M2 golden logs were regenerated once for these additive keys —
values only, no behavior change). As every card's event is implemented,
`events=True` moves toward becoming the default and the flag becomes the
historical "Ops-only" toggle.

- **Registry.** `src/struggler/events.py` maps a card id → an `Event`
  (`resolve(engine, side)`, plus an `eligible` predicate for
  preconditions). A card *absent* from the registry has no event yet: in
  events mode it is a no-op discard, exactly as in M2. This is what lets
  M3 grow card-by-card without touching the game loop.
- **Firing paths (mandates #1–#2).** An event fires when its owner (or a
  NEUTRAL card's player) plays it as its event; and — per the core TS
  rule — when the **opponent's** card is played for Ops, its event *also*
  fires, with the phasing player choosing the order via an
  `EVENT_OPS_ORDER` decision (`event_first`/`ops_first`). Ordering is
  implemented on the decision stack itself: an `EVENT_RESUME` marker is
  slipped beneath the first half's sub-decisions so the second half runs
  only after they drain. Dice inside events (the "war" family) are logged
  `WAR_ROLL` CHANCE decisions, never silent `random` calls (mandate #3).
- **Headline firing.** Non-scoring events now fire during the headline
  too. Headline resolution is stack-driven: both cards are chosen, their
  order is frozen (higher Ops first, ties US-first) into
  `_headline_pending`, and each card resolves in turn — if its event
  enqueues sub-decisions (e.g. a war roll) those drain before the next
  headline card resolves, the same interrupt order the action-round path
  uses. `serialize()` carries `headline_resolving`/`headline_pending`.
- **Player-choice steps (tier 2).** An event that lets a player distribute
  influence enqueues its own decisions through two generic, fully
  serializable step types: `EVENT_INFLUENCE` (place / remove / remove-all
  one country at a time, for N steps, honouring a per-country cap, a
  control filter, and an "uncontrolled only" filter; it re-pushes itself
  until N hits 0 or no legal target remains) and `EVENT_CHOICE` (a branch,
  e.g. Warsaw Pact's "remove or add", routed by `events.CHOICE_ROUTERS` so
  the stack stores only the event id and the chosen option — never a
  function). These steps live on the same decision stack, so they are
  hosted correctly inside a headline or an opponent's Ops play.
- **Implemented so far** (100 card events registered, plus Defectors via the
  headline hook and UN Intervention via its rule-modifier play mode — the
  entire non-scoring deck; the trickier ones have a dedicated unit test, and
  the loop is covered by a property test and the `m3_events.json` golden).
  Grouped by the primitive they reuse:
  - *Immediate fixed board/VP/DEFCON/Space effects:* Duck and Cover, Fidel,
    Nasser, Romanian Abdication, De Gaulle Leads France, Captured Nazi
    Scientist, Nuclear Test Ban, Allende, Portuguese Empire Crumbles,
    Panama Canal Returned, Sadat Expels Soviets, John Paul II Elected Pope,
    Camp David Accords, Iranian Hostage Crisis, The Iron Lady, An Evil
    Empire, U-2 Incident, Cultural Revolution, Ortega Elected, Tear Down
    This Wall, Kitchen Debates, OPEC, Alliance for Progress, Reagan Bombs
    Libya, One Small Step, AWACS Sale to Saudis.
  - *War family (attacker chosen, seeded CHANCE roll):* Korean War,
    Arab-Israeli War (fixed target); Indo-Pakistani War, Iran-Iraq War,
    Brush War (attacker picks the target via `WAR_TARGET`).
  - *Events that conduct Operations (`push_event_operations`):* CIA Created,
    Lone Gunman, ABM Treaty.
  - *Forced random discard (`RANDOM_DISCARD`, a seeded CHANCE decision that
    reveals only the drawn card):* Five Year Plan (a discarded USSR event
    fires), Terrorism (opponent discards, twice after Iranian Hostage
    Crisis).
  - *Per-turn coup/realign modifiers:* Nuclear Subs (US Battleground coups
    skip the DEFCON degrade), Latin American Death Squads (±1 to Americas
    coup rolls), SALT Negotiations (-1 to both sides' coups), Iran-Contra
    Scandal (-1 to US realignment via `_realignment_modifier`), Chernobyl
    (a chosen region bars USSR Ops influence, via `_chernobyl_blocks`).
    Set-DEFCON branch: How I Learned to Stop Worrying (`set_defcon` + 5
    military Ops).
  - *Persistent game-long triggers (`game_effects`):* Yuri and Samantha
    (USSR +1 VP per US coup, in `_handle_coup_roll`), Flower Power (USSR +2
    VP per US war-card play, via `_maybe_flower_power`, cancelled by An Evil
    Empire).
  - *Dice-contest / branch (`push_dice_contest` — both roll, ties reroll,
    higher wins, logged as `CONTEST_ROLL`):* Olympic Games (opponent
    boycotts or a +2 contest), Summit (regional-domination modifiers, winner
    takes 2 VP then adjusts DEFCON), Wargames (only at DEFCON 2: give the
    opponent 6 VP and final-score the game).
  - *Reclaim from the discard pile (`push_take_from_discard`):* SALT
    Negotiations (also DEFCON +2) — the player takes one non-scoring card
    from the public discard back to hand.
  - *Revealing/taking cards from the opponent's hand (the reveal is
    sanctioned by the card, so surfacing the involved cards as decision
    options is correct, not a leak):* Aldrich Ames Remix (USSR discards a
    chosen US card), Grain Sales to Soviets (one random USSR card revealed
    via a CHANCE step, the US takes it for its Ops or returns it for 2),
    Ask Not… (discard any own cards and redraw as many, via
    `draw_cards_to_hand`), The Cambridge Five (place in a region whose
    scoring card the US holds).
  - *Per-turn regional Ops bonus:* Vietnam Revolts — generalizes the China
    Card's all-in-region +1 into a reusable "bonus region" (`_ops_bonus_region`
    / `_in_bonus_region`); the China Card is "asia", Vietnam Revolts sets a
    turn effect giving USSR plays "se_asia".
  - *Influence then an optional free operation (`push_free_coup_or_realign`):*
    Junta — place 2 Influence in the Americas, then optionally a free Coup or
    Realignment there (the free-op choice is stacked beneath the placement so
    it resolves afterwards).
  - *Player-choice influence (`EVENT_INFLUENCE`):* COMECON, Marshall Plan,
    Decolonization, Suez Crisis, Truman Doctrine, Warsaw Pact Formed
    (branch), Socialist Governments, Muslim Revolution, Colonial Rear
    Guards, Liberation Theology, The Voice of America, Puppet Governments,
    OAS Founded, Pershing II Deployed, The Reformer, Solidarity, Marine
    Barracks Bombing; Independent Reds (match-influence branch).
  - *Persistent per-turn modifiers:* Containment, Brezhnev Doctrine, Red
    Scare/Purge (consulted via `_effective_ops`, cleared at end of turn).
  - *Persistent game-long legality (`game_effects`):* NATO (eligible only
    after Marshall Plan or Warsaw Pact; USSR may no longer coup/realign
    US-controlled Europe), De Gaulle and Willy Brandt (each lift NATO for
    one country), US/Japan Mutual Defense Pact (locks Japan), The Reformer
    (bars USSR coups in Europe). Enforced in `_usable_coup_realign_target`
    (which distinguishes coup from realignment for The Reformer), consulted
    by both target enumerations. Eligibility flags also gate Arab-Israeli
    War (Camp David), Socialist Governments (Iron Lady) and Solidarity
    (John Paul II).
  - *Rule-modifier (tier 4):* UN Intervention — a `un_intervention` play
    mode that spends the held UN Intervention card to use an opponent's
    (implemented, eligible) event card for Ops with its event cancelled.
  - *Take-and-play from a hand or the discard pile:* Missile Envy
    (`missile_envy_take`/`missile_envy_use` — take the opponent's highest-Ops
    card, opponent breaks ties; use it for Ops, or its Event when it is neutral
    or the taker's own; Missile Envy itself passes to the opponent's hand),
    Star Wars (`play_card_from_discard` — eligible only while the US leads the
    Space Race; take a non-scoring discard and fire its event now).
  - *Free coup with a conditional repeat:* Che (`push_che_coup`/`begin_che_coup`
    — a free USSR coup against a non-Battleground Central/South America/Africa
    target; a second one against a different such country if the first removed
    US Influence, capped at two via the `che` context on the `COUP_ROLL`).
  - *Deferred per-turn conditions:* Cuban Missile Crisis (DEFCON→2; a coup by
    the flagged side loses the game, checked in `_handle_coup_roll`; the at-risk
    side may defuse by removing 2 Influence from Cuba/West Germany), We Will
    Bury You (DEFCON −1; USSR +3 VP at end of turn unless the US plays UN
    Intervention, which clears the `we_will_bury_you` turn effect).
  - *Scoring-time modifiers / extra rounds (`_scoring_overrides`,
    `_total_action_rounds`/`_side_for_play_index`):* Formosan Resolution (Taiwan
    scores as an Asian Battleground while the US controls it; nullified once the
    China Card is played), Shuttle Diplomacy (one USSR-controlled Battleground is
    dropped at the next Middle East/Asia scoring, then consumed), North Sea Oil
    (OPEC becomes ineligible game-long; the US plays one extra action round this
    turn). `board.region_tier` gained optional `extra_battlegrounds`/`ignored`
    sets so these scoring adjustments are additive and leave every other caller
    unchanged.
  - *A further batch on the existing primitives:* East European Unrest and South
    African Unrest (player-choice influence — `push_event_influence` gained an
    `amount` per selection for the Late-War 2-per-country removal), Blockade and
    Latin American Debt Crisis (the "US discards a printed-3+-Ops card or suffer"
    branch, the US choosing from its own hand as Ask Not already does),
    Glasnost (VP/DEFCON, then 4 Ops if The Reformer is active), Soviets Shoot
    Down KAL-007 (VP/DEFCON, then 4 Ops if the US controls South Korea), Ussuri
    River Skirmish (take the China Card from the USSR, or +4 Influence in Asia),
    Arms Race (score off the Military Operations track), De-Stalinization (a
    relocate flow: remove up to 4 USSR Influence, then replace it in
    non-US-controlled countries, max 2 each).
  - *The last four subsystems, closing out the deck:*
    - A persistent reactive hook consulted from board mechanics rather than
      another event: NORAD (`game_effects["norad"]`, checked in
      `Engine._change_defcon` — every time DEFCON *moves* to level 2, the US
      adds 1 Influence to a country where it already has some, via
      `_push_norad_influence`; a stable DEFCON 2 does not refire it).
    - Two immediate conditionals: Special Relationship (2 VP while the US
      Controls the UK — the card's own eligibility predicate — plus one free
      Realignment roll via the new `push_free_realignment` if NATO is also in
      effect), Nixon Plays the China Card (eligible only while the USSR holds
      the China Card; the USSR either discards a non-scoring card to keep it,
      or the US takes it face down/unusable this turn).
    - A hidden peek at the draw pile: Our Man in Tehran. The examined cards
      live in `Engine._our_man_queue`/`_our_man_kept` — plain serialized state
      (mandate #5) deliberately excluded from `observe()` — while the
      `EVENT_CHOICE` decision offered to the US only ever contains
      `"keep"`/`"remove"`, never the card identity, so the opponent's
      observation cannot infer which card is under consideration even though
      `pending_decision` is otherwise shared (mandate #4). Kept cards return to
      the draw pile and it is reshuffled through the seeded RNG once all (up
      to 5) cards are decided.
    - A headline-cancellation interaction: Defectors has no `EVENTS` entry —
      its entire effect only makes sense at headline time (a documented
      restriction of the physical card), so it is implemented purely as a
      headline-order hook, `_apply_defectors_headline`, called from
      `_advance_once` once both headline picks are frozen: a US headline of
      Defectors discards the USSR's headlined card unresolved; a USSR headline
      of Defectors instead gives the US 1 VP. Playing it in an ordinary action
      round is therefore correctly a no-op discard, exactly like an
      unimplemented event.
    - A persistent per-player operating lock: Bear Trap (traps the USSR) and
      Quagmire (traps the US) — independent of who actually plays the card,
      the same way Duck and Cover always favors the US regardless of who plays
      it. `Engine._trap_key_for`/`_push_trap_step`, hooked into the
      action-rounds branch of `_advance_once`, replace the trapped side's
      normal `ACTION_ROUND_PLAY` with a mandatory-when-possible discard of an
      Ops-2+ card (`QUAGMIRE_DISCARD`) followed by a seeded `QUAGMIRE_ROLL`
      CHANCE die that frees the side on a 5–6; with no legal card to discard,
      that action round is simply wasted (no decision offered at all) — a
      documented simplification of that edge case.
- **China Card bonus.** Playing the China Card for Ops grants its +1
  ("all Ops used in Asia") for influence (an all-or-nothing invariant in
  the placement step: the 5th point is offered only while nothing has
  gone outside Asia) and for coups (+1 Op and +1 military Op against an
  Asian target). The realignment case is not modeled (rare).
- **Known limitations** (tracked here as the contract): every non-scoring
  card in the deck now has an implemented event (or, for Defectors and UN
  Intervention, an equivalent mechanism outside the `EVENTS` registry — see
  above). M3's remaining work is fidelity, not coverage: a still-growing set
  of documented rough edges and unconfirmed numeric/mechanical details, not
  missing subsystems.
  - *Documented simplifications* (consistent with the rest of M3, noted in
    `events.py` docstrings at the card in question): Missile Envy does not
    force the opponent to play the received Missile Envy card on their next
    action round (it is simply added to their hand); Cuban Missile Crisis
    offers its defuse immediately rather than at any later point in the turn;
    Shuttle Diplomacy is filed to the discard when played rather than kept "in
    front of you" (only the effect flag matters); Star Wars fires the taken
    card's event exactly like a normal event play; Glasnost's and Soviets
    Shoot Down KAL-007's follow-up Operations are offered as full Operations
    (Influence/Coup/Realignment) rather than the card's narrower
    Coup/Realignment or Influence/Realignment wording; a trapped side
    (Bear Trap/Quagmire) with no Ops-2+ card simply wastes that action round,
    with no roll offered. Ortega's free coup, Tear Down This Wall's
    Operations, and Junta's "single country" are older examples of the same
    pattern.
  - *VERIFY: reconstructed from memory, not independently reconfirmed against
    the physical card text here* (flagged in `events.py` at the card, the same
    convention `board.py`/`engine.py` already use for unconfirmed numeric
    constants like `SPACE_RACE_BOXES`): Special Relationship's and Nixon Plays
    the China Card's exact wording. Re-verifying these against the physical
    cards (or GMT's published card list) and correcting `events.py` if they're
    off is the concrete next step, not new subsystem work.
  - Still unmodeled generally: the Space Race headline-reveal perk and the
    region bonus for realignment.

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
