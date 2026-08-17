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

**Exception**: physical mode's `DEAL_CARD` and the physical-hand-sourced
options on a few other kinds (`ACTION_ROUND_PLAY`/`HEADLINE_PLAY`/
`RANDOM_DISCARD`/`QUAGMIRE_DISCARD`/`HELD_CARD_DISCARD`, plus a few M3
`EVENT_CHOICE` candidate lists) may run into the hundreds early in a
game. This is narrowly scoped to decisions that only ever reach the
physical-mode operator console (see "Bot framework" below) — never a
bot/RL `Player`, which is what this mandate exists to keep tractable for.
The console presentation layer never dumps a giant numbered menu (it
matches free text against a card's printed number or name instead); the
`Decision.options`/`legal_actions()` contract itself is unchanged — still
the literal, exhaustive, replay-log-faithful legal set.

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
`src/struggler/data/cards.json`), and cite the physical game as the
source in comments/docs, not the reference repo.

Card *mechanics* (event text implementation) are out of scope until
M3, and when implemented, must be designed against mandates #1–#2
(decisions/actions), never adapted from the reference repo's
single-action model.

Each card entry also carries `event_summary`: a short, hand-maintained
paraphrase of what `events.py` actually does mechanically for that
card (used by the LLM prompt, see "LLM reasoning layer" below), `null`
for a card with no implemented event yet. Unlike the other fields,
this one is *not* a fact about the physical game — it is engine-derived
documentation of `events.py`'s behavior, kept in `cards.json` because
it belongs with the rest of a card's data, not because it is sourced
from the physical card text. It can drift from `events.py` as M3
evolves; there is no automated sync check in v1 (documented known
limitation, see `bots/llm/player.py`).

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

**Reachability within one Operations spend (rule 6.1.1).** Placing
Influence is atomic per point (mandate #2), but legality is *not*
re-derived from the live board after each point: "all markers must be
placed with, or adjacent to, friendly markers that were in place at the
start of the phasing player's Action Round" — a point placed earlier in
the same Ops spend does not itself unlock a further-away country later in
that same spend. `Engine._ops_round_snapshot` freezes `board.influence`
the moment an Ops-driven placement chain begins (`_maybe_push_place_influence`
/ `_maybe_push_bonus_influence`, threaded into `Board.is_reachable` via its
optional `influence` override), is reused for every point in that chain,
and clears once it ends; it round-trips through `serialize()`/`deserialize()`
so a save/resume mid-chain can't reopen the chaining bug. Event-driven
placement (`push_event_influence`) is unaffected — rule 6.1.1's own
exception excludes it, and it was never reachability-gated in the first
place (its candidates are fixed lists, not adjacency-derived).

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
    via a CHANCE step, the US plays it in full — Event or Ops — or returns it
    for 2 Ops of its own; an empty USSR hand grants the US 2 Ops directly),
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
    or the taker's own; Missile Envy itself passes to the opponent's hand,
    which must spend its next action round playing it for Ops —
    `game_effects["missile_envy_forced"]`), Star Wars (`play_card_from_discard`
    — eligible only while the US leads the
    Space Race; take a non-scoring discard and fire its event now).
  - *Free coup with a conditional repeat:* Che (`push_che_coup`/`begin_che_coup`
    — a free USSR coup against a non-Battleground Central/South America/Africa
    target; a second one against a different such country if the first removed
    US Influence, capped at two via the `che` context on the `COUP_ROLL`).
  - *Deferred per-turn conditions:* Cuban Missile Crisis (DEFCON→2; a coup by
    the flagged side loses the game, checked in `_handle_coup_roll`; the at-risk
    side may defuse — Cuba for the USSR, West Germany or Turkey for the US —
    offered fresh at the start of each of its action rounds for the rest of
    the turn via `Engine._push_cmc_defuse_offer`), We Will Bury You (DEFCON
    −1; USSR +3 VP at end of turn unless the US plays UN Intervention, which
    clears the `we_will_bury_you` turn effect).
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
    - A headline-cancellation interaction, plus a separate action-round
      trigger: Defectors has no `EVENTS` entry — neither of its two printed
      clauses is an ordinary `resolve(engine, side)` event. Headlined by the
      US, `_apply_defectors_headline` (called from `_advance_once` once both
      headline picks are frozen, since it must act before either headline
      card resolves) discards the USSR's headlined card unresolved. Played by
      the USSR in a normal action round — Event or Ops, not Space Race —
      `_maybe_defectors_action_round` (hooked into `_handle_play_mode`
      alongside Flower Power) instead gives the US 1 VP. The USSR headlining
      it, or the US playing it in an action round, have no printed effect and
      are correctly no-ops. (This was flipped in an earlier pass — the VP
      previously fired on a USSR *headline* instead of a USSR action-round
      play — corrected after re-confirming the physical card's wording.)
    - A persistent per-player operating lock: Bear Trap (traps the USSR) and
      Quagmire (traps the US) — independent of who actually plays the card,
      the same way Duck and Cover always favors the US regardless of who plays
      it. `Engine._trap_key_for`/`_push_trap_step`, hooked into the
      action-rounds branch of `_advance_once`, replace the trapped side's
      normal `ACTION_ROUND_PLAY` with a mandatory-when-possible discard of an
      Ops-2+ card (`QUAGMIRE_DISCARD`) followed by a seeded `QUAGMIRE_ROLL`
      CHANCE die that frees the side on a 1–4 (confirmed against the physical
      card text, #44); with no legal card to discard, that action round is
      simply wasted with no roll at all — except a scoring card in hand must
      still be played (a scoring card may never be held past end of turn).
- **China Card bonus.** Playing the China Card for Ops grants its +1
  ("all Ops used in Asia") for influence (an all-or-nothing invariant in
  the placement step: the 5th point is offered only while nothing has
  gone outside Asia), for coups (+1 Op and +1 military Op against an Asian
  target), and for realignment (one extra roll, offered only while every
  attempt this Ops-spend has targeted Asia — the same all-or-nothing rule,
  in `_maybe_push_realignment_target`).
- **Known limitations** (tracked here as the contract): every non-scoring
  card in the deck now has an implemented event (or, for Defectors and UN
  Intervention, an equivalent mechanism outside the `EVENTS` registry — see
  above). M3's remaining work is fidelity, not coverage: a still-growing set
  of documented rough edges and unconfirmed numeric/mechanical details, not
  missing subsystems.
  - *Re-verified against the physical card text and corrected* (this was a
    full audit pass, not spot checks): Special Relationship and Nixon Plays
    The China Card were both substantively wrong, not just imprecise (see
    below); Grain Sales to Soviets' "take" branch now plays the taken card in
    full — Event or Ops via `Engine.push_full_card_play`, the ordinary
    Event/Ops/Space-Race choice — instead of forcibly spending only its Ops
    value, and its "USSR hand empty ⇒ US gets 2 Ops" branch (previously
    unreachable) now fires; Glasnost's and Soviets Shoot Down KAL-007's
    follow-up Operations are now restricted to Influence/Realignment
    (`push_event_operations(..., allow_coup=False)`), never Coup, per their
    printed text; Ortega Elected in Nicaragua's free Coup-only op (against a
    Nicaragua neighbor) and Tear Down This Wall's free Coup-or-Realignment op
    (in Europe) are now modeled via `push_free_coup_or_realign`; Junta places
    its +2 Influence in a *single* chosen country (`amount=2, remaining=1`),
    not split across two; NORAD now gates on Canada being US-controlled (its
    printed precondition); OPEC's country list had a phantom 8th field
    (Nigeria) not on the physical card, removed; Southeast Asia Scoring now
    weighs Thailand at 2 VP, other countries at 1 (was flat 1 VP each);
    Summit's dice contest no longer rerolls ties (`push_dice_contest(...,
    reroll_ties=False)`) — its printed text explicitly says not to, unlike
    Olympic Games, which still does; One Small Step withholds VP for the
    first of its two Space Race steps (`advance_space_race_box(...,
    award_vp=False)`), scoring only the second, per its own wording; NATO's
    protection of US-controlled Europe now also covers Brush War, not just
    Coup/Realignment (`Engine._nato_protects`, shared with
    `_usable_coup_realign_target`); Chernobyl's region choice is now always
    made by the US (`Side.US` hardcoded, not the phasing `side`) — the printed
    card names the US specifically, and the old code let the USSR pick its
    own block when it played the card for Ops; Formosan Resolution is now
    nullified only when the *US* plays The China Card, not whenever the card
    changes hands either way; Quagmire now also nullifies NORAD, per its own
    printed text (mirrors NORAD's own "nullified by Quagmire" line); a trapped
    side (Bear Trap/Quagmire) with no Ops-2+ card still must play any scoring
    card in hand, but no longer rolls in that case (see the escape-roll
    correction below); The Cambridge Five is now blocked during Late War
    (`turn < 8`), per its printed restriction.
  - *Fixed in a follow-up pass*: Missile Envy now forces its recipient to
    spend their next action round playing it for Operations
    (`game_effects["missile_envy_forced"]`, enforced in
    `_push_action_round_play`/`_handle_action_round_play` — yields to a
    scoring-card deadline if both apply at once, since that one is a hard
    constraint); Cuban Missile Crisis's defuse is now offered fresh at the
    start of *every* one of the trapped side's action rounds for the rest of
    the turn (`Engine._push_cmc_defuse_offer`, wired into the turn loop via
    `Engine._dispatch_action_round`), not just once immediately, and the US
    branch now correctly offers Turkey as an alternative to West Germany (the
    physical card names both; the engine previously only modeled West
    Germany). Star Wars was re-examined and found already correct — it was
    flagged as a "simplification" in error; its behavior (event-only, no
    scoring cards, filed exactly like a normal event play) already matches
    the printed card. Defectors' +1 VP clause was inverted: it had fired on
    the USSR *headlining* the card, but the printed text ties it to the USSR
    playing it in a normal action round (Event or Ops, not Space Race)
    instead — the headline clause is Defectors-cancels-headline only, with no
    separate USSR-headline effect (`Engine._maybe_defectors_action_round`,
    hooked into `_handle_play_mode` next to Flower Power; corrected after
    the user supplied the exact physical card text, #103).
  - *Documented simplifications that remain* (noted in `events.py`
    docstrings at the card in question): Shuttle Diplomacy is filed to the
    discard when played rather than kept "in front of you" until its delayed
    effect triggers (only the effect flag matters — a card-manipulation event
    like Star Wars could in principle retrieve it slightly earlier than the
    physical game allows, but the effect it would re-apply is idempotent, so
    this has no actual gameplay consequence); Aldrich Ames Remix's "USA
    reveals their hand face-up until end of turn" is a momentary reveal (the
    decision options), not an ongoing visibility grant surfaced through
    `observe()` — deferred because it would add a new hidden/shared-visibility
    field to the public `Observation` API surface, a larger change than a
    card-logic fix.
  - *Resolved*: Bear Trap/Quagmire's escape-roll direction. Confirmed against
    the physical card text (#44: "On the next action round, [side] must
    discard an Operations card worth 2 or more and roll 1-4 to cancel this
    event"): **1-4 frees the trapped side, 5-6 leaves it trapped** — the
    engine previously had this backwards (freed on 5-6). Also corrected in
    the same pass: with no Ops-2+ card to discard, the round is now wasted
    with *no* roll at all (the trap persists untouched) rather than the
    previous behavior of forcing a roll anyway; a scoring card in hand is
    still forced into play regardless (a scoring card may never be held past
    end of turn — the one exception), but that no longer implies a roll or a
    "remaining rounds skipped this turn" flag, since the physical card never
    described one.
  - Still unmodeled generally: the Space Race box 4 headline-reveal-order
    perk (requiring the opponent to select their Headline Event first).
    Box 6 (may discard the Held Card at end of turn) and box 8 (an extra
    Action Round) are implemented, granted
    only to the first side to reach the box and cancelled outright — not
    transferred — the instant the second side also reaches it (6.4.4), via
    `Engine._update_space_race_ability` and the `game_effects` keys
    `space_race_discard_holder` / `space_race_extra_round_holder`.
  - *Physical mode* (see "Bot framework" below): every hand-touching event
    is now wired for a physical hidden hand, including the three where the
    *deciding* side needs to inspect the *opponent's* hand — Aldrich Ames
    Remix, Missile Envy (`choose_side` overridden to `Side.CHANCE` so the
    operator, not the bot, answers), and The Cambridge Five (a per-scoring-
    card yes/no query sequence instead of a single choice). Our Man in
    Tehran remains a documented no-op under physical mode, since it needs
    the draw pile's real contents, which physical mode makes unknown to the
    engine itself, not just hidden from a player.

## Bot framework

The engine's job is to be a fair arbiter, not to know who's playing. Every
seat — human or bot — plugs in through the same interface, so human-vs-human,
human-vs-bot, and bot-vs-bot are one code path, and adding a new bot never
touches the engine.

### The `Player` interface

`struggler.engine.player.Player` is a structural `Protocol`, not a base
class: any object with a matching `choose_action` method is a `Player`, no
inheritance required (mandate-consistent with the rest of this file's
API-surface philosophy — the contract is a shape, not a class hierarchy).

```python
class Player(Protocol):
    def choose_action(self, observation: Observation, history: Sequence[Event]) -> Action:
        """Pick one action from `observation.pending_decision.options`."""
```

- A player only ever sees `observe(side)` (mandate #4) and returns one
  `Action` drawn verbatim from `pending_decision.options` (mandate #2) —
  the same constraints a human at the console has.
- `history` is every resolved `(Decision, Action)` pair since this player
  was last consulted (opponent moves and CHANCE rolls included), as a
  `engine.player.Event` list. Bots are free to ignore it; it exists so a
  player *can* condition on what just happened without re-deriving it from
  `Observation` alone.
- `Side.CHANCE` decisions (coup/realignment/space-race rolls, ...) never
  reach a `Player` at all in an ordinary game — `struggler.runner.play_game`
  resolves them directly from the pre-drawn single option `Decision.options`
  already carries (mandate #3: the roll already happened via the engine's
  seeded RNG; there is nothing left to decide). The one exception is
  physical mode (below), where a `Side.CHANCE` entry in `players` is the
  operator console.
- `struggler.runner.play_game(engine, {Side.US: ..., Side.USSR: ...})` runs
  a game to completion, building the shared `Event` history and dispatching
  each non-CHANCE decision to the registered `Player` for that decision's
  `actor`.

### Building players

There is no registry: `src/main.py`'s `build_player(kind, *, seed=0)` is a
plain `if`/`elif` over kind names (`"human"`, `"first"`, `"random"`,
`"greedy"`, `"llm"`), each branch constructing the corresponding `Player`
directly (`HumanPlayer`, `FirstLegalPlayer`/`RandomPlayer` from
`bots/naive.py`, `GreedyPlayer` from `bots/greedy.py`, `LLMPlayer` from
`bots/llm/player.py` — the `"llm"` branch also picks a provider client via
`STRUGGLER_LLM_PROVIDER`/`STRUGGLER_LLM_MODEL`). Adding a new bot means
implementing `Player` and adding one branch to `build_player` — no
self-registration, no import-order dependency, no indirection between a
name and the class it builds.

### Physical mode

`Engine.new_game(..., physical_mode=True, physical_side=Side.US | Side.USSR)`
lets one seat be a real human playing the physical board game, with the
engine as referee/state-tracker — the setup for testing a bot/AI against a
physical opponent. `physical_mode` is a construction-time `Engine` flag, not
a `build_player` kind, because it changes engine behavior (dealing, dice),
not just which `Player` answers decisions; `src/main.py`'s `--physical
{us,ussr}` flag builds the engine this way (the bot side is still built
normally from `--us`/`--ussr`).

Two things the engine cannot know on its own once one seat is physical:

- **The physical side's hand is genuinely unknown to the engine itself**
  (not merely hidden from the opponent via `observe()`, mandate #4's usual
  guarantee) — there's no seeded RNG that can predict what a real shuffle
  dealt. `Engine.hidden_pool` (a plain `list[str]`, mandate #5) tracks real
  card ids not yet matched to a known location; the physical hand's own
  entries are the `HIDDEN_CARD` (`"?"`) sentinel until a card is revealed
  (played, discarded, or otherwise disclosed by an event), at which point
  `Engine.declare_physical_card`/`_reveal_in_hand`/`_hand_remove_known`
  move it out of the pool for good.
- **Every dice roll — both sides' — is entered manually**, since there is
  one physical board and real dice are used for every roll on it,
  regardless of which side nominally triggered it. Each `*_ROLL` call site
  uses `Engine._d6_actions`, which — under `physical_mode` — pushes all six
  possible outcomes as `Decision.options` (still `actor=Side.CHANCE`)
  instead of one pre-drawn value (mandate #3: chance is still fully exposed
  as a decision, just resolved by a human instead of the seeded RNG).

Because there is a **single shared physical deck**, the non-physical side's
hand can't be auto-dealt by the seeded shuffle either: the operator declares
it card by card too (`DecisionKind.DEAL_CARD`, `actor=Side.CHANCE`, since
dealing isn't a strategic choice). The physical side's own hand is topped
up silently (nothing new is *learned* by the engine, still just a count).

All of this — both hands' dealing, every dice roll, and the physical side's
own moves — is answered by one `struggler.engine.physical.OperatorConsolePlayer`
instance, registered in `players` under **both** `physical_side` and
`Side.CHANCE`; `runner.play_game` routes to it accordingly. The bot side's
own `Player` is completely untouched — it still only ever computes its own
strategic decisions from `Observation`/`history`, unaware anything is
different about this game.

**Known limitations** (documented simplifications, not missing
subsystems): the engine can't enforce the "must play a scoring card" rule
for a hand it can't see the true contents of, so every not-yet-accounted-for
card is offered at `ACTION_ROUND_PLAY` regardless — the physical player
(who can see their own hand) is trusted to honor that rule themselves, the
same trust model any human player already gets for rules `HumanPlayer`
doesn't independently re-verify. UN Intervention's mode is never offered to
a physical-side player (the engine can't verify hidden-hand membership of
a specific card), which fails safe (no crash, just an unavailable option)
rather than being fixed here. Our Man in Tehran is a documented no-op
under `physical_mode`: it peeks at the *draw pile's* real contents, which
physical mode makes unknown to the engine itself, not merely hidden from
a player, so there is nothing to queue instead of `HIDDEN_CARD`
placeholders. Every other M3 event is wired for a physical hand,
including the three that need the *deciding* side's actor overridden to
`Side.CHANCE` (the operator, not a bot that cannot see the target hand,
answers): Aldrich Ames Remix and Missile Envy's giver-side pick source
candidates from `_physical_hand_candidates`/route the choice to
`Side.CHANCE` the same way `DEAL_CARD` does; The Cambridge Five instead
asks one per-scoring-card yes/no `EVENT_CHOICE` query at a time
(`_push_cambridge_five_query`), stopping the moment the hand's last open
`HIDDEN_CARD` slot is filled (asking further would have nowhere left to
reveal an answer into) — the same "candidates must respect the hand's
*true*, always-public size" invariant `_physical_hand_candidates` and
`push_random_discard`'s physical branch already enforce elsewhere.
Missile Envy's picked card stays visible in the physical giver's hand
(`_reveal_in_hand`, not an immediate removal) until `missile_envy_use`
resolves it one or two decisions later — mirroring the non-physical path
exactly, and, like Grain Sales' revealed-but-undecided card, avoiding a
window where the card is tracked nowhere at all. `missile_envy_use`'s
`_file_card` call passes `already_removed_from_hand=True` since the taken
card was never genuinely in the *taker's* hand to begin with (same
pattern as Star Wars' `play_card_from_discard`) — needed for a physical
taker, since otherwise `_file_card` would misread an ordinary card
transfer as one of the taker's own cards leaving and strip an unrelated
placeholder. Cards where the deciding side owns the hand being asked
about (Blockade, Latin American Debt Crisis, Ask Not…, Nixon Plays the
China Card, Quagmire/Bear Trap discard, Held Card discard) and the
random-reveal cards (Grain Sales to Soviets, Five Year Plan, Terrorism)
are wired too. The one place the must-play-a-scoring-card rule *is*
enforced for a physical hand: a trapped side's Ops-2+-less round
(`_push_trap_step`'s fallback) offers any scoring-card candidates as a
genuine `QUAGMIRE_DISCARD` decision (`context["forced_scoring"]`) instead
of auto-resolving one the way the non-physical path does — auto-filing
would risk firing a `hidden_pool` card that isn't actually in this hand,
since the pool is a superset, not a location.

### Game-level logging

Separate from any LLM player's own reasoning log
(`bots/llm/conversation_log.py`, that player's private conversation
state), `runner.play_game(engine, players, log_path=...)` can record the
game itself as it's played, via `engine.replay.GameLogWriter`. It writes
a lean `{seed, new_game, include_optional, events, actions, winner}`
replay log — the same `new_game`/`actions` shape the "Deterministic
replay logs" testing strategy reads (`run_replay`), but without a
`checkpoints` section: that's golden-fixture furniture for pinning a
byte-for-byte `engine.serialize()` snapshot, which a live game isn't
being checked against, and `seed + actions` alone is already sufficient
to reproduce it exactly. Each `actions` entry is `encode_event`'s output,
not a bare `{kind, payload}` — actor, and (when it targets a country)
that country's resulting influence/control plus DEFCON/VP/turn/round,
the same fields `engine.human._format_event` shows a human player between
prompts — so the file reads as a play-by-play, not raw internal state.
`replay.py` is now both the reader (golden fixtures under
`tests/replays/`, via `run_replay`/`run_with_checkpoints`) and the writer
(live games, via `GameLogWriter`) of one format, not two modules
maintaining it separately. The file is atomically rewritten after every
step (same tempfile+`os.replace`, warn-not-raise pattern as
`conversation_log.save`), so a crash mid-game still leaves a replayable,
if truncated, log. `src/main.py` enables this by default
(`./logs/{timestamp}_game.json`; `--game-log-path` to override,
`--no-game-log` to disable), independent of whether either seat is an
LLM. This resolves the open question the LLM-tier roadmap note below used
to defer ("do the model's reasoning turns count as 'moves' in a replay
log, or stay external to it"): they stay external — the game log is the
engine-level action record, the LLM conversation log is a separate,
player-private artifact, and the two are never merged.

### Roadmap

Four tiers, in the order they're worth building — each one a strictly
bigger investment than the last, and each fully usable on its own once
built:

1. **Trivial baselines** (done): `FirstLegalPlayer` (deterministic, always
   the first legal option) and `RandomPlayer` (uniform over legal options,
   using its own seeded RNG — never the engine's, so a bot's choices never
   perturb or depend on the engine's own dice sequence, keeping replay logs
   reproducible regardless of which bots produced them). These exist mainly
   as a floor to measure every later bot against.
2. **Greedy / rule-based** (current — `bots/greedy.py`): observe the
   state, score every legal action of the *current* decision with
   hand-crafted heuristics, take the top score. No lookahead, no search, no
   opponent modeling — see "Greedy bot design" below.
3. **LLM reasoning layer** (future): craft a prompt carrying the
   `Observation`, the `Event` history (or a summarized form of it), and the
   model's own prior reasoning for this game, and let the model pick an
   action each decision. The natural-language reasoning trace is itself
   useful output (an explainable "why"), unlike Greedy or RL. Implementing
   this is "only" prompt engineering plus response parsing into a legal
   `Action` — it needs nothing new from the engine, since `Player` already
   receives everything an LLM prompt would need and returns everything
   `step()` needs to advance. The one new plumbing question this tier
   raised — do the model's reasoning turns count as "moves" in a replay
   log, or stay external to it — is answered in "Game-level logging"
   above: they stay external.
4. **Self-play reinforcement learning** (future, most promising long-term,
   most expensive to build): train a model by having it play itself
   repeatedly via `play_game`, using `Engine.winner` as the terminal reward.
   The most future-relevant reason `GreedyPlayer` is built as weighted
   features over `board_value()` rather than an if/elif cascade: a linear
   (or larger) model over the same feature set, with *learned* instead of
   hand-set weights, is a structurally compatible next step — swap
   `GreedyWeights` for trained parameters, or replace `board_value()` with
   a learned value function outright, without redesigning how a `Player`
   plugs into the engine. `Engine.serialize()`/`deserialize()` (mandate #5)
   are what make self-play cheap: cloning state for search/training doesn't
   need a bespoke copy path.

### Greedy bot design: the decision space, and how it's scored

The hard part of a Twilight Struggle bot is not "evaluate a board" — it's
that a turn is never one decision. `pending_decision.kind` (see
`DecisionKind`) ranges over ~20 shapes: place one Influence point, pick a
Coup or Realignment target, choose Influence vs. Coup vs. Realignment for
this Ops spend, choose which card to headline or play this round, choose
Ops vs. Event vs. Space Race for a played card, and (once M3's event layer
is on) another ~13 event-specific shapes (WAR_TARGET, EVENT_CHOICE,
EVENT_INFLUENCE, EVENT_OPS_ORDER, QUAGMIRE_DISCARD, HELD_CARD_DISCARD,
EVENT_RESUME, ...). Mandate #2 (atomic actions) is exactly what makes this
tractable for a greedy bot: every one of those decisions offers **tens** of
options, never thousands, so "score every legal option, take the best" is
cheap even without any pruning.

`GreedyPlayer` (`bots/greedy.py`) handles this with one scorer function
per `DecisionKind`, dispatched from a `_SCORERS` table, all funneling
through a single static evaluator:

```python
def board_value(weights: GreedyWeights, board: Board, side: Side) -> float:
    """Regional Presence/Domination/Control tiers, plus a flat bonus per
    country Controlled (extra for Battlegrounds). Higher is better for `side`."""
```

- **Influence placement**: score = the `board_value` swing from adding
  that one point (a real, cheap simulation on a scratch `Board` — not a
  multi-turn lookahead, just "what does this single atomic action change
  right now").
- **Coup / Realignment targets**: the outcome is a die roll, so the score
  is the *expectation* (average roll = 3.5) of the same `board_value`
  swing, not a real simulated outcome — realignment's dice cancel neatly in
  expectation (`own_bonus - opp_bonus`), since both sides roll.
- **Ops type** (Influence vs. Coup vs. Realignment): reuses the same
  per-target scorers over a proxy target list built from public board data
  (`Board.is_reachable`, `Board.influence_cost`, the `COUP_MIN_DEFCON`
  table) — not a duplicate of the engine's exact legality (NATO-style locks
  aren't replicated here), since a wrong guess here only costs a slightly
  worse **choice**, never an illegal `Action` (the engine's real
  `legal_actions()` is always what's actually offered downstream).
- **DEFCON safety** (CLAUDE.md's worked example, priority #1): any Coup
  attempt drops DEFCON by 1 for the *acting* side (Nuclear Subs excepted);
  if DEFCON is already 2, that is the acting side's own loss. This is
  checked at the OPS_TYPE decision (refusing "coup" outright, so the
  suicidal choice is never made in the first place) and again defensively
  at COUP_TARGET (in case OPS_TYPE's cheaper proxy legality missed a lock
  the real engine enforces) — `defcon_self_kill_penalty` in `GreedyWeights`
  is orders of magnitude above every other weight specifically so this
  never gets outweighed by board value.
- **Which card, and how to spend it** (headline pick, action-round card
  pick, Ops vs. Event vs. Space Race mode): a card not worth its Ops value
  right now is worth more sent to the Space Race track instead (its
  expected VP, computed from `SPACE_RACE_BOXES`' roll odds, against the
  Ops-point value forfeited) — the concrete form of "send bad cards to the
  Space Race." A scoring card's headline/play value is its `score_region()`
  net VP, signed favorably or unfavorably for the acting side.

**Known limitation, by design** (approved scope for v1 — see the milestone
note in `bots/greedy.py`'s module docstring): only the 7 core M1/M2
decision kinds get real heuristics (`PLACE_INFLUENCE`, `COUP_TARGET`,
`REALIGNMENT_TARGET`, `OPS_TYPE`, `HEADLINE_PLAY`, `ACTION_ROUND_PLAY`,
`PLAY_MODE`). Every M3 event-specific decision kind falls back to the first
legal option — the same incremental, card-by-card growth pattern M3 itself
used; extend `_SCORERS` as each one earns a heuristic worth writing, rather
than guessing at all ~13 up front. `tests/test_greedy.py` covers the DEFCON
safety rule, the fallback behavior, and a win-rate sanity check
(`GreedyPlayer` vs. `RandomPlayer` over many seeds, both seat assignments) —
a regression net for "the heuristics still actually help," not a claim of
strategic strength.

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
board mechanics and each M3 card individually. `tests/test_engine_m2.py`
pins every `Engine.new_game(...)` call to `events=False` explicitly (the
module docstring's "no events fire" is a claim this file must keep being
true of, not an assumption that ambient defaults happen to satisfy);
the events-on equivalent of its full-game invariant test lives in
`tests/test_events.py`.

### Test-writing policy
Before writing a new test helper or fixture, check `tests/conftest.py`
first. A near-duplicate invariant checker or setup helper copy-pasted
across test files is a bug waiting to happen, not just clutter: it was
exactly how a real defect stayed hidden here once — `test_engine_m2.py`
kept its own copy of the "where can a card be" invariant checker, which
predated the M3 headline-resolution-order mechanism (`_headline_pending`)
and was never taught about it, while the copy in `test_events.py` was
fixed. The stale copy then flagged perfectly valid games as broken.
Lesson applied: that checker now lives once, in `conftest.py`.

Going forward:
- A test must assert on real state (board/VP/DEFCON/decision-stack
  contents/serialized output) — never "the call didn't raise" or "the
  result is not None" as its only assertion.
- Test volume should track the mandate that motivates it (e.g. "one
  test per M3 card," CLAUDE.md's own testing-strategy requirement) —
  don't add tests for hypothetical future behavior, and don't multiply
  near-identical tests for closely related branches of one mechanic
  when a single parametrized test would cover them.
- When a new engine mechanic introduces a new place a piece of state
  (a card, a flag) can transiently live, update the shared invariant
  helper in `conftest.py` once, rather than letting each test file's
  own copy drift out of sync with it.

## Tooling and conventions

- **Python**: 3.12+.
- **Environment**: conda (`environment.yml` at repo root).
- **Tests**: `pytest`, plus `hypothesis` for property-based tests.
- **License**: MIT.
- **Language**: all code, comments, docstrings, and commit messages in
  English.
- **Layout**: `src/struggler/` package (src-layout to avoid accidental
  implicit imports of the working directory during tests), split by
  concern: `engine/` is the rules engine itself (state, board, cards,
  events, replay, and the `Player`/`HumanPlayer` contract that bots plug
  into), `bots/` holds the automated `Player` implementations (wired up by
  `src/main.py`'s `build_player`, see "Building players" above), and
  `data/` (inside the package) holds the game's JSON facts (`cards.json`,
  `countries.json`, `rules.json`). Tests live under `tests/`, golden
  replay logs under `tests/replays/`.
