# Bot framework

The engine's job is to be a fair arbiter, not to know who's playing. Every
seat — human or bot — plugs in through the same interface, so human-vs-human,
human-vs-bot, and bot-vs-bot are one code path, and adding a new bot never
touches the engine.

## The `Player` interface

`struggler.engine.player.Player` is a structural `Protocol`, not a base
class: any object with a matching `choose_action` method is a `Player`, no
inheritance required — consistent with the rest of the project's
API-surface philosophy: the contract is a shape, not a class hierarchy.

```python
class Player(Protocol):
    def choose_action(self, observation: Observation, history: Sequence[Event]) -> Action:
        """Pick one action from `observation.pending_decision.options`."""
```

- A player only ever sees `observe(side)` (mandate #4) and returns one
  `Action` drawn verbatim from `pending_decision.options` (mandate #2) —
  the same constraints a human at the console has.
- `history` is every resolved `(Decision, Action)` pair so far, oldest
  first (opponent moves and CHANCE rolls included), as a
  `engine.player.Event` list — one shared, ever-growing list, not a
  per-player delta since that seat was last consulted. The one ordering
  exception is the headline: both `HEADLINE_PLAY` events are buffered by
  `engine.replay.HistoryBuilder` (used internally by `runner.play_game`)
  and appended together once the second pick is locked in, so the second
  picker's `history` can't leak the first pick. Bots are free to ignore it;
  it exists so a player *can* condition on what just happened without
  re-deriving it from `Observation` alone.
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

## Building players

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

## Physical mode

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

Every hand-touching event is wired for a physical hidden hand. Three need
the *deciding* side's actor overridden to `Side.CHANCE`, so the operator —
not a bot that cannot see the target hand — answers: Aldrich Ames Remix and
Missile Envy source candidates from `_physical_hand_candidates` and route
the choice the same way `DEAL_CARD` does, while The Cambridge Five asks one
per-scoring-card yes/no `EVENT_CHOICE` query at a time
(`_push_cambridge_five_query`). All three respect one invariant that
`_physical_hand_candidates` and `push_random_discard`'s physical branch
enforce elsewhere too: **candidates must respect the hand's true,
always-public size**, so Cambridge Five stops the moment the last open
`HIDDEN_CARD` slot is filled — asking further would have nowhere left to
reveal an answer into.

Two placement details matter for a physical hand. Missile Envy's picked
card stays visible in the giver's hand (`_reveal_in_hand`, not an immediate
removal) until `missile_envy_use` resolves it one or two decisions later —
mirroring the non-physical path, and, like Grain Sales' revealed-but-
undecided card, avoiding a window where the card is tracked nowhere at all.
And `missile_envy_use`'s `_file_card` call passes
`already_removed_from_hand=True`, since the taken card was never genuinely
in the *taker's* hand (the same pattern as Star Wars'
`play_card_from_discard`); without it, `_file_card` would misread an
ordinary card transfer as one of the taker's own cards leaving and strip an
unrelated placeholder.

Cards where the deciding side owns the hand being asked about (Blockade,
Latin American Debt Crisis, Ask Not…, Nixon Plays the China Card,
Quagmire/Bear Trap discard, Held Card discard) and the random-reveal cards
(Grain Sales to Soviets, Five Year Plan, Terrorism) are wired too.

What physical mode cannot enforce or model is listed in
[LIMITATIONS.md](LIMITATIONS.md).

## Game-level logging

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

**Resuming a live game** (`--resume-game-log <path>`, `src/main.py`) is the
other direction: `engine.replay.replay_history(log)` replays a game log's
`actions` (same mechanism as `run_replay`) and, alongside it, rebuilds the
`Player`-facing `history` via `HistoryBuilder`, so a fresh `Player`
consulted from that point on sees the same `history` it would have live —
in particular satisfying a resumed `LLMPlayer`'s contract that `history` be
at least as long as its restored `last_seen` (`bots/llm/conversation_log.py`).
Hand-trimming a log's `actions` before resuming (e.g. to undo a bad play)
is the intended way to correct a game already in progress; an `LLMPlayer`
resumed with `--resume` alongside it should have its own conversation log
trimmed in step (drop the trailing message/journal entries for the undone
decisions, and roll `last_seen` back to match), or its memory and the
actual game state will disagree. `play_game`'s `initial_actions` parameter
lets the on-disk log continue accumulating at the same path instead of
restarting.

## Roadmap

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
3. **LLM reasoning layer** (built — `bots/llm/`): a prompt carrying the
   `Observation`, the `Event` history (or a summarized form of it), and the
   model's own prior reasoning for this game lets the model pick an
   action each decision. The natural-language reasoning trace is itself
   useful output (an explainable "why"), unlike Greedy or RL. It needed
   nothing new from the engine, since `Player` already receives everything
   an LLM prompt would need and returns everything
   `step()` needs to advance — the tier is prompt engineering
   (`prompt.py`, `rules_primer.py`) plus response parsing into a legal
   `Action` (`schema.py`), over a provider-agnostic `LLMClient` with
   Anthropic and OpenAI adapters. The one new plumbing question this tier
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

## Greedy bot design: the decision space, and how it's scored

The hard part of a Twilight Struggle bot is not "evaluate a board" — it's
that a turn is never one decision. `pending_decision.kind` (see
`DecisionKind`) ranges over ~20 shapes: place one Influence point, pick a
Coup or Realignment target, choose Influence vs. Coup vs. Realignment for
this Ops spend, choose which card to headline or play this round, choose
Ops vs. Event vs. Space Race for a played card, and (with the event layer
on) another ~13 event-specific shapes (WAR_TARGET, EVENT_CHOICE,
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
  (`Board.is_reachable`, `Board.influence_cost`, the
  `RULES["coup_min_defcon"]` table) — not a duplicate of the engine's exact
  legality (NATO-style locks aren't replicated here), since a wrong guess
  here only costs a slightly
  worse **choice**, never an illegal `Action` (the engine's real
  `legal_actions()` is always what's actually offered downstream).
- **DEFCON safety** (priority #1): any Coup
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

**Known limitation, by design** (approved scope for v1 — see the note in
`bots/greedy.py`'s module docstring): only the 7 core board decision kinds
get real heuristics (`PLACE_INFLUENCE`, `COUP_TARGET`,
`REALIGNMENT_TARGET`, `OPS_TYPE`, `HEADLINE_PLAY`, `ACTION_ROUND_PLAY`,
`PLAY_MODE`). Every event-specific decision kind falls back to the first
legal option — the same incremental, card-by-card growth pattern the event
layer itself used; extend `_SCORERS` as each one earns a heuristic worth
writing, rather than guessing at all ~13 up front. `tests/test_greedy.py`
covers the DEFCON safety rule, the fallback behavior, and a win-rate sanity
check (`GreedyPlayer` vs. `RandomPlayer` over many seeds, both seat
assignments) — a regression net for "the heuristics still actually help,"
not a claim of strategic strength.

