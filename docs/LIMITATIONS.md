# Known limitations

What this engine does not model, or models in a deliberately simplified
way.

## Rules fidelity

- **Space Race box 4** — the headline-reveal-order perk (requiring the
  opponent to select their Headline Event first) is not modeled. Boxes 6
  and 8 are (see [CARDS.md](CARDS.md)).
- **Shuttle Diplomacy** is filed to the discard pile when played, rather
  than kept "in front of you" until its delayed effect triggers. Only the
  effect flag matters mechanically. A card-manipulation event such as Star
  Wars could in principle retrieve it slightly earlier than the physical
  game allows, but the effect it would re-apply is idempotent, so this has
  no actual gameplay consequence.
- **Aldrich Ames Remix**'s "USA reveals their hand face-up until end of
  turn" is modeled as a momentary reveal — the decision options — rather
  than an ongoing visibility grant surfaced through `observe()`. Modeling
  it properly would add a new hidden/shared-visibility field to the public
  `Observation` API, which is a larger change than a card-logic fix.

## Data

- **`event_summary` can drift.** The field is a hand-maintained paraphrase
  of what `events.py` does for a card (see [CARDS.md](CARDS.md)); nothing
  automatically checks it against the code. It feeds the LLM bot's prompt,
  so drift degrades that bot's understanding rather than the engine's
  behavior.

## Physical mode

Physical mode makes one seat a real human playing the physical board, with
the engine as referee. Some things the engine simply cannot know.

- **The must-play-a-scoring-card rule is not enforced** for a hand the
  engine cannot see. Every not-yet-accounted-for card is offered at
  `ACTION_ROUND_PLAY`, and the physical player — who can see their own hand
  — is trusted to honor the rule, the same trust model any human player
  already gets for rules `HumanPlayer` does not independently re-verify.

  The one place it *is* enforced: a trapped side's Ops-2+-less round
  (`_push_trap_step`'s fallback) offers any scoring-card candidates as a
  genuine `QUAGMIRE_DISCARD` decision (`context["forced_scoring"]`) instead
  of auto-resolving one the way the non-physical path does. Auto-filing
  would risk firing a `hidden_pool` card that is not actually in that hand,
  since the pool is a superset, not a location.
- **Our Man in Tehran is a no-op.** It peeks at the *draw pile's* real
  contents, which physical mode makes unknown to the engine itself, not
  merely hidden from a player, so there is nothing to queue instead of
  `HIDDEN_CARD` placeholders.

Every other event is wired for a physical hidden hand, including the three
where the *deciding* side must inspect the *opponent's* hand — Aldrich Ames
Remix and Missile Envy (whose `choose_side` is overridden to `Side.CHANCE`
so the operator, not a bot that cannot see the target hand, answers), and
The Cambridge Five (a per-scoring-card yes/no query sequence via
`_push_cambridge_five_query` instead of a single choice, stopping the moment
the hand's last open `HIDDEN_CARD` slot is filled).

## Bots

- **`GreedyPlayer` scores only the 7 core decision kinds**
  (`PLACE_INFLUENCE`, `COUP_TARGET`, `REALIGNMENT_TARGET`, `OPS_TYPE`,
  `HEADLINE_PLAY`, `ACTION_ROUND_PLAY`, `PLAY_MODE`). Every event-specific
  decision kind falls back to the first legal option. This is approved
  scope, not an oversight: extend `_SCORERS` as each one earns a heuristic
  worth writing, rather than guessing at all ~13 up front.
- **`LLMPlayer` resends its whole conversation every call** and has no
  context-budget safety valve. What gets *persisted* per turn is trimmed to
  the event delta only (`prompt.build_history_entry`) -- the board report,
  hand dossier, and cards-in-play a call was answered against are a snapshot
  of that instant and are never resent from history, only ever recomputed
  fresh for the live call -- but the event history and the model's own past
  responses still grow without bound. A long enough game can still in theory
  approach the model's context limit or a provider's tokens-per-minute rate
  limit.
- **Resuming an `LLMPlayer` from its log does not restore `_rng`'s exact
  position.** Acceptable because `_rng` is only consulted on the fallback
  path (picking *a* legal action after total LLM failure); mandate #3's
  determinism guarantee is about the engine's own RNG, not a bot's internal
  fallback RNG.
- **The LLM plan-queue matcher checks legality, not optimality.** A step
  predicted past a CHANCE roll can be silently consumed if it happens to
  remain legal regardless of the roll's actual outcome. The system prompt
  asks the model to avoid this; the mechanism itself does not enforce it.
- **The OpenAI adapter's exact SDK call shape is unverified** against a
  live API. The Anthropic adapter's is current.
