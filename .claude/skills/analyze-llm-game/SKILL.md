---
name: analyze-llm-game
description: Review a Twilight Struggle game the LLM bot played -- whether its turn plans made sense, whether decisions actually followed them, and what any fallback/error entries mean. Use when the user asks to analyze, review, or critique a game in logs/, or asks about the LLM bot's turn plans, decision quality, or errors in a specific game log.
---

# Analyzing an LLM-played game

A played game leaves up to three files in `logs/`, sharing one timestamp
prefix (`src/main.py`'s `build_player`/`play_game` wiring):

- `<prefix>_game.json` -- the engine-level replay log (`GameLogWriter`):
  every action, both sides, DEFCON/VP after each one. Authoritative for
  "what actually happened."
- `<prefix>_us.json` / `<prefix>_ussr.json` -- one `ConversationSnapshot`
  per LLM-controlled seat (`bots/llm/conversation_log.py`). Has
  `turn_plan_history` (the intent for every turn) and `journal` (one entry
  per LLM call: a decision's `justification`, or a `fallback_used`/
  `fallback_reason` when every retry failed). A seat played by a
  non-LLM bot (`first`/`random`/`greedy`) has no such file.

These files are large (several hundred KB, often one line) -- don't `Read`
them directly. `scripts/analyze_llm_game.py` does the extraction:

```
python scripts/analyze_llm_game.py logs/<prefix>              # overview
python scripts/analyze_llm_game.py logs/<prefix> --errors     # every fallback entry
python scripts/analyze_llm_game.py logs/<prefix> --turn N [--side us|ussr]   # full detail for one turn
```

If the user doesn't name a prefix, use the most recently modified
`*_game.json` in `logs/`; if several games look equally current, ask which
one.

## Process

1. **Run the overview first.** It gives: winner, the VP/DEFCON/card-plays
   timeline per turn, one line per turn plan objective per LLM side, and a
   count of fallback entries. `vp` is signed: negative favors USSR
   (auto-win at -20), positive favors US (auto-win at +20) -- this is also
   printed in the overview header, but restate it correctly rather than
   guessing which side a VP number favors.

2. **If `winner` is `null` or there are fallback entries, run `--errors`
   before anything else.** Read the actual `fallback_reason` text -- don't
   assume it means what the user guesses it means. In particular:
   - An HTTP 429 mentioning "tokens per min (TPM)" or "rate_limit" is a
     **request-size rate limit** (input + output tokens in that one
     request), not the `max_tokens` *output* cap -- raising `max_tokens`
     does not fix this and can make it worse. The actual cause is almost
     always the conversation growing too large for a provider's per-minute
     budget; see `docs/LIMITATIONS.md`'s `LLMPlayer` bullet and
     `player.py`'s module docstring for what is and isn't bounded (event
     history and the model's own past responses still grow without limit,
     even though the board report/hand dossier/cards-in-play no longer
     re-accumulate turn over turn).
   - Check whether the game log's last action lines up with the failing
     decision (the overview's "ends mid-turn ..." note flags this) --
     that tells you whether the run actually stalled there versus recovered
     on a random fallback action and kept going.
   - Report the root cause, not just the symptom, and don't propose a fix
     inline unless asked -- flag it and let the user decide whether to act
     on it.

3. **Walk each turn a plan exists for**, per LLM side, with
   `--turn N --side <side>`. For each turn, judge:
   - Does `assessment` read the board correctly? Spot-check one or two of
     its claims against that turn's game-log action slice (also printed by
     `--turn`).
   - Do `objective`/`card_plan`/`military_ops_plan`/`defend` follow from
     that assessment, and are they internally consistent (e.g. Military
     Operations plan actually meets that turn's DEFCON requirement)?
   - For each decision's justification: does it say it's following the
     plan, or explain a departure with a real board-state reason (per
     `docs/BOTS.md`'s turn-plan section, that's exactly what
     `justification` is supposed to do)? A justification that ignores the
     plan without explanation, or whose stated reason doesn't hold up
     against what the game log shows, is worth flagging.
   - Did the things the plan said to `defend` actually survive that turn,
     per the game log?

4. **Look across turns, not just within one.** The per-turn view hides
   patterns like: a region flagged weak in one turn's `assessment` but
   never invested in over several turns, until an opponent Scoring card
   cashes in on exactly that neglect; a card whose `card_plan` entry never
   actually gets played that turn; a DEFCON-safety rule honored in one
   turn's decisions but violated in a later one despite being named a
   priority earlier.

5. **Write the verdict in the language the user asked in.** Structure: a
   short per-turn take (plan sound? followed well? notable departures?),
   then one overall verdict, then a separate section for any technical
   errors found (root cause, concrete evidence from the logs -- not
   speculation). Don't implement fixes as part of this review; surface them
   and let the user decide whether to act.

## Notes on the extraction script

- `turn_plan_history` only exists from snapshot version 3 on. For an older
  log, the script reconstructs each turn's plan from the `journal`
  entries of kind `"turn_plan"` (their `raw_responses[0]` holds the full
  plan JSON; the persisted `justification` on those entries is only the
  short objective) -- and numbers them by counting `turn_plan` entries in
  order, so a game played with `plan_turns=False` for part of it (no
  turn_plan entries at all, e.g. via `--no-turn-plan`) will not line up
  with real turn numbers. If a turn's plan looks obviously mismatched,
  check `journal` `kind` values directly before trusting the numbering.
- `--turn` prints every field of a decision's justification and the turn's
  full plan verbatim -- for a turn with many decisions this can still be
  long. Prefer reading the overview and `--errors` first, then drill into
  the 1-3 turns that actually need scrutiny rather than walking all of
  them at full detail by default.
