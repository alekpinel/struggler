# Testing strategy

Testing is first-class, not an afterthought — this is a rules engine;
correctness *is* the product.

## Deterministic replay logs (primary strategy)

A replay log is a JSON file:

```json
{
  "seed": 12345,
  "actions": [ {"kind": "place_influence", "payload": {"country": "Poland"}}, ... ],
  "checkpoints": [ {"after_step": 10, "state": { ...full serialize() dict... } } ]
}
```

- Replaying `seed` + `actions` through a fresh `Engine` must reproduce every
  checkpoint's state exactly (`==` on the deserialized dict, not a hash —
  exact equality gives a diffable failure).
- Golden replay logs are checked into `tests/replays/` and grow with the
  project: each new mechanic (a coup, a realignment, each card) gets at
  least one golden replay exercising it.
- Because chance is a logged `CHANCE` decision (mandate #3), replay logs are
  fully deterministic even through dice-driven mechanics — there is no
  separate "RNG trace" to keep in sync.

The current goldens are `influence_basic.json`, `full_game_ops_only.json`,
`events.json` and `physical_basic.json`.

## Property-based invariant tests

Using `hypothesis` to generate random *legal* action sequences (always drawn
from `legal_actions()`, never arbitrary payloads) and assert invariants that
must hold after every `step()`:

- DEFCON always in `[1, 5]`; game ends immediately at DEFCON 1.
- Influence values are never negative.
- `serialize()` → `deserialize()` → `serialize()` is idempotent.
- `observe(player)` never contains a key/value that reveals hidden
  information (checked structurally, not just spot-checked fields).
- `legal_actions()` is never empty while `pending_decision` is not `None`
  (the engine must never deadlock).

## Unit tests

Standard per-mechanic tests (e.g. "coup in a country at DEFCON 2 with an
effective defcon-adjustment card active behaves like X") for board
mechanics, and one per card.

`tests/test_engine_ops_only.py` pins every `Engine.new_game(...)` call to
`events=False` explicitly — the module docstring's "no events fire" is a
claim that file must keep being true of, not an assumption that ambient
defaults happen to satisfy. The events-on equivalent of its full-game
invariant test lives in `tests/test_events.py`.

## Test-writing policy

Before writing a new test helper or fixture, check `tests/conftest.py`
first. A near-duplicate invariant checker or setup helper copy-pasted across
test files is a bug waiting to happen, not just clutter: it is exactly how a
real defect stayed hidden here once. The Ops-only test module kept its own
copy of the "where can a card be" invariant checker, which predated the
headline-resolution-order mechanism (`_headline_pending`) and was never
taught about it, while the copy in `test_events.py` was fixed. The stale
copy then flagged perfectly valid games as broken. That checker now lives
once, in `conftest.py`.

Going forward:

- A test must assert on real state (board/VP/DEFCON/decision-stack
  contents/serialized output) — never "the call didn't raise" or "the result
  is not None" as its only assertion.
- Test volume should track the mandate that motivates it (e.g. one test per
  card) — don't add tests for hypothetical future behavior, and don't
  multiply near-identical tests for closely related branches of one mechanic
  when a single parametrized test would cover them.
- When a new engine mechanic introduces a new place a piece of state (a
  card, a flag) can transiently live, update the shared invariant helper in
  `conftest.py` once, rather than letting each test file's own copy drift
  out of sync with it.
