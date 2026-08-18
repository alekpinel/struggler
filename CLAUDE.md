# Working on struggler

Project documentation lives in `docs/`. Read the relevant document before
changing the area it covers — they are the binding contract, not background
reading, and a change to what they specify should update them in the same
commit.

| Document | Read it before touching |
| --- | --- |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | The decision stack, the public API, `Engine`, core types |
| [docs/CARDS.md](docs/CARDS.md) | `events.py`, `cards.json`, anything card-related |
| [docs/BOTS.md](docs/BOTS.md) | `bots/`, the `Player` protocol, physical mode |
| [docs/TESTING.md](docs/TESTING.md) | Adding or changing any test |
| [docs/LIMITATIONS.md](docs/LIMITATIONS.md) | Before "fixing" something that may be a documented simplification |

The five architectural mandates in `docs/ARCHITECTURE.md` are
non-negotiable. Code referring to "mandate #3" means that list. An
implementation that violates one is wrong regardless of whether it passes
the tests.

## Conventions

- **Python**: 3.12+.
- **Tests**: `pytest`, plus `hypothesis` for property-based tests. Run the
  full suite before committing; it takes well under a minute.
- **Environment**: conda (`environment.yml`), or `pip install -e ".[test]"`.
- **License**: MIT.
- **Language**: all code, comments, docstrings, and commit messages in
  English.
- **Layout**: `src/struggler/` package (src-layout, to avoid accidental
  implicit imports of the working directory during tests), split by
  concern:
  - `engine/` — the rules engine itself: state, board, cards, events,
    replay, and the `Player`/`HumanPlayer` contract that bots plug into.
  - `bots/` — the automated `Player` implementations, wired up by
    `src/main.py`'s `build_player`.
  - `data/` — the game's JSON facts (`cards.json`, `countries.json`,
    `rules.json`).

  Tests live under `tests/`, golden replay logs under `tests/replays/`.

## Two things that have bitten this codebase before

- **Check `tests/conftest.py` before writing a test helper.** A
  near-duplicate invariant checker copy-pasted across test files once let a
  real defect hide for weeks. See `docs/TESTING.md`.
- **Don't re-derive placement legality from the live board mid-Ops-spend.**
  Rule 6.1.1 freezes reachability at the start of the action round; see the
  reachability section of `docs/ARCHITECTURE.md`.
