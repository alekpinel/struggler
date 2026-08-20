# struggler

Welcome to struggler! An engine for *Twilight Struggle* (GMT Games, 2005), built so AI agents can be trained and evaluated against it.

This repo contains the engine of the whole game, that can be used to test different bots against each other, a human player or even an external player.

It also contains an LLM bot: it reads the rules and a strategy guideline as its knowledge base, gets the current board state at every decision, and drafts a turn plan it then plays every turn (and decides to follow the plan or diverging from it).

To understand the full story, read the article [Giving AI the atom bomb buttons](https://siestainsolaris.substack.com/p/giving-ai-the-atom-bomb-buttons)

## Install

Python 3.12+.

The easiest way is to use a [conda environment](https://www.anaconda.com/docs/getting-started/miniconda/install/windows-gui-install).

```sh
conda env create -f environment.yml
conda activate struggler
pip install -e ".[test]"
pip install -e ".[llm]"  # optional if you plan to use the llm
```

### Configure an LLM bot

To use the llm bot, you need to set up your api keys. This implementation supports anthropic and openai.

```sh
export ANTHROPIC_API_KEY=...   # for provider=anthropic
export OPENAI_API_KEY=...      # for provider=openai (the default)
```

Provider and model are picked via environment variables, each overridable per run:

| Variable | Default | Purpose |
| --- | --- | --- |
| `STRUGGLER_LLM_PROVIDER` | `openai` | `anthropic` or `openai` — used for both the per-decision client and the once-per-turn planning client |
| `STRUGGLER_LLM_MODEL` | provider's built-in default | model for in-decision calls |
| `STRUGGLER_LLM_PLAN_MODEL` | provider's built-in default | model for the turn-planning call (same provider as above) |

## Play a game

You can just use the provided main to run any game.

```sh
python src/main.py                                  # human vs human
python src/main.py --us greedy --ussr greedy --seed 1  # bot vs bot
python src/main.py --ussr llm                       # human (US) vs llm bot (USSR)
python src/main.py --physical us --ussr llm         # bot vs a real physical board
```
The options for the players are:
- human
- first
- random
- greedy
- llm

But you can create your own implementation using the engine like this:

```python
from struggler.engine import Engine

engine = Engine.new_game(seed=12345)

while not engine.is_terminal:
    decision = engine.pending_decision
    action = pick(decision.options) # your agent has to pick the decision
    engine.step(action)

print(engine.winner)
```

Every game defaults to a saved replay log under `./logs/`
(`--game-log-path` to pick a location, `--no-game-log` to disable). Resume
one later with `--resume-game-log`, which rebuilds the game from that file
and keeps appending to it — useful as-is, or after hand-trimming the file's
`actions` to undo a bad play before continuing:

```sh
python src/main.py --resume-game-log logs/2026-08-18_10-58_game.json \
  --ussr llm --ussr-log-path logs/2026-08-18_10-58_ussr.json --resume
```

`--resume` additionally reloads an LLM player's own conversation memory
from its log see [docs/BOTS.md](docs/BOTS.md) for the resumption
contract, including keeping that memory in sync if you trim the game log.

## Add a new bot

Every player, human or bot, uses the same `Player` interface:
implement `choose_action(observation, history) -> Action`, returning one
action drawn from `observation.pending_decision.options`, then add one
branch to `build_player` in [src/main.py](src/main.py) mapping a new kind
name (for `--us`/`--ussr`) to it.  See [docs/BOTS.md](docs/BOTS.md) for the full `Player` contract and how the existing bots (`first`, `random`, `greedy`,
`llm`) are built.

## Status

All 110 cards are implemented, including every non-scoring card's event.
See [docs/LIMITATIONS.md](docs/LIMITATIONS.md) for any known limitations.

## Documentation

| Document | Contents |
| --- | --- |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | The public API, core types |
| [docs/CARDS.md](docs/CARDS.md) | Card data policy, the event layer, per-card coverage |
| [docs/BOTS.md](docs/BOTS.md) | The `Player` interface, physical mode, bot roadmap |
| [docs/TESTING.md](docs/TESTING.md) | Replay logs, property tests, test-writing policy |
| [docs/LIMITATIONS.md](docs/LIMITATIONS.md) | What the engine does not model |

## Tests

```sh
pytest
```

## License

Released under the [MIT License](LICENSE).

## Disclaimer

This is an unofficial, fan-made project. It is **not affiliated with,
endorsed by, or sponsored by GMT Games** or the designers of *Twilight
Struggle*. *Twilight Struggle* is a trademark of GMT Games, LLC.

No copyrighted material from the published game is redistributed here. The
data files under `src/struggler/data/` record only factual attributes of the
physical game — card names and numbers, Operations values, allegiance, deck,
country adjacency and Battleground status — which this project re-entered
independently from the published game components. The one free-text field,
`event_summary`, is a short hand-written description of what this engine's
own code does for that card, not a reproduction of the printed card. No card
event text, artwork, rulebook prose, or other copyrightable expression is
included anywhere in this repository.

Playing this engine is not a substitute for owning the game. If you enjoy
*Twilight Struggle*, buy a copy from [GMT Games](https://www.gmtgames.com/).

## Author

Built by [Alejandro Pinel Martínez](https://github.com/alekpinel)
