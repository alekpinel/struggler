# struggler

An API-first, deterministic rules engine for *Twilight Struggle* (GMT
Games, 2005), built so AI agents can be trained and evaluated against it.

The engine is a state machine driven entirely through a narrow public API.
It never assumes whose turn it is, never resolves a die roll silently, and
never shows a player information they could not see at the table. Those
constraints are what make it usable as a training and evaluation
environment rather than just a game implementation.

```python
from struggler.engine import Engine

engine = Engine.new_game(seed=12345)

while not engine.is_terminal:
    decision = engine.pending_decision
    # decision.actor is Side.US, Side.USSR, or Side.CHANCE. A CHANCE
    # decision carries exactly one pre-rolled option, so there is nothing
    # to decide; struggler.runner.play_game resolves those for you.
    action = pick(decision.options)          # your agent, or a bundled bot
    engine.step(action)

print(engine.winner)
```

## Install

Python 3.12+.

```sh
pip install -e ".[test]"     # engine + test dependencies
pip install -e ".[llm]"      # optional: the LLM-backed bot
```

A conda environment is also provided: `conda env create -f environment.yml`.

## Play a game

```sh
python src/main.py                                     # human vs human
python src/main.py --us greedy --ussr random --seed 1   # bot vs bot
python src/main.py --ussr greedy                        # human (US) vs bot (USSR)
python src/main.py --physical us --ussr greedy          # bot vs a real physical board
```

Every seat — human, scripted bot, LLM, or a human playing the physical
board with the engine as referee — plugs in through the same `Player`
interface, so all of those are one code path.

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
from its log — see [docs/BOTS.md](docs/BOTS.md) for the resumption
contract, including keeping that memory in sync if you trim the game log.

## What makes it different

Five mandates the implementation is held to, described in full in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md):

1. **A pending-decision stack, not "one turn = one action."** Card events
   interrupt; resolving a decision can push sub-decisions that must resolve
   before control returns.
2. **An atomic action space.** Spending 4 Ops is four single-point
   decisions, not one choice among thousands of combinations. Tens of legal
   options per decision, never thousands.
3. **A seeded, injectable RNG.** Same seed plus same actions gives
   byte-identical state on any machine. Dice are exposed as explicit
   `CHANCE` decisions, so a replay log is a complete record of a game.
4. **A per-player observation function.** `observe(player)` is the only
   sanctioned view, and hidden information is *absent* from it rather than
   masked.
5. **Flat, serializable state.** `serialize()` returns JSON primitives, so
   replay logs are diffable and greppable and cloning state for search or
   training is cheap.

## Status

All 110 cards are implemented, including every non-scoring card's event.
Remaining work is rules fidelity rather than coverage — see
[docs/LIMITATIONS.md](docs/LIMITATIONS.md) for what is deliberately not
modeled.

## Documentation

| Document | Contents |
| --- | --- |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | The five mandates, the public API, core types |
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
