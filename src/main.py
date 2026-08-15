"""CLI entry point: play a game with any mix of human/bot seats.

Examples:
    python src/main.py                                    # human vs human
    python src/main.py --us greedy --ussr random --seed 1  # bot vs bot
    python src/main.py --ussr greedy                       # human (US) vs bot (USSR)
    python src/main.py --ussr llm

Bots register themselves with `struggler.engine.player_registry` when their
module is imported (see player_registry.py); importing them below is what
makes them available here, and is the only place that needs editing to
offer a new one on this CLI.
"""

from __future__ import annotations

import argparse

import struggler.bots.greedy  # noqa: F401  (registers "greedy")
import struggler.bots.llm.player  # noqa: F401  (registers "llm")
import struggler.bots.naive  # noqa: F401  (registers "random", "first")
from struggler.engine import Engine, Side
from struggler.engine.player import Player
from struggler.engine.player_registry import available, build_player
from struggler.runner import play_game


def main() -> None:
    parser = argparse.ArgumentParser(description="Play a Twilight Struggle game.")
    parser.add_argument("--us", choices=available(), default="human")
    parser.add_argument("--ussr", choices=available(), default="human")
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    engine = Engine.new_game(seed=args.seed)
    players: dict[Side, Player] = {
        Side.US: build_player(args.us, seed=args.seed + 1),
        Side.USSR: build_player(args.ussr, seed=args.seed + 2),
    }

    winner = play_game(engine, players)
    print(f"\nGanador: {winner}")


if __name__ == "__main__":
    main()
