"""CLI entry point: play a game with any mix of human/bot seats.

Examples:
    python src/main.py                                    # human vs human
    python src/main.py --us greedy --ussr random --seed 1  # bot vs bot
    python src/main.py --ussr greedy                       # human (US) vs bot (USSR)

Bots are looked up by name in `struggler.players.PLAYER_REGISTRY`; adding a
new one there (see registry.py) makes it available here for free.
"""

from __future__ import annotations

import argparse

from struggler.engine import Engine
from struggler.players import PLAYER_REGISTRY, Player, build_player
from struggler.runner import play_game
from struggler.types import Side


def main() -> None:
    parser = argparse.ArgumentParser(description="Play a Twilight Struggle game.")
    parser.add_argument("--us", choices=sorted(PLAYER_REGISTRY), default="human")
    parser.add_argument("--ussr", choices=sorted(PLAYER_REGISTRY), default="human")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--events", action="store_true", help="enable M3 card events")
    args = parser.parse_args()

    engine = Engine.new_game(seed=args.seed, events=args.events)
    players: dict[Side, Player] = {
        Side.US: build_player(args.us, seed=args.seed + 1),
        Side.USSR: build_player(args.ussr, seed=args.seed + 2),
    }

    winner = play_game(engine, players)
    print(f"\nGanador: {winner}")


if __name__ == "__main__":
    main()
