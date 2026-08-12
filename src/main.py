"""CLI entry point: play a game with any mix of human/bot seats.

Examples:
    python src/main.py                                   # human vs human
    python src/main.py --us random --ussr first --seed 1  # bot vs bot
    python src/main.py --ussr random                      # human (US) vs bot (USSR)
"""

from __future__ import annotations

import argparse

from struggler.engine import Engine
from struggler.players import FirstLegalPlayer, HumanPlayer, Player, RandomPlayer
from struggler.runner import play_game
from struggler.types import Side


def build_player(kind: str, seed: int) -> Player:
    if kind == "human":
        return HumanPlayer()
    if kind == "random":
        return RandomPlayer(seed=seed)
    if kind == "first":
        return FirstLegalPlayer()
    raise ValueError(f"unknown player kind: {kind!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Play a Twilight Struggle game.")
    parser.add_argument("--us", choices=["human", "random", "first"], default="human")
    parser.add_argument("--ussr", choices=["human", "random", "first"], default="human")
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
