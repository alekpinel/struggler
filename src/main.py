"""CLI entry point: play a game with any mix of human/bot seats.

Examples:
    python src/main.py                                    # human vs human
    python src/main.py --us greedy --ussr random --seed 1  # bot vs bot
    python src/main.py --ussr greedy                       # human (US) vs bot (USSR)
    python src/main.py --ussr llm
"""

from __future__ import annotations

import argparse
import os

from struggler.bots.greedy import GreedyPlayer
from struggler.bots.llm.client import LLMClient
from struggler.bots.llm.player import LLMPlayer
from struggler.bots.naive import FirstLegalPlayer, RandomPlayer
from struggler.engine import Engine, Side
from struggler.engine.human import HumanPlayer
from struggler.engine.player import Player
from struggler.runner import play_game


def build_llm_client(provider: str = "openai", model: str = "gpt-5.6-luna"):
    if provider == "anthropic":
        from struggler.bots.llm.anthropic_client import AnthropicClient
        client: LLMClient = AnthropicClient(model=model)
    elif provider == "openai":
        from struggler.bots.llm.openai_client import OpenAIClient
        client = OpenAIClient(model=model)
    else:
        raise ValueError(
            f"unknown STRUGGLER_LLM_PROVIDER: {provider!r} (expected 'anthropic' or 'openai')"
        )
    return client

def build_player(kind: str, *, seed: int = 0, resume: bool = False) -> Player:
    if kind == "human":
        return HumanPlayer()
    if kind == "first":
        return FirstLegalPlayer()
    if kind == "random":
        return RandomPlayer(seed=seed)
    if kind == "greedy":
        return GreedyPlayer()
    if kind == "llm":
        client = build_llm_client()
        log_path = f"./logs/{seed}.json"
        return LLMPlayer(client=client, seed=seed, log_path=log_path, resume=resume)
    raise ValueError(f"unknown player kind: {kind!r} (expected human/first/random/greedy/llm)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Play a Twilight Struggle game.")
    parser.add_argument("--us", default="human")
    parser.add_argument("--ussr", default="human")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume any LLM player(s) from their existing log file at "
            "--seed instead of starting with fresh memory. Never implied "
            "by reusing a seed -- must be passed explicitly. Note this "
            "only resumes each LLMPlayer's own conversation memory, not "
            "the Engine's game state, which always starts fresh."
        ),
    )
    args = parser.parse_args()

    engine = Engine.new_game(seed=args.seed)
    players: dict[Side, Player] = {
        Side.US: build_player(args.us, seed=args.seed + 1, resume=args.resume),
        Side.USSR: build_player(args.ussr, seed=args.seed + 2, resume=args.resume),
    }

    winner = play_game(engine, players)
    print(f"\nWinner: {winner}")


if __name__ == "__main__":
    main()
