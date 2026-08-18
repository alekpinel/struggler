"""CLI entry point: play a game with any mix of human/bot seats.

Examples:
    python src/main.py                                    # human vs human
    python src/main.py --us greedy --ussr random --seed 1  # bot vs bot
    python src/main.py --ussr greedy                       # human (US) vs bot (USSR)
    python src/main.py --ussr llm
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

from struggler.bots.greedy import GreedyPlayer
from struggler.bots.llm.client import LLMClient
from struggler.bots.llm.player import LLMPlayer
from struggler.bots.naive import FirstLegalPlayer, RandomPlayer
from struggler.engine import Engine, Side
from struggler.engine.human import HumanPlayer
from struggler.engine.physical import OperatorConsolePlayer
from struggler.engine.player import Player
from struggler.engine.replay import HistoryBuilder, replay_history
from struggler.runner import play_game


DEFAULT_LLM_PROVIDER = "openai"
DEFAULT_LLM_MODELS = {"anthropic": "claude-opus-5", "openai": "gpt-5.6-luna"}


def build_llm_client(provider: str | None = None, model: str | None = None):
    """Build an LLM client, defaulting to the STRUGGLER_LLM_* environment.

    `provider` and `model` override the environment when given; otherwise
    `STRUGGLER_LLM_PROVIDER` / `STRUGGLER_LLM_MODEL` are consulted, falling
    back to this module's defaults.
    """
    provider = provider or os.environ.get("STRUGGLER_LLM_PROVIDER", DEFAULT_LLM_PROVIDER)
    if provider not in DEFAULT_LLM_MODELS:
        raise ValueError(
            f"unknown STRUGGLER_LLM_PROVIDER: {provider!r} (expected 'anthropic' or 'openai')"
        )
    model = model or os.environ.get("STRUGGLER_LLM_MODEL") or DEFAULT_LLM_MODELS[provider]
    if provider == "anthropic":
        from struggler.bots.llm.anthropic_client import AnthropicClient
        client: LLMClient = AnthropicClient(model=model)
    else:
        from struggler.bots.llm.openai_client import OpenAIClient
        client = OpenAIClient(model=model)
    return client


def build_player(
    kind: str,
    *,
    seed: int = 0,
    resume: bool = False,
    log_path: str | None = None,
    side_label: str = "",
) -> Player:
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
        if log_path is None:
            if resume:
                raise ValueError(
                    "--resume requires an explicit log path (--us-log-path/--ussr-log-path): "
                    "log filenames are timestamped at creation, not derived from --seed."
                )
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
            suffix = f"_{side_label}" if side_label else ""
            log_path = f"./logs/{timestamp}{suffix}.json"
        return LLMPlayer(client=client, seed=seed, log_path=log_path, resume=resume)
    raise ValueError(f"unknown player kind: {kind!r} (expected human/first/random/greedy/llm)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Play a Twilight Struggle game.")
    parser.add_argument("--us", default="human")
    parser.add_argument("--ussr", default="human")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--physical",
        choices=["us", "ussr"],
        default=None,
        help=(
            "Physical-board mode: the given side is a real human playing the "
            "physical board game. --us/--ussr for THAT side is ignored (the "
            "operator console answers for it); the other side is still built "
            "from --us/--ussr as normal. All dealing and dice route to the "
            "operator console too, for both sides."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume an LLM player from its existing log file instead of "
            "starting with fresh memory. Requires --us-log-path and/or "
            "--ussr-log-path (log filenames are timestamped at creation"
        ),
    )
    parser.add_argument(
        "--us-log-path",
        default=None,
        help="Log path for the US LLM player. Defaults to a new timestamped file under ./logs/.",
    )
    parser.add_argument(
        "--ussr-log-path",
        default=None,
        help="Log path for the USSR LLM player. Defaults to a new timestamped file under ./logs/.",
    )
    parser.add_argument(
        "--game-log-path",
        default=None,
        help="Path for the full game replay log. Defaults to a new timestamped file under ./logs/.",
    )
    parser.add_argument(
        "--no-game-log",
        action="store_true",
        help="Disable saving the game replay log.",
    )
    parser.add_argument(
        "--resume-game-log",
        default=None,
        help=(
            "Resume a live game from an existing game-log file (see "
            "engine.replay.replay_history) instead of starting a new one -- "
            "e.g. after hand-trimming the file's \"actions\" to undo a bad "
            "play before continuing. The engine and the Player-facing "
            "history are both rebuilt from the log, and play continues to "
            "accumulate at the same path. Physical mode and its side are "
            "read from the log itself, not from --physical; --us/--ussr "
            "still select the bot for the non-physical (or either) side, "
            "with its own RNG seeded off the log's own --seed, not this "
            "process's."
        ),
    )
    args = parser.parse_args()

    if args.resume_game_log:
        with open(args.resume_game_log, encoding="utf-8") as f:
            log = json.load(f)
        engine, history = replay_history(log)
        seed = log["seed"]
        if log.get("physical_mode"):
            physical_side = Side(log["physical_side"])
            bot_side = physical_side.opponent
            bot_kind = args.us if bot_side is Side.US else args.ussr
            operator = OperatorConsolePlayer()
            bot_log_path = args.us_log_path if bot_side is Side.US else args.ussr_log_path
            players: dict[Side, Player] = {
                physical_side: operator,
                Side.CHANCE: operator,
                bot_side: build_player(
                    bot_kind, seed=seed + 1, resume=args.resume,
                    log_path=bot_log_path, side_label=bot_side.value.lower(),
                ),
            }
        else:
            players = {
                Side.US: build_player(
                    args.us, seed=seed + 1, resume=args.resume, log_path=args.us_log_path, side_label="us"
                ),
                Side.USSR: build_player(
                    args.ussr, seed=seed + 2, resume=args.resume, log_path=args.ussr_log_path, side_label="ussr"
                ),
            }
        winner = play_game(
            engine,
            players,
            log_path=args.resume_game_log,
            history_builder=HistoryBuilder(initial_history=history),
            initial_actions=log["actions"],
        )
        print(f"\nWinner: {winner}")
        return

    if args.physical:
        physical_side = Side.US if args.physical == "us" else Side.USSR
        bot_side = physical_side.opponent
        bot_kind = args.us if bot_side is Side.US else args.ussr
        engine = Engine.new_game(seed=args.seed, physical_mode=True, physical_side=physical_side)
        operator = OperatorConsolePlayer()
        bot_log_path = args.us_log_path if bot_side is Side.US else args.ussr_log_path
        players: dict[Side, Player] = {
            physical_side: operator,
            Side.CHANCE: operator,
            bot_side: build_player(
                bot_kind, seed=args.seed + 1, resume=args.resume,
                log_path=bot_log_path, side_label=bot_side.value.lower(),
            ),
        }
    else:
        engine = Engine.new_game(seed=args.seed)
        players = {
            Side.US: build_player(
                args.us, seed=args.seed + 1, resume=args.resume, log_path=args.us_log_path, side_label="us"
            ),
            Side.USSR: build_player(
                args.ussr, seed=args.seed + 2, resume=args.resume, log_path=args.ussr_log_path, side_label="ussr"
            ),
        }

    if args.no_game_log:
        game_log_path = None
    elif args.game_log_path is not None:
        game_log_path = args.game_log_path
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        game_log_path = f"./logs/{timestamp}_game.json"

    winner = play_game(engine, players, log_path=game_log_path)
    print(f"\nWinner: {winner}")


if __name__ == "__main__":
    main()
