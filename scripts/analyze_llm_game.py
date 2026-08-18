"""Structured dump of a played game for reviewing an `LLMPlayer`'s turn
plans against what actually happened -- the extraction half of the
"analyze a game" skill (see .claude/skills/analyze-llm-game/SKILL.md). The
judgement (did the plan make sense, was it followed well) is still a
reasoning task for whoever reads this output; this script only gets the
raw material out of the two log formats without blowing the caller's
context budget on the full files, which routinely run past 500KB-900KB of
mostly-repeated board state (see docs/BOTS.md's turn-plan section and
LIMITATIONS.md's `LLMPlayer` note).

Two file formats feed this:
  - a `runner.play_game(..., log_path=...)` game log (`engine/replay.py`,
    `GameLogWriter`): `{seed, actions: [...], winner}`, the authoritative
    play-by-play (both sides, CHANCE rolls included).
  - a `bots/llm/conversation_log.py` `ConversationSnapshot`, one per LLM
    seat: `{turn_plan_history, journal, ...}`. `turn_plan_history` (snapshot
    version 3+) has every turn's plan already parsed; older logs only have
    it inline in `journal` entries of kind "turn_plan" via `raw_responses`,
    so this script reconstructs it from there when the field is missing or
    empty (see `_turn_plans_for`).

Usage:
    python scripts/analyze_llm_game.py logs/2026-08-18_15-56
    python scripts/analyze_llm_game.py logs/2026-08-18_15-56 --turn 4 --side ussr
    python scripts/analyze_llm_game.py logs/2026-08-18_15-56 --errors
    python scripts/analyze_llm_game.py --game logs/x_game.json --ussr logs/y_ussr.json

With no --turn, prints a compact overview: game result, the VP/DEFCON/card
timeline per turn, one line per turn plan objective per side, and every
fallback (error) journal entry. With --turn N, prints that turn's full plan
and every decision's justification for the requested side(s), interleaved
with that turn's actual game-log actions so plan and execution sit next to
each other.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


# -- loading ------------------------------------------------------------------


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _discover(prefix: Path) -> tuple[Path | None, dict[str, Path]]:
    """Given `logs/2026-08-18_15-56`, find `<prefix>_game.json` and any of
    `<prefix>_us.json` / `<prefix>_ussr.json` that exist next to it."""
    game_path = Path(f"{prefix}_game.json")
    side_paths = {
        side: p
        for side in ("us", "ussr")
        if (p := Path(f"{prefix}_{side}.json")).exists()
    }
    return (game_path if game_path.exists() else None), side_paths


# -- game log -------------------------------------------------------------


def _actions_by_turn(game: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    by_turn: dict[int, list[dict[str, Any]]] = {}
    for action in game["actions"]:
        by_turn.setdefault(action["turn"], []).append(action)
    return by_turn


def _card_plays_line(actions: list[dict[str, Any]]) -> str:
    plays = [
        f"{a['actor']}:{a['payload']['card']}({a['kind'].removesuffix('_play')})"
        for a in actions
        if a["kind"] in ("headline_play", "action_round_play")
    ]
    return ", ".join(plays) if plays else "(no card plays recorded)"


def print_overview(game: dict[str, Any] | None, side_snapshots: dict[str, dict[str, Any]]) -> None:
    if game is not None:
        actions = game["actions"]
        print(f"=== GAME: seed={game['seed']} winner={game.get('winner')} "
              f"actions={len(actions)} ===")
        print("(vp: negative favors USSR, positive favors US; auto-win at +/-20)")
        by_turn = _actions_by_turn(game)
        for turn in sorted(by_turn):
            turn_actions = by_turn[turn]
            last = turn_actions[-1]
            print(
                f"  Turn {turn}: defcon={last['defcon']} vp={last['vp']} "
                f"({len(turn_actions)} actions) -- {_card_plays_line(turn_actions)}"
            )
        last_action = actions[-1] if actions else None
        if game.get("winner") is None and last_action is not None:
            print(
                f"  NOTE: winner is null and the log ends mid-turn {last_action['turn']} "
                f"action_round {last_action['action_round']} ({last_action['actor']} "
                f"{last_action['kind']}) -- check --errors for why."
            )
        print()

    for side, snapshot in side_snapshots.items():
        print(f"=== {side.upper()} turn plans (provider={snapshot.get('provider')} "
              f"model={snapshot.get('model')}) ===")
        for turn, plan in sorted(_turn_plans_for(snapshot).items()):
            print(f"  Turn {turn}: {plan.get('objective', '(no objective recorded)')}")
        errors = _fallback_entries(snapshot)
        if errors:
            print(f"  {len(errors)} fallback/error decision(s) -- see --errors")
        print()


# -- conversation log -------------------------------------------------------------


def _turn_plans_for(snapshot: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """`{turn: plan_dict}`, preferring the structured `turn_plan_history`
    (snapshot version 3+) and falling back to parsing each `journal` entry
    of kind "turn_plan" from its `raw_responses[0]` for older logs that
    never had that field."""
    history = snapshot.get("turn_plan_history") or []
    if history:
        return {plan["turn"]: plan for plan in history}

    plans: dict[int, dict[str, Any]] = {}
    turn = 0
    for entry in snapshot.get("journal", []):
        if entry.get("kind") != "turn_plan":
            continue
        turn += 1  # best-effort: the Nth turn_plan journal entry is turn N
        raw = entry.get("raw_responses") or []
        if not raw:
            continue
        try:
            plans[turn] = json.loads(raw[0])
        except (json.JSONDecodeError, IndexError):
            continue
    return plans


def _fallback_entries(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    return [e for e in snapshot.get("journal", []) if e.get("fallback_used")]


def _journal_by_turn(snapshot: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    """Segments `journal` at each kind=="turn_plan" entry -- the journal
    itself doesn't carry a turn number per entry, but a new turn always
    starts with exactly one such entry (see player.py's `_make_turn_plan`),
    or with no entry at all if planning is off (`plan_turns=False`), in
    which case everything lands under turn 0."""
    by_turn: dict[int, list[dict[str, Any]]] = {}
    turn = 0
    for entry in snapshot.get("journal", []):
        if entry.get("kind") == "turn_plan":
            turn += 1
        by_turn.setdefault(turn, []).append(entry)
    return by_turn


def print_turn_detail(
    turn: int,
    game: dict[str, Any] | None,
    side_snapshots: dict[str, dict[str, Any]],
) -> None:
    for side, snapshot in side_snapshots.items():
        plan = _turn_plans_for(snapshot).get(turn)
        print(f"=== {side.upper()} TURN {turn} PLAN ===")
        if plan is None:
            print("  (no plan recorded for this turn -- planning may have failed or been off)")
        else:
            for key in (
                "assessment", "objective", "military_ops_plan", "scoring_cards",
                "card_plan", "influence_targets", "defend", "contingencies",
            ):
                value = plan.get(key)
                if value:
                    print(f"  {key}: {json.dumps(value, indent=4, ensure_ascii=False)}")
        print()

        entries = _journal_by_turn(snapshot).get(turn, [])
        decisions = [e for e in entries if e.get("kind") == "decision"]
        print(f"=== {side.upper()} TURN {turn} DECISIONS ({len(decisions)}) ===")
        for i, entry in enumerate(decisions, 1):
            if entry.get("fallback_used"):
                print(f"  [{i}] FALLBACK: {entry.get('fallback_reason')}")
            else:
                print(f"  [{i}] {entry.get('justification')}")
        print()

    if game is not None:
        turn_actions = _actions_by_turn(game).get(turn, [])
        print(f"=== GAME LOG, TURN {turn} ({len(turn_actions)} actions) ===")
        for a in turn_actions:
            extra = f" country={a['country']}" if "country" in a else ""
            print(
                f"  ar{a['action_round']} {a['actor']:>6} {a['kind']:<20} "
                f"{json.dumps(a['payload'])}{extra} "
                f"[defcon={a['defcon']} vp={a['vp']}]"
            )


def print_errors(side_snapshots: dict[str, dict[str, Any]]) -> None:
    for side, snapshot in side_snapshots.items():
        errors = _fallback_entries(snapshot)
        print(f"=== {side.upper()} FALLBACK/ERROR ENTRIES ({len(errors)}) ===")
        for entry in errors:
            print(f"  decision_id={entry.get('decision_id')} at {entry.get('timestamp')}")
            print(f"    {entry.get('fallback_reason')}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "prefix", nargs="?", default=None,
        help="Shared log prefix, e.g. logs/2026-08-18_15-56 "
        "(finds <prefix>_game.json and <prefix>_us.json/<prefix>_ussr.json)",
    )
    parser.add_argument("--game", type=Path, help="Explicit game log path, overrides prefix discovery")
    parser.add_argument("--us", type=Path, help="Explicit US conversation log path")
    parser.add_argument("--ussr", type=Path, help="Explicit USSR conversation log path")
    parser.add_argument("--turn", type=int, default=None, help="Show full detail for one turn instead of the overview")
    parser.add_argument("--side", choices=("us", "ussr"), default=None, help="Restrict --turn/--errors to one side")
    parser.add_argument("--errors", action="store_true", help="List every fallback/error decision instead of the overview")
    args = parser.parse_args()

    game_path = args.game
    side_paths = {}
    if args.us:
        side_paths["us"] = args.us
    if args.ussr:
        side_paths["ussr"] = args.ussr

    if args.prefix is not None:
        discovered_game, discovered_sides = _discover(Path(args.prefix))
        game_path = game_path or discovered_game
        for side, path in discovered_sides.items():
            side_paths.setdefault(side, path)

    if game_path is None and not side_paths:
        parser.error("give a prefix, or at least one of --game/--us/--ussr")

    game = _load(game_path) if game_path is not None else None
    side_snapshots = {side: _load(path) for side, path in side_paths.items()}
    if args.side is not None:
        side_snapshots = {args.side: side_snapshots[args.side]} if args.side in side_snapshots else {}

    if args.errors:
        print_errors(side_snapshots)
    elif args.turn is not None:
        print_turn_detail(args.turn, game, side_snapshots)
    else:
        print_overview(game, side_snapshots)


if __name__ == "__main__":
    main()
