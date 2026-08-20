"""Roll back one side's most recent LLM decision so a resumed game lets it
be re-examined, per the hand-trim workflow docs/BOTS.md's "Resuming a live
game" section describes.

A `--side`'s `LLMPlayer` decides in *batches*: one real call can predict
several steps, only the first of which is guaranteed to match when reached
-- the rest get silently consumed later with no further call, until the
plan runs out, mismatches, or the turn changes (see `bots/llm/player.py`).
"The last decision" therefore means "since the side's last real LLM call",
not just the last single game action: the side log's last `journal` entry
of kind "decision" names that call's `decision_id`; this replays the game
log to find the action where that exact decision was pending and cuts from
there, dropping every action that call produced or that got consumed from
its resulting plan. The side log is trimmed to match: the call's message
pair and journal entry are dropped, its `plan` (entirely a product of the
call being undone) is cleared, and -- if the call also happened to be the
turn's planning call -- that call's message pair/journal entry go too, with
`turn_plan`/`planned_turn`/`turn_plan_history` rolled back to the previous
turn's. `last_seen` is then recomputed from whatever journal entry is now
last (0 if none remain) rather than trusted as still-current, so the result
is itself safe to undo again -- running this script N times in a row undoes
the last N calls.

Usage:
    python scripts/undo_last_decision.py logs/2026-08-19_11-15 --side ussr
    python scripts/undo_last_decision.py --game logs/x_game.json --side ussr --side-log logs/x_ussr.json
    python scripts/undo_last_decision.py logs/2026-08-19_11-15 --side ussr --dry-run

Then relaunch, e.g.:
    python src/main.py --resume-game-log logs/2026-08-19_11-15_game.json \\
        --ussr llm --resume --ussr-log-path logs/2026-08-19_11-15_ussr.json
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from struggler.engine.replay import HistoryBuilder, decode_action, make_engine
from struggler.engine.types import Side


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _existing(path: str) -> Path | None:
    p = Path(path)
    return p if p.exists() else None


def _backup(path: Path) -> Path:
    """Copy `path` to a `.bak` sibling, numbering it (`.bak.1`, `.bak.2`,
    ...) if a plain `.bak` already exists -- so running this script several
    times in a row keeps a backup of every intermediate state, not just the
    most recent one."""
    candidate = path.with_suffix(path.suffix + ".bak")
    n = 1
    while candidate.exists():
        candidate = path.with_suffix(path.suffix + f".bak.{n}")
        n += 1
    shutil.copy2(path, candidate)
    return candidate


def find_decision(game_log: dict[str, Any], actions: list[dict[str, Any]], decision_id: int) -> tuple[int, Side, int]:
    """Replay `actions` (using `game_log`'s own new_game/seed/... setup, but
    that action list -- so this also works on an already-trimmed prefix),
    returning `(action_index, actor, pre_len)` for the decision with id
    `decision_id`: its index in `actions`, who faced it, and how many
    history entries existed right before it -- i.e. what a resumed player's
    `last_seen` must equal to treat it as still-pending.
    """
    engine = make_engine({**game_log, "actions": actions})
    builder = HistoryBuilder()
    for index, action_data in enumerate(actions):
        decision = engine.pending_decision
        pre_len = len(builder.history)
        action = decode_action(action_data)
        engine.step(action)
        builder.record(decision, action, engine)
        if decision.id == decision_id:
            return index, decision.actor, pre_len
    raise ValueError(
        f"decision id {decision_id} (from the side log's journal) was never pending "
        f"while replaying the game log -- the two files may not belong together, or "
        f"the game log was already trimmed past this point"
    )


def trim_side_log(snapshot: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return `(new_snapshot, removed_journal_entries)` with the side log's
    last decision call (and its turn-plan call, if any) undone. `last_seen`
    is left untouched here -- the caller fills it in once it knows the new
    last decision's own `pre_len` (see `find_decision`)."""
    journal = snapshot["journal"]
    if not journal or journal[-1]["kind"] != "decision":
        raise ValueError("the side log's journal has no decision entry left to undo")

    drop_turn_plan = len(journal) >= 2 and journal[-2]["kind"] == "turn_plan"
    removed = journal[-2:] if drop_turn_plan else journal[-1:]
    new_journal = journal[: -len(removed)]

    messages_to_drop = 2 * len(removed)  # one user+assistant pair committed per call
    if len(snapshot["messages"]) < messages_to_drop:
        raise ValueError("side log has fewer messages than the call(s) being undone")
    new_messages = snapshot["messages"][:-messages_to_drop]

    usage_removed = {"input_tokens": 0, "output_tokens": 0}
    for entry in removed:
        for key, value in entry.get("usage", {}).items():
            usage_removed[key] = usage_removed.get(key, 0) + value
    new_cumulative = {
        key: max(0, value - usage_removed.get(key, 0)) for key, value in snapshot["cumulative_usage"].items()
    }

    new_turn_plan_history = list(snapshot.get("turn_plan_history", []))
    if drop_turn_plan:
        turn_plan_entry = journal[-2]
        if not turn_plan_entry.get("fallback_used") and new_turn_plan_history:
            new_turn_plan_history = new_turn_plan_history[:-1]
        new_turn_plan = new_turn_plan_history[-1] if new_turn_plan_history else None
        new_planned_turn = new_turn_plan["turn"] if new_turn_plan else None
    else:
        new_turn_plan = snapshot.get("turn_plan")
        new_planned_turn = snapshot.get("planned_turn")

    new_snapshot = dict(snapshot)
    new_snapshot["journal"] = new_journal
    new_snapshot["messages"] = new_messages
    new_snapshot["plan"] = []
    new_snapshot["turn_plan"] = new_turn_plan
    new_snapshot["planned_turn"] = new_planned_turn
    new_snapshot["turn_plan_history"] = new_turn_plan_history
    new_snapshot["cumulative_usage"] = new_cumulative
    return new_snapshot, removed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "prefix", nargs="?", default=None,
        help="Shared log prefix, e.g. logs/2026-08-19_11-15 (finds <prefix>_game.json and <prefix>_<side>.json)",
    )
    parser.add_argument("--game", type=Path, help="Explicit game log path, overrides prefix discovery")
    parser.add_argument("--side", required=True, choices=("us", "ussr"), help="Which side's last decision to undo")
    parser.add_argument("--side-log", type=Path, help="Explicit conversation log path for --side, overrides prefix discovery")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change without writing anything")
    parser.add_argument("--no-backup", action="store_true", help="Don't write .bak copies before overwriting")
    args = parser.parse_args()

    game_path = args.game
    side_path = args.side_log
    other_side = "us" if args.side == "ussr" else "ussr"
    other_path = None
    if args.prefix is not None:
        game_path = game_path or _existing(f"{args.prefix}_game.json")
        side_path = side_path or _existing(f"{args.prefix}_{args.side}.json")
        other_path = _existing(f"{args.prefix}_{other_side}.json")

    if game_path is None or side_path is None:
        parser.error("need a game log and a side log: give a prefix, or both --game and --side-log")

    try:
        game_log = _load(game_path)
        snapshot = _load(side_path)
        side = Side(args.side.upper())

        journal = snapshot["journal"]
        if not journal or journal[-1]["kind"] != "decision":
            raise ValueError("the side log's journal has no decision entry left to undo")
        decision_id = journal[-1]["decision_id"]

        action_index, actor, _ = find_decision(game_log, game_log["actions"], decision_id)
        if actor is not side:
            raise ValueError(
                f"decision {decision_id} (the side log's last one) belongs to {actor.value} "
                f"in the game log, not {side.value} -- side log/game log mismatch"
            )
        kept_actions = game_log["actions"][:action_index]
        removed_actions = game_log["actions"][action_index:]

        new_snapshot, removed_journal = trim_side_log(snapshot)

        remaining_journal = new_snapshot["journal"]
        if remaining_journal:
            prev_decision_id = remaining_journal[-1]["decision_id"]
            _, _, prev_pre_len = find_decision(game_log, kept_actions, prev_decision_id)
            new_snapshot["last_seen"] = prev_pre_len
        else:
            new_snapshot["last_seen"] = 0

        if other_path is not None:
            other_snapshot = _load(other_path)
            other_last_seen = other_snapshot.get("last_seen", 0)
            if other_last_seen > len(kept_actions):
                raise ValueError(
                    f"{other_path} (the {other_side.upper()} side's own log) already saw "
                    f"{other_last_seen} history events, more than the {len(kept_actions)} that "
                    f"would remain -- undo that side's decisions too first, or its resumption "
                    f"contract will break"
                )
    except (ValueError, KeyError) as exc:
        parser.error(str(exc))
        return

    drop_turn_plan = len(removed_journal) == 2
    print(f"Undoing {args.side.upper()}'s last decision (decision id {decision_id}):")
    for action in removed_actions:
        extra = f" country={action['country']}" if "country" in action else ""
        print(f"  ar{action['action_round']} {action['actor']:>6} {action['kind']:<20} {json.dumps(action['payload'])}{extra}")
    print(f"  ({len(removed_actions)} game-log action(s) removed, {len(kept_actions)} remain)")
    for entry in removed_journal:
        label = "turn-plan call" if entry["kind"] == "turn_plan" else "decision call"
        detail = entry.get("justification") or entry.get("fallback_reason") or "(no justification recorded)"
        print(f"  dropping {label}: {detail}")
    if drop_turn_plan:
        print("  (this decision was the first of a new turn -- its turn-plan call is undone too)")

    if args.dry_run:
        print("\n(dry run -- nothing written)")
        return

    if not args.no_backup:
        print(f"backed up {game_path} -> {_backup(game_path)}")
        print(f"backed up {side_path} -> {_backup(side_path)}")

    new_game_log = dict(game_log)
    new_game_log["actions"] = kept_actions
    new_game_log["winner"] = None
    _write(game_path, new_game_log)
    _write(side_path, new_snapshot)

    print(f"\nWrote {game_path} and {side_path}.")
    print("Relaunch with:")
    print(
        f"  python src/main.py --resume-game-log {game_path} "
        f"--{args.side} llm --resume --{args.side}-log-path {side_path}"
    )
    if not game_log.get("physical_mode"):
        print(f"  (add --{other_side} <kind> too if {other_side.upper()} isn't a human)")


if __name__ == "__main__":
    main()
