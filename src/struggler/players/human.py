"""A console `Player`: prompts a human for each decision via stdin/stdout."""

from __future__ import annotations

from struggler.types import Action, Observation


class HumanPlayer:
    """Prints the current decision and reads a choice from the console.

    Only ever returns an `Action` object taken verbatim from
    `observation.pending_decision.options` — never a hand-built payload —
    so the engine's legality check always passes.
    """

    def choose_action(self, observation: Observation) -> Action:
        decision = observation.pending_decision
        print(
            f"\n[{observation.side.value}] turn {observation.turn}, "
            f"action round {observation.action_round}, "
            f"DEFCON {observation.defcon}, VP {observation.vp}"
        )
        print(f"Decision: {decision.kind.value} (context: {dict(decision.context)})")
        for i, action in enumerate(decision.options):
            print(f"  [{i}] {action.kind.value} {dict(action.payload)}")

        while True:
            raw = input(f"Choose an option [0-{len(decision.options) - 1}]: ").strip()
            if raw.isdigit() and int(raw) < len(decision.options):
                return decision.options[int(raw)]
            print("Invalid choice, try again.")
