"""The rules engine: the pending-decision-stack state machine and the
public API contract from CLAUDE.md (`Engine`, plus the dataclasses/enums
a caller needs to drive it: `Side`, `DecisionKind`, `Action`, `Decision`,
`Observation`). Board/card/event internals live in their own submodules
(`.board`, `.cards`, `.events`, `.replay`, `.rules`) and are reached
directly by anything that needs them.
"""

from __future__ import annotations

from struggler.engine.core import Engine
from struggler.engine.types import (
    Action,
    Card,
    CardSide,
    Decision,
    DecisionKind,
    Observation,
    Period,
    Region,
    ScoringTier,
    Side,
    Subregion,
)

__all__ = [
    "Engine",
    "Action",
    "Card",
    "CardSide",
    "Decision",
    "DecisionKind",
    "Observation",
    "Period",
    "Region",
    "ScoringTier",
    "Side",
    "Subregion",
]
