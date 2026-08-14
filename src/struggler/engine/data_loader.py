"""Central "resolve path under data/, open, json.load" helper (mandate #5):
every module that reads a data/*.json file routes through here instead of
each hand-rolling its own Path(__file__).resolve()... + open() + json.load().
"""

from __future__ import annotations

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_json(filename: str) -> dict:
    """Load and parse `filename` from struggler/data/."""
    with (DATA_DIR / filename).open("r", encoding="utf-8") as f:
        return json.load(f)
