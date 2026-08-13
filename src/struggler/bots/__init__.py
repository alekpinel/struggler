"""Automated `Player` implementations.

Deliberately empty: importing `struggler.bots` alone registers nothing.
Each bot module (`greedy`, `naive`, ...) self-registers with
`struggler.engine.player_registry.register` when *it* is imported, so
callers opt in to exactly the bots they want available by importing their
modules explicitly (see `main.py`).
"""

from __future__ import annotations
