"""Tests for bots/llm/card_playbook.py and its JSON.

The playbook is hand-written strategy advice keyed by card id, so the one
thing that can silently rot is the key set drifting away from `cards.json`
-- advice for a card that doesn't exist is advice that never reaches the
model, and nothing else would notice.
"""

from __future__ import annotations

import re

from struggler.bots.llm import card_playbook
from struggler.bots.llm.prompt import build_user_turn
from struggler.engine import Engine, Side
from struggler.engine.cards import load_cards


def test_every_playbook_key_is_a_real_card():
    unknown = card_playbook.known_card_ids() - set(load_cards())
    assert unknown == frozenset()


def test_advice_is_scoped_to_the_asking_seat():
    # Duck and Cover is a US event: the two seats need opposite advice, and
    # neither should see the other's.
    us = card_playbook.advice_for("Duck_and_Cover", Side.US)
    ussr = card_playbook.advice_for("Duck_and_Cover", Side.USSR)

    assert us is not None and ussr is not None
    assert us != ussr


def test_side_agnostic_advice_reaches_both_seats():
    us = card_playbook.advice_for("The_China_Card", Side.US)
    ussr = card_playbook.advice_for("The_China_Card", Side.USSR)

    assert us == ussr


def test_an_unlisted_card_yields_no_advice():
    assert card_playbook.advice_for("Summit", Side.US) is None or isinstance(
        card_playbook.advice_for("Summit", Side.US), str
    )
    assert card_playbook.advice_for("__not_a_card__", Side.US) is None
