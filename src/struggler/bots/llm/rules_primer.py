"""A prose explanation of Twilight Struggle's mechanics, for the LLM prompt
(see prompt.py).

The engine already hands the model raw numeric constants (`rules.json`) and
raw board data (`countries.json`), but neither explains the *formulas*
those numbers feed into -- Control, Coup/Realignment resolution, regional
scoring tiers, DEFCON effects, victory conditions, and so on. Without this,
the model has to reconstruct Twilight Struggle's rules from general
training knowledge alone, which is exactly the gap this module closes.

Written and verified against two sources: the physical rulebook (GMT Games,
"Twilight Struggle -- Deluxe Edition" rules booklet) and this engine's own
implementation (`engine/board.py`, `engine/core.py`), cross-checked
line-by-line. Where the two disagree, this primer describes the engine's
*actual* behavior -- the model plays against the engine, not the printed
rulebook -- and the deviation is called out below.

Only mechanics this engine actually implements are covered (see
docs/CARDS.md). The rulebook's Tournament Play, Chinese Civil War Variant, and
Late War Scenario sections describe rules this engine does not implement
and are deliberately omitted -- including them would mislead the model
about what's actually in play.
"""

from __future__ import annotations

RULES_PRIMER = "\n".join(
    [
        "RULES PRIMER (Twilight Struggle mechanics, as this engine actually "
        "implements them -- only mechanics this engine implements are "
        "described; where this differs from the printed rulebook, this is "
        "the engine's real behavior):",
        "",
        "TURN STRUCTURE: each turn runs (1) DEFCON improves by 1 if below 5, "
        "(2) both players are dealt back up to hand size, (3) Headline phase, "
        "(4) Action Rounds, (5) required-Military-Operations check, "
        "(6) The China Card flips face up for its current holder, (7) turn "
        "advances. Headline phase: both players secretly pick a card, then "
        "reveal simultaneously; the card with the higher Ops value (its "
        "'Headline Value') resolves first, ties go to the US card; a "
        "Scoring card has Headline Value 0 and always resolves second (US "
        "first among two Scoring headlines); neither player gets Ops points "
        "from a headlined card, only its Event. Action Rounds: USSR always "
        "acts first in each round; players alternate playing one card per "
        "round for 6 rounds (turns 1-3) or 7 rounds (turns 4-10); a card "
        "left over at the end is 'held' for next turn (Scoring cards can "
        "never be held).",
        "",
        "CARD PLAY -- EVENT VS. OPS VS. SPACE RACE: playing a non-Scoring "
        "card is a three-way choice, made fresh every time: spend it for "
        "its Event, spend its Ops value, or discard it to attempt a Space "
        "Race advance instead (eligible whenever the card's Ops value "
        "meets the next box's requirement and you haven't used up this "
        "turn's Space Race attempt yet -- see SPACE RACE below). A card is "
        "never split between two of these; exactly one applies. A Scoring "
        "card must be played the turn it's drawn, for its Event (Scoring "
        "cards have no Ops/Space Race use). IMPORTANT -- choosing 'ops' does "
        "NOT let you dodge an opponent's Event: if you play a card for Ops "
        "and that card's Event belongs to your OPPONENT, the Event still "
        "fires -- you (the phasing player) only choose whether it resolves "
        "before or after your Ops are spent (the engine's EVENT_OPS_ORDER "
        "decision); you cannot make it not happen this way. Space Race is "
        "the ONLY mode that avoids an opponent's Event firing (discarding "
        "the card for Space Race never triggers any Event, yours or "
        "theirs) -- if the goal is specifically to deny/avoid a dangerous "
        "opponent Event, Space Race is the way to do that, not 'ops'. A "
        "NEUTRAL card's Event never fires this way either, only when "
        "played as the Event. Playing your OWN card's Event just resolves "
        "it directly, no ordering choice. The China Card has no Event -- "
        "it is always Ops-only.",
        "",
        "OPERATIONS -- WHAT OPS CAN BE SPENT ON: once you've chosen to "
        "spend a card's Ops (rather than its Event or a Space Race "
        "attempt), that Ops value picks exactly one of: place Influence, "
        "attempt Realignment rolls, or attempt a Coup.",
        "  - Influence placement: costs 1 Op per point in a country that's "
        "uncontrolled or friendly-Controlled, 2 Ops per point in a country "
        "the opponent Controls. You may only place Influence in a country "
        "that is adjacent to your own superpower's home space, already "
        "holds your Influence, or is adjacent to another country where you "
        "already hold Influence.",
        "  - Coup attempt: no adjacency requirement, but the opponent must "
        "hold at least 1 Influence there. Formula: "
        "margin = die_roll + ops_spent - 2*country.stability (+ any "
        "modifiers). If margin > 0, remove that many opponent Influence "
        "points from the country (any leftover margin becomes your own "
        "Influence there instead). Every Coup attempt degrades DEFCON by 1 "
        "(exception: a US Coup in a Battleground country while Nuclear "
        "Subs is in effect this turn does not degrade DEFCON).",
        "  - Realignment roll: no adjacency or existing-Influence "
        "requirement either, but the opponent must hold at least 1 "
        "Influence there; costs 1 Op per roll, and multiple rolls (even at "
        "the same country) may be bought with one card's Ops. Each side "
        "rolls a die and adds: +1 if their own superpower is adjacent to "
        "the target, +1 for every country adjacent to the target that they "
        "Control, and +1 if they already hold more Influence in the target "
        "than their opponent does. The higher total wins; the margin is "
        "removed from the LOSING side's Influence in that country -- "
        "meaning a losing roll costs the acting side their own Influence, "
        "not just a wasted attempt. Ties remove nothing. Realignment never "
        "adds Influence to a country. The region bonus below (China Card / "
        "Vietnam Revolts) also applies here: one extra roll if every "
        "attempt this Ops-spend stayed inside the bonus region.",
        "  - Space Race: discard the card to attempt to advance your "
        "marker one box, if the card's Ops value is at least the next "
        "box's requirement; roll the die and advance if it lands within "
        "that box's success range. Boxes award Victory Points (first- and "
        "second-to-arrive amounts differ) and some grant a special ability "
        "that only the first side to reach that box holds, cancelled the "
        "instant the second side also reaches it.",
        "  - Coup and Realignment region lock: a region can only be "
        "targeted while DEFCON is at or above that region's minimum: "
        "Europe needs DEFCON >= 5, Asia >= 4, Middle East >= 3. Africa, "
        "Central America, and South America have no DEFCON restriction.",
        "",
        "CONTROL: a side Controls a country when "
        "(their Influence there) - (opponent's Influence there) >= "
        "country.stability. Otherwise the country is uncontrolled.",
        "",
        "DEFCON: ranges 1-5 and can never rise above 5. If DEFCON ever "
        "reaches 1, the game ends immediately and the side that CAUSED the "
        "drop to 1 loses (their opponent wins).",
        "",
        "REQUIRED MILITARY OPERATIONS: at the end of every turn, each side "
        "needs to have spent Military Operations points (from Coups and "
        "War-family Events) at least equal to the current DEFCON number. "
        "Falling short awards the opponent 1 VP per point of shortfall.",
        "",
        "REGIONAL SCORING (triggered by playing that region's Scoring card, "
        "or at end-of-turn-10 final scoring for every region at once):",
        "  - Presence: the side Controls at least 1 country in the region.",
        "  - Domination: the side Controls MORE countries in the region "
        "than the opponent, AND Controls more Battleground countries there "
        "than the opponent, AND Controls at least 1 non-Battleground "
        "country there (this also implies Controlling >= 1 Battleground).",
        "  - Control: the side Controls EVERY Battleground country in the "
        "region AND Controls more countries overall than the opponent. "
        "Europe has no separate Control VP value -- Controlling every "
        "country in Europe wins the game outright the moment Europe is "
        "scored (see VICTORY below), instead of paying out VP.",
        "  - Bonus VP on top of the tier's base VP: +1 VP per Battleground "
        "country the side Controls in that region, +1 VP per country the "
        "side Controls in that region that is adjacent to the ENEMY "
        "superpower's home space.",
        "  - The two sides' totals are compared and only the NET "
        "difference is applied to the VP track.",
        "",
        "CHINA CARD: no Event, Ops-only. If every one of its Ops points is "
        "spent inside Asia (including Southeast Asia), it grants +1 extra "
        "Op; a Coup targeting an Asian country with this bonus active also "
        "gets +1 Op and +1 Military Operations credit; a Realignment spend "
        "gets one extra attempt under the same all-or-nothing condition "
        "(see the Realignment roll bullet above). Passed face-down to the "
        "opponent after use; flips face up for them at end of turn.",
        "",
        "VICTORY: the game ends immediately, in favor of whichever side "
        "triggers it, the instant any of: (a) a side's VP total reaches "
        "+/-20 (an automatic win for that side), (b) a side Controls every "
        "country in Europe at the moment Europe is scored, (c) DEFCON hits "
        "1 (the side that caused it loses). If none of these have happened "
        "by the end of turn 10, every region is scored once more as final "
        "scoring and whichever side then holds a positive (US) or negative "
        "(USSR) VP total wins; an exact 0 is a draw.",
    ]
)
