"""Short, hand-maintained mechanical summaries of each implemented card
event, for the LLM prompt (see prompt.py).

Per the "no full card text" design decision: these are NOT the physical
card's printed text, only a one-line paraphrase of what the engine's
`events.py` actually does mechanically -- derived from the short inline
comments already present there, the same source of truth. A card id
absent from this dict has no implemented event in this game version:
playing it as "event" is a no-op discard, exactly like in the real engine
(see events.py's `EVENTS` registry and CLAUDE.md's M3 section).

This can drift from `events.py` as M3 evolves; there is no automated sync
check in v1 (documented known limitation, see player.py).
"""

from __future__ import annotations

EVENT_MECHANICAL_SUMMARIES: dict[str, str] = {
    # -- immediate, fixed board/VP/DEFCON/space effects --
    "Duck_and_Cover": "DEFCON -1, then US +VP equal to (5 - new DEFCON).",
    "Fidel": "USSR gains Control of Cuba outright.",
    "Romanian_Abdication": "USSR gains Control of Romania outright.",
    "Nasser": "USSR +2 Influence in Egypt; remove half (round up) of US Influence there.",
    "De_Gaulle_Leads_France": "Remove 2 US / +1 USSR Influence in France; NATO no longer protects France.",
    "Captured_Nazi_Scientist": "Phasing side advances one Space Race box (its VP too).",
    "Nuclear_Test_Ban": "Phasing side +VP equal to (DEFCON - 2), then DEFCON improves 2 levels.",
    "Korean_War": "USSR war vs South Korea, win on 4-6, +2 VP and seizes target, +2 military ops on success.",
    "Arab_Israeli_War": "USSR war vs Israel (blocked if Camp David Accords in effect), win on 4-6, +2 VP, +2 military ops.",
    "Allende": "USSR +2 Influence in Chile.",
    "Portuguese_Empire_Crumbles": "USSR +2 Influence in Angola and +2 in SE African States.",
    "Panama_Canal_Returned": "US +1 Influence each in Panama, Costa Rica, Venezuela.",
    "Sadat_Expels_Soviets": "Remove all USSR Influence from Egypt; US +1 Influence there.",
    "John_Paul_II_Elected_Pope": "Remove 2 USSR / +1 US Influence in Poland; enables Solidarity.",
    "Camp_David_Accords": "US +1 VP, +1 Influence each in Israel/Jordan/Egypt; blocks Arab-Israeli War.",
    "Iranian_Hostage_Crisis": "Remove all US Influence from Iran, USSR +2 there; makes Terrorism hit the US twice.",
    "The_Iron_Lady": "US +1 VP, USSR +1 Influence in Argentina, remove all USSR Influence from UK; blocks Socialist Governments.",
    "An_Evil_Empire": "US +1 VP; cancels Flower Power.",
    "U2_Incident": "USSR +1 VP.",
    "Cultural_Revolution": "If US holds the China Card, USSR takes it face up; else USSR +1 VP.",
    "Ortega_Elected_in_Nicaragua": "Remove all US Influence from Nicaragua; USSR gets a free Coup-only (no Realignment) op vs a Nicaragua neighbor.",
    "Tear_Down_This_Wall": "Cancels Willy Brandt; US +3 Influence in East Germany; US gets a free Coup-or-Realignment op in Europe.",
    "Kitchen_Debates": "If the US controls more Battlegrounds than the USSR, US +2 VP.",
    "OPEC": "USSR +VP equal to how many of a fixed oil-country list it controls (unless North Sea Oil is in effect).",
    "Alliance_for_Progress": "US +VP equal to Battlegrounds it controls in Central/South America.",
    "Reagan_Bombs_Libya": "US +VP equal to (USSR Influence in Libya // 2).",
    "One_Small_Step": "If behind on the Space Race, phasing side advances 2 boxes; VP is scored for the second box only, not the first.",
    "AWACS_Sale_to_Saudis": "US +2 Influence in Saudi Arabia; blocks Muslim Revolution.",
    "CIA_Created": "US conducts 1 Op of Operations.",
    "Lone_Gunman": "USSR conducts 1 Op of Operations.",
    "ABM_Treaty": "DEFCON +1, then phasing side conducts 4 Ops of Operations.",
    # -- player-choice influence (place/remove across candidate countries) --
    "COMECON": "USSR +1 Influence each to 4 non-US-controlled Eastern Europe countries.",
    "Marshall_Plan": "US +1 Influence each to 7 non-USSR-controlled Western Europe countries; enables NATO.",
    "Decolonization": "USSR +1 Influence each to 4 countries in Africa and/or Southeast Asia.",
    "Suez_Crisis": "Remove up to 4 US Influence total from France/UK/Israel, max 2 per country.",
    "Truman_Doctrine": "Remove all USSR Influence from one uncontrolled European country.",
    "Warsaw_Pact_Formed": "Branch: remove all US Influence from 4 Eastern Europe countries, OR USSR +5 Influence there (max 2/country); enables NATO.",
    "Socialist_Governments": "Remove up to 3 US Influence from Western Europe, max 2 per country.",
    "Muslim_Revolution": "Remove all US Influence from 2 of a fixed Middle-East/North-Africa country list.",
    "Colonial_Rear_Guards": "US +1 Influence each to 4 countries in Africa/Southeast Asia.",
    "Liberation_Theology": "USSR +1 Influence each to up to 3 Central America countries, max 2 per country.",
    "The_Voice_Of_America": "Remove up to 4 USSR Influence from non-Europe countries, max 2 per country.",
    "Puppet_Governments": "US +1 Influence each to up to 3 countries with zero Influence from either side.",
    "OAS_Founded": "US +1 Influence each to up to 2 Central/South America countries.",
    "Pershing_II_Deployed": "USSR +1 VP; remove up to 3 US Influence from Western Europe, max 1 per country.",
    "The_Reformer": "USSR +1 Influence in Europe, total 6 if USSR is ahead on VP else 4, max 2 per country; USSR may no longer coup in Europe.",
    "Solidarity": "US +3 Influence in Poland (requires John Paul II Elected Pope to have fired).",
    "Marine_Barracks_Bombing": "Remove all US Influence from Lebanon; remove up to 2 more US Influence from the rest of the Middle East.",
    "East_European_Unrest": "Remove USSR Influence from 3 Eastern Europe countries (1 each Early/Mid War, 2 each Late War).",
    "South_African_Unrest": "Branch: USSR +2 Influence in South Africa, OR +1 there and +2 in an adjacent country.",
    "Latin_American_Death_Squads": "+1 to the phasing side's own coup rolls, -1 to the opponent's, in Central/South America for the rest of the turn.",
    "Iran_Contra_Scandal": "US Realignment rolls are -1 for the rest of the turn.",
    "Chernobyl": "US names a region; USSR may not add Influence there via Operations for the rest of the turn.",
    "Junta": "Phasing side +2 Influence in a single Central/South America country, then may make one free Coup or Realignment there.",
    "The_Cambridge_Five": "USSR +1 Influence in a country from a region the US holds a scoring card for (blocked during Late War).",
    # -- wars where the attacker picks the target --
    "Indo_Pakistani_War": "Attacker's war vs India or Pakistan (their choice), win on 4-6, +2 VP, +2 military ops.",
    "Iran_Iraq_War": "Attacker's war vs Iran or Iraq (their choice), win on 4-6, +2 VP, +2 military ops.",
    "Brush_War": "Attacker's war vs any stability-1-or-2 country (their choice), win on 3-6, +1 VP, +3 military ops.",
    # -- match-influence --
    "Independent_Reds": "US Influence in one chosen Eastern European country is raised to match USSR's there.",
    # -- forced random discard --
    "Five_Year_Plan": "USSR randomly discards a card; if it's a USSR event, that event fires too.",
    "Terrorism": "Opponent randomly discards a card (twice if USSR plays it after Iranian Hostage Crisis).",
    # -- persistent per-turn modifiers --
    "Containment": "All US Operations +1 for the rest of the turn.",
    "Brezhnev_Doctrine": "All USSR Operations +1 for the rest of the turn.",
    "Red_Scare_Purge": "Opponent's Operations -1 (min 1) for the rest of the turn.",
    "Nuclear_Subs": "US Battleground coups don't degrade DEFCON this turn.",
    "Vietnam_Revolts": "USSR +2 Influence in Vietnam; USSR gets +1 Op for the rest of the turn on plays fully spent in Southeast Asia.",
    # -- persistent game-long triggers/legality --
    "NATO": "USSR may no longer Coup, Realign, or play Brush War against US-controlled Europe (requires Marshall Plan or Warsaw Pact Formed).",
    "US_Japan_Mutual_Defense_Pact": "US gains Control of Japan; USSR may never Coup/Realign against Japan again.",
    "Willy_Brandt": "USSR +1 VP, +1 Influence in West Germany; NATO no longer protects West Germany.",
    "Flower_Power": "USSR +2 VP every time the US plays a war card, until An Evil Empire is played.",
    "Yuri_and_Samantha": "USSR +1 VP for every US coup attempt, for the rest of the game.",
    "Formosan_Resolution": "While the US controls Taiwan, it scores as an Asian Battleground; nullified once the US plays the China Card (not when the USSR does).",
    "Shuttle_Diplomacy": "At the next Middle East or Asia scoring, one USSR-controlled Battleground doesn't count.",
    "North_Sea_Oil": "OPEC can no longer be played as an event; US gets one extra action round this turn.",
    "NORAD": "If Canada is US-controlled, whenever DEFCON moves to 2, US adds 1 Influence to a country where it already has some.",
    # -- set-DEFCON --
    "How_I_Learned_to_Stop_Worrying": "Set DEFCON to any level, then +5 to the phasing side's Military Operations track.",
    # -- reclaim from discard --
    "Salt_Negotiations": "DEFCON +2; both sides' coup rolls -1 for the rest of the turn; phasing side may reclaim one non-scoring card from the discard pile.",
    # -- dice-contest / branch --
    "Olympic_Games": "Opponent chooses: a 2-die contest (sponsor +2, winner +2 VP) or boycott (DEFCON -1, sponsor conducts 4 Ops).",
    "Summit": "Both roll +1 per region Dominated/Controlled; winner +2 VP and may raise/lower/leave DEFCON.",
    "Wargames": "Only at DEFCON 2: phasing side may give the opponent 6 VP and end the game now, or decline.",
    # -- revealing/taking opponent hand cards --
    "Aldrich_Ames_Remix": "USSR sees the US hand and picks one card the US must discard.",
    "Grain_Sales_to_Soviets": "One USSR card is randomly revealed; US plays it in full (Event or Ops) or returns it (US conducts 2 Ops instead); if the USSR hand is empty, US just gets 2 Ops.",
    "Ask_Not_What_Your_Country_Can_Do_For_You": "Phasing side may discard any number of non-scoring hand cards and redraw that many.",
    # -- take-and-play --
    "Missile_Envy": "Exchange for the opponent's highest-Ops card (opponent breaks ties); Missile Envy passes to them and they must play it for Ops on their next action round; the taker uses the taken card for Ops, or its Event if eligible.",
    "Star_Wars": "Only while the US leads the Space Race: US takes a non-scoring card from the discard pile and fires its event immediately.",
    # -- free coup with conditional repeat --
    "Che": "USSR makes a free Coup vs a non-Battleground country in Central/South America/Africa; if it removes any US Influence, a second free Coup vs a different such country.",
    # -- deferred per-turn conditions --
    "Cuban_Missile_Crisis": "DEFCON set to 2; a Coup by the flagged opponent this turn loses them the game; they may defuse at the start of any of their action rounds this turn by removing 2 Influence from Cuba (USSR) or West Germany/Turkey (US).",
    "We_Will_Bury_You": "DEFCON -1; USSR +3 VP at end of turn unless the US plays UN Intervention first.",
    # -- scoring-time modifiers --
    "Blockade": "Unless the US discards a 3+-Ops card, remove all US Influence from West Germany.",
    "Glasnost": "USSR +2 VP, DEFCON +1; if The Reformer is in effect, USSR then gets 4 Ops restricted to Influence/Realignment (no Coup).",
    "Latin_American_Debt_Crisis": "Unless the US discards a 3+-Ops card, USSR doubles its Influence in up to 2 South America countries.",
    "Soviets_Shoot_Down_KAL_007": "DEFCON -1, US +2 VP; if the US controls South Korea, US then gets 4 Ops restricted to Influence/Realignment (no Coup).",
    "Ussuri_River_Skirmish": "If USSR holds the China Card, US takes it face up; else US +4 Influence in Asia, max 2 per country.",
    "Arms_Race": "If the phasing side leads on the Military Operations track, +3 VP if it also meets the DEFCON threshold, else +1 VP.",
    "De_Stalinization": "USSR relocates up to 4 Influence: remove from anywhere it has some, then re-place the same number in non-US-controlled countries, max 2 per country.",
    "Special_Relationship": "Only eligible while the US controls the UK. If NATO is not in effect: US +1 Influence to a UK neighbor. If NATO is in effect: US +2 Influence to one Western Europe country, and US +2 VP.",
    "Nixon_Plays_The_China_Card": "If the US already holds the China Card: US +2 VP. If the USSR holds it: the US takes it face down (unusable this turn). No discard-to-keep option.",
    # -- hidden peek at the draw pile --
    "Our_Man_In_Tehran": "US looks at the top up to 5 draw-pile cards one at a time, keeping or removing each; kept cards are reshuffled back in.",
    # -- persistent per-player operating lock --
    "Bear_Trap": "Traps the USSR: each action round it must discard a 2+-Ops card and roll a die (1-4 frees it, 5-6 stays trapped); with no such card, the round is wasted with no roll (any scoring card in hand is still forced into play).",
    "Quagmire": "Traps the US: each action round it must discard a 2+-Ops card and roll a die (1-4 frees it, 5-6 stays trapped); with no such card, the round is wasted with no roll (any scoring card in hand is still forced into play).",
    # -- headline hook + a separate action-round trigger (no EVENTS entry; see events.py's note) --
    "Defectors": "A US headline of Defectors discards the USSR's headlined card unresolved. Separately, if the USSR plays it in a normal action round (Event or Ops, not Space Race), US +1 VP. A USSR headline or a US action-round play have no effect.",
    # -- rule-modifier play mode (no EVENTS entry; see core.py's un_intervention play mode) --
    "UN_Intervention": "Play mode 'un_intervention': spend this card to use an opponent's already-implemented, eligible event card for its Ops value with that card's event cancelled.",
}
