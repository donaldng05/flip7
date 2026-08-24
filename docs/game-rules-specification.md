# Flip 7 Game Rules Specification

## 1. Authority and Scope

This document is the authoritative normative rules specification for this
repository's Flip 7 game engine.

The source authority is the official **Flip 7 Ruleset Edition 3.1**. This
specification translates those table rules into deterministic engine behavior
for simulation, testing, reinforcement-learning environments, and evaluation.

This document defines only the published Flip 7 rules needed for an accurate
game simulator. It does not define house rules, strategy advice, UI behavior,
reward shaping, observation encodings, training policy, or production service
behavior.

Normative keywords are used as follows:

- **Must**: required engine behavior.
- **Must not**: prohibited engine behavior.
- **May**: legal player choice or allowed implementation representation.
- **Decision**: an explicit project ruling for an ambiguity or implementation
  detail not fully specified by the table rules.

## 2. Core Terminology

- **Game**: one complete contest consisting of one or more rounds, ending after
  a round in which at least one player reaches the game-end threshold.
- **Round**: one pass through drawing, actions, staying, busting, scoring, and
  dealer rotation.
- **Deck**: the face-down draw pile for the current game.
- **Discarded cards**: cards from completed rounds and cards discarded by rule
  effects. They are unavailable until a reshuffle is required.
- **Active player**: a player still eligible to receive cards or choose Hit or
  Stay in the current round. A player who stays, freezes, busts, or ends the
  round with Flip 7 is no longer active.
- **Inactive player**: a player who has left the current round by staying,
  freezing, busting, or round end.
- **Number card**: a card with value 0 through 12. Number cards determine busts,
  round score, and Flip 7.
- **Action card**: one of Freeze, Flip Three, or Second Chance. Action cards do
  not count as Number cards.
- **Modifier card**: one of +2, +4, +6, +8, +10, or x2. Modifier cards do not
  count as Number cards.
- **Hit**: a voluntary player action that requests one card from the deck.
- **Stay**: a voluntary player action, legal only when the player has at least
  one card in front of them, that banks the player's current cards for
  end-of-round scoring and makes the player inactive.
- **Bust**: the state caused when a player receives a duplicate Number card and
  does not use Second Chance to cancel that duplicate.
- **Flip 7**: the condition where a player has seven unique Number cards in
  front of them during a round.
- **Round score**: points earned by a non-busted player in a round.
- **Cumulative score**: a player's total score across all completed rounds in a
  game.

## 3. Deck Composition

A single Flip 7 deck must contain exactly 94 cards:

| Card category | Cards | Count |
| --- | --- | ---: |
| Number cards | 0 | 1 |
| Number cards | 1 | 1 |
| Number cards | 2 | 2 |
| Number cards | 3 | 3 |
| Number cards | 4 | 4 |
| Number cards | 5 | 5 |
| Number cards | 6 | 6 |
| Number cards | 7 | 7 |
| Number cards | 8 | 8 |
| Number cards | 9 | 9 |
| Number cards | 10 | 10 |
| Number cards | 11 | 11 |
| Number cards | 12 | 12 |
| Action cards | Freeze | 3 |
| Action cards | Flip Three | 3 |
| Action cards | Second Chance | 3 |
| Modifier cards | +2 | 1 |
| Modifier cards | +4 | 1 |
| Modifier cards | +6 | 1 |
| Modifier cards | +8 | 1 |
| Modifier cards | +10 | 1 |
| Modifier cards | x2 | 1 |

Total Number cards: 79. Total Action cards: 9. Total Modifier cards: 6.

The baseline engine must validate player count before setup:

- 3 to 18 players are valid with one official deck.
- 1 or 2 players are out-of-scope challenge variants and must be rejected by the
  baseline engine.
- More than 18 players requires an explicit multi-deck extension and must be
  rejected by the baseline engine.

## 4. Setup

1. Create the configured deck and shuffle it.
2. Choose the first dealer by configured deterministic setup, random selection,
   or external table agreement.
3. Seat order is fixed for the game.
4. At the start of each round, all players in a valid baseline game become
   active and have no cards in front of them.
5. The dealer deals one face-up card at a time in seat order, beginning with the
   player to the dealer's left and ending with the dealer, until each player who
   remains active has received one initial dealt card, except where action-card
   resolution changes card counts.
6. If an Action card is dealt during initial dealing, dealing pauses and that
   Action card resolves immediately. Initial dealing resumes afterward with the
   next player in seat order who still needs an initial dealt card.
7. An Action card dealt as a player's own initial card satisfies that player's
   initial-card requirement, even if the card is assigned to another player and
   the dealt player has no cards in front of them afterward.
8. A player targeted by an Action card before receiving their own initial dealt
   card still needs an initial dealt card if they remain active. If that player
   becomes inactive before receiving an initial dealt card, they are skipped for
   the rest of the initial deal and remain inactive with only the cards, if any,
   produced by the action that affected them.
9. An inactive player with no cards in front of them scores zero for the round.
   An inactive player with only Modifier cards scores those Modifier cards
   according to the normal scoring order, except that x2 alone contributes zero.
10. Cards dealt, drawn, placed, set aside, or discarded are public information.

The dealer participates as a player. Dealer status controls dealing and
rotation; it does not grant special scoring or action privileges.

## 5. Round Flow

A round proceeds through these phases:

1. **Initial deal**: each player who remains active receives one initial
   face-up card, with Action cards resolved immediately.
2. **Turn cycle**: the dealer offers each active player, in seat order, a legal
   Hit or Stay decision.
3. **Card resolution**: every dealt or drawn card resolves according to its type.
4. **Round end**: the round ends immediately when one player achieves Flip 7, or
   when no active players remain.
5. **Scoring**: all non-busted players score their cards in the defined scoring
   order.
6. **Cleanup and rotation**: cards from the round are set aside, cumulative
   scores are updated, the dealer passes left, and the next round begins unless
   the game has ended.

## 6. Legal Player Decisions

An active player who is offered a normal turn must choose a legal action:

- **Hit**: receive and resolve the next card from the deck.
- **Stay**: become inactive and keep current cards for end-of-round scoring.

Hit is legal for an active player when the round has not ended and the player
can receive a card. Stay is legal only for an active player with at least one
card in front of them.

When a player receives a targetable Action card, the player resolving that card
must choose a legal active target unless the rule gives no choice.

An inactive player must not choose Hit, Stay, or targets. A busted player must
not receive Action cards as a target.

## 7. Card Resolution

### 7.1 Number Cards

When a player receives a Number card:

1. If the player does not already have that number, place the card with the
   player's Number cards.
2. If the player already has that number and does not have a usable Second
   Chance, the player busts immediately.
3. If the player already has that number and has a usable Second Chance, discard
   the Second Chance and the duplicate Number card. The player does not bust.
4. After a non-busting Number card is placed, check whether the player has seven
   unique Number cards. If so, the player achieves Flip 7 and the round ends
   immediately.

Only duplicate Number cards can cause a bust. Action cards and Modifier cards
must not cause a bust.

### 7.2 Modifier Cards

When a player receives a Modifier card, place it with that player's cards.

Modifier cards:

- do not count toward the seven Number cards required for Flip 7;
- cannot cause a bust;
- score only if the player does not bust;
- may be the only scoring cards a player has if the player has no Number cards,
  except that x2 alone contributes no points without Number-card value to double.

### 7.3 Action Cards

Action cards are playable on any active player, including the player who
received the Action card.

If exactly one active player exists when a targetable Action card must be
assigned, that player must be the target.

If an Action card cannot legally affect its chosen target because the target is
no longer active when resolution reaches that card, the Action card is discarded.

## 8. Action Card Rules

### 8.1 Freeze

Freeze targets one active player.

The target immediately banks the points represented by their current cards and
becomes inactive for the rest of the round. A frozen player is not busted and
scores at round end.

Freeze itself is discarded after resolution.

### 8.2 Flip Three

Flip Three targets one active player.

The target must receive up to the next three cards, one at a time. Each received
card counts toward the three-card total, including Number cards, Modifier cards,
and Action cards.

The Flip Three sequence stops immediately if the target busts or achieves Flip
7. If the target achieves Flip 7, the round ends immediately. If the target
busts, the target scores zero for the round and any set-aside Flip Three or
Freeze cards from that sequence are discarded.

During a Flip Three sequence:

1. Number cards resolve normally.
2. Modifier cards are placed normally.
3. Second Chance resolves immediately according to the Second Chance rules.
4. Additional Flip Three and Freeze cards are temporarily set aside and count
   toward the three required cards.
5. If the target busts or achieves Flip 7 before all three cards are drawn, the
   set-aside Action cards are discarded.
6. If the target receives all three required cards without busting or achieving
   Flip 7, the player who is resolving the original Flip Three assigns each
   set-aside Flip Three or Freeze, in the order drawn, to a legal active target.
   Those assigned Action cards then resolve in assignment order.

The original Flip Three card is discarded after its sequence and any deferred
Action-card assignment is complete.

### 8.3 Second Chance

Second Chance is kept by the player who receives it, subject to the one-card
limit.

A player may have at most one Second Chance in front of them. If a player who
already has Second Chance receives another Second Chance, that player must assign
the newly received Second Chance to another active player who does not already
have one. If no such active player exists, the newly received Second Chance is
discarded.

When a player with Second Chance receives a duplicate Number card, the Second
Chance and duplicate Number card are discarded immediately. The player's original
Number card remains, and the player does not bust.

Unused Second Chance cards are discarded at the end of the round, including
Second Chance cards held by players who froze or by a player whose Flip 7 ended
the round.

## 9. Scoring

Only non-busted players score points for a round. A busted player scores zero,
regardless of cards received before busting.

For each non-busted player, calculate round score in this order:

1. Sum the values of the player's Number cards.
2. If the player has x2, double the Number-card sum.
3. Add all additive Modifier cards: +2, +4, +6, +8, and +10.
4. If the player achieved Flip 7, add the 15-point Flip 7 bonus.

The x2 Modifier doubles only the sum of Number cards. It does not double
additive Modifier cards or the Flip 7 bonus.

After round scores are calculated, add each player's round score to their
cumulative game score.

## 10. Flip 7

A player achieves Flip 7 immediately when they have seven unique Number cards in
front of them.

When Flip 7 occurs:

1. The round ends immediately.
2. The Flip 7 player is not busted.
3. The Flip 7 player receives the 15-point Flip 7 bonus during scoring.
4. All other non-busted players score their current cards.
5. No further pending voluntary turns occur.
6. Deferred Flip Three and Freeze Action cards not yet assigned are discarded.

Action and Modifier cards do not count toward Flip 7.

## 11. Round End, Cleanup, and Reshuffle

A round ends when either:

- one player achieves Flip 7; or
- no active players remain because every player has busted, stayed, or frozen.

At round end:

1. Calculate and apply round scores.
2. Discard all cards from the round, including unused Second Chance cards.
3. Do not shuffle round cards back into the deck unless the draw pile is empty
   and a reshuffle is required.
4. Pass the remaining deck to the player on the dealer's left. That player
   becomes the dealer for the next round.

When the deck runs out, shuffle all discarded cards to form a new draw pile. If
a reshuffle is needed mid-round, cards currently in front of players remain in
place, including cards in front of busted players, frozen players, and players
who have stayed.

## 12. Game End and Winner

The game-end threshold is 200 cumulative points.

The game does not end in the middle of a round solely because a player reaches
or exceeds 200 points during a provisional scoring calculation. The game ends
after round scoring if at least one player has a cumulative score of 200 or
more.

When the game ends, the player with the highest cumulative score wins.

If multiple players tie for the highest cumulative score after the final round,
the engine must report a tied win among those players unless a future approved
rules source specifies a tie-breaker.

## 13. Visibility and Engine State

The engine must preserve complete internal state sufficient to reproduce the
game exactly:

- seat order and dealer;
- active/inactive/busted status;
- each player's visible cards;
- draw pile order;
- discarded cards;
- pending Flip Three sequences and deferred Action cards;
- cumulative scores;
- round and game completion state.

Player observations for agents may hide draw-pile order or other complete-state
details, but the core engine must resolve rules from complete internal state.

## 14. Worked Examples

### 14.1 Normal Hit and Stay

Player A starts a round with Number cards 6 and 8. On their next turn they Hit
and receive +4. They later Stay. If they do not bust before round end, they score
6 + 8 + 4 = 18 points.

### 14.2 Duplicate-Number Bust

Player B has Number cards 3, 5, and 10, with no Second Chance. They Hit and
receive another 5. Player B busts immediately and scores zero for the round.

### 14.3 Second Chance Prevents Bust

Player C has Number cards 2 and 11 plus one Second Chance. They receive another
11. The duplicate 11 and Second Chance are discarded. Player C keeps the original
11, does not bust, and remains active.

### 14.4 Freeze

Player D has Number cards 9 and 12 plus +6. Another player assigns Freeze to
Player D. Player D becomes inactive and will score 9 + 12 + 6 = 27 points at
round end unless the game state is already ending for another reason.

### 14.5 Flip Three With Deferred Action

Player E receives Flip Three. The next three cards are 4, Freeze, and +8. The 4
and +8 are placed with Player E's cards, and Freeze is set aside while still
counting as one of the three cards. If Player E does not bust or achieve Flip 7,
the player resolving Flip Three assigns that deferred Freeze to a legal active
target, and it resolves after the three-card sequence.

### 14.6 Flip Three Stops on Bust

Player F has Number cards 7 and 10, with no Second Chance. During Flip Three,
the first card is another 10. Player F busts immediately, receives no remaining
Flip Three cards, and scores zero for the round.

### 14.7 Flip Three Discards Deferred Action After Bust

Player G receives Flip Three. The next cards are Freeze and a duplicate Number
card that busts Player G. The Flip Three sequence stops immediately, no third
card is drawn, and the deferred Freeze is discarded.

### 14.8 Flip Three Stops on Flip 7

Player H has six unique Number cards. During Flip Three, the first card is a
new unique Number card. Player H achieves Flip 7 immediately. The round ends,
remaining Flip Three cards are not drawn, and Player H receives the 15-point
Flip 7 bonus during scoring.

### 14.9 Modifier Scoring Order

Player I has Number cards 4, 8, and 12 plus x2, +6, and +10. Player I scores
(4 + 8 + 12) x 2 + 6 + 10 = 64 points.

### 14.10 Game Ending

Player J starts a round with 184 cumulative points. They score 20 points in the
round, reaching 204. Player K scores 35 points in the same round, reaching 207.
Because at least one player reached 200, the game ends after the round. Player K
wins with the highest cumulative score.

### 14.11 Final-Round Tie

Player L and Player M both finish the final round with 205 cumulative points,
and no player has more. The engine reports both Player L and Player M as tied
winners.

### 14.12 Second Chance Transfer

Player N already has a Second Chance. They receive another Second Chance. Player
N must assign the newly received Second Chance to another active player who does
not already have one. If no such player exists, the newly received Second Chance
is discarded.

### 14.13 Initial Deal Action Before Initial Card

During the initial deal, Player A receives Freeze as their own initial card and
assigns it to Player C, who has not yet received an initial dealt card. Player C
becomes inactive with no cards and is skipped when the initial deal reaches their
seat. Player A has satisfied their own initial-card requirement, remains active
with no cards, may Hit on their normal turn, and may not Stay until they have at
least one card in front of them.

## 15. Decision Log and Ambiguities

### D1. Seat Order and First Turn After Initial Deal

**Decision:** Seat order is fixed, dealer passes left, and normal Hit/Stay
offers begin with the active player to the dealer's left.

**Reason:** The official rules require dealer rotation to the left and normal
turn-order play. This gives deterministic engine behavior while preserving the
published turn structure.

### D2. Initial Deal Action Timing

**Decision:** Action cards dealt during the initial deal resolve immediately.
The dealt Action card satisfies the dealt player's initial-card requirement.
Being targeted by another player's initial-deal Action card does not satisfy the
target's initial-card requirement. If the target remains active, the target still
receives an initial dealt card when the deal reaches them. If the target becomes
inactive before receiving an initial dealt card, the target is skipped for the
rest of the initial deal and scores only cards actually in front of them, or zero
if they have none.

**Reason:** The official rules say to pause dealing, resolve the Action card,
and then continue dealing. This decision preserves that flow while making the
engine state deterministic when an action affects a player before their own
initial dealt card.

### D3. Deferred Flip Three Actions

**Decision:** Flip Three and Freeze cards revealed during a Flip Three sequence
are deferred and count toward the three-card total. Under the supplied Ruleset
Edition 3.1 baseline, they resolve only after all three cards are drawn and only
if the target has not busted. If the target busts, the sequence stops and those
deferred Action cards are discarded. They are also discarded if the target
achieves Flip 7, because the round ends immediately.

**Reason:** The official rules state that these cards count toward the three
cards and resolve after all three cards are drawn, with the supplied Edition 3.1
review finding clarifying that this happens only if the target has not busted.

### D4. Deferred Action Target Becomes Inactive

**Decision:** If a deferred Action card is assigned to a player who is no longer
active by the time that deferred card would resolve, discard the Action card.

**Reason:** Action cards can be played only on active players. This avoids
retroactively affecting players who stayed, froze, busted, or were removed from
the active set by an earlier deferred action.

### D5. Tied Highest Score at Game End

**Decision:** A tied highest cumulative score after the final round produces
multiple winners.

**Reason:** Ruleset Edition 3.1 states that the player with the most points wins
after a round in which at least one player reaches 200, but no tie-breaker is
specified. Reporting tied winners is the least-inventive deterministic behavior.

### D6. Baseline Player Count

**Decision:** The baseline engine accepts 3 to 18 players with one official
94-card deck. It rejects 1 or 2 players as out-of-scope challenge variants. It
also rejects more than 18 players unless a future explicit multi-deck extension
is approved and configured.

**Reason:** The official baseline rules are for 3 to 18 players with one deck.
The official recommendation for more than 18 players requires a second deck, so
that behavior is outside the baseline one-deck engine.

### D7. Scoring Frozen Players

**Decision:** A frozen player keeps their current cards and scores them during
the same end-of-round scoring pass as other non-busted players.

**Reason:** Freeze says the player banks collected points and is out of the
round. A single scoring pass keeps cumulative scoring deterministic and matches
the rule that round scores are applied at round end.

### D8. Winner Is Determined After Full Round Scoring

**Decision:** If multiple players reach or exceed 200 in the same round, all
round scores are applied before the winner is determined.

**Reason:** Ruleset Edition 3.1 states that at the end of the round, when at
least one player reaches 200, the player with the most points wins.

### D9. Initial Deal Starts To The Dealer's Left

**Decision:** Initial dealing and later Hit/Stay offers both begin with the
active player to the dealer's left and proceed in fixed seat order.

**Reason:** The official rules deal and offer turns in turn order, with the
dealer included. Starting left of the dealer makes both sequences deterministic
and consistent.

### D10. Exhausted Draw And Discard Piles

**Decision:** If a draw is required when both the draw pile and discard pile are
empty, the engine must raise a focused domain error and must not invent cards or
silently reuse cards that remain in front of players.

**Reason:** Cards in front of players stay in place during a reshuffle. If no
unplayed cards remain, the published rules do not define a replacement card.
