"""Pure Flip 7 scoring functions and score breakdowns."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from flip7.core.cards import Card, ModifierCard, ModifierCardName, NumberCard
from flip7.core.state import GameState, PlayerStatus

_ADDITIVE_MODIFIERS: dict[ModifierCardName, int] = {
    ModifierCardName.PLUS_2: 2,
    ModifierCardName.PLUS_4: 4,
    ModifierCardName.PLUS_6: 6,
    ModifierCardName.PLUS_8: 8,
    ModifierCardName.PLUS_10: 10,
}

FLIP7_BONUS = 15


@dataclass(frozen=True, slots=True)
class ScoreBreakdown:
    """Deterministic round-score components for one player."""

    number_sum: int
    doubled_number_sum: int
    modifier_total: int
    flip7_bonus: int
    round_score: int
    busted: bool
    player_id: int = 0


def score_cards(
    cards: Sequence[Card],
    *,
    busted: bool = False,
    flip7: bool = False,
    player_id: int = 0,
) -> ScoreBreakdown:
    """Score one player's cards using the official scoring order."""
    if busted:
        return ScoreBreakdown(
            number_sum=0,
            doubled_number_sum=0,
            modifier_total=0,
            flip7_bonus=0,
            round_score=0,
            busted=True,
            player_id=player_id,
        )

    number_sum = sum(card.value for card in cards if isinstance(card, NumberCard))
    has_times_two = any(
        isinstance(card, ModifierCard) and card.name is ModifierCardName.TIMES_2
        for card in cards
    )
    doubled_number_sum = number_sum * 2 if has_times_two else number_sum
    modifier_total = sum(
        _ADDITIVE_MODIFIERS[card.name]
        for card in cards
        if isinstance(card, ModifierCard) and card.name in _ADDITIVE_MODIFIERS
    )
    flip7_bonus = FLIP7_BONUS if flip7 else 0
    return ScoreBreakdown(
        number_sum=number_sum,
        doubled_number_sum=doubled_number_sum,
        modifier_total=modifier_total,
        flip7_bonus=flip7_bonus,
        round_score=doubled_number_sum + modifier_total + flip7_bonus,
        busted=False,
        player_id=player_id,
    )


def calculate_round_scores(state: GameState) -> tuple[ScoreBreakdown, ...]:
    """Score every seated player from the current round state."""
    return tuple(
        score_cards(
            player.cards,
            busted=player.status is PlayerStatus.BUSTED,
            flip7=state.flip7_player_id == player.player_id,
            player_id=player.player_id,
        )
        for player in state.players
    )
