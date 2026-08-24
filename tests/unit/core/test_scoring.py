"""Tests for pure Flip 7 scoring and score-breakdown rules."""

from flip7.core.cards import (
    ActionCard,
    ActionCardName,
    ModifierCard,
    ModifierCardName,
    NumberCard,
)
from flip7.core.scoring import ScoreBreakdown, calculate_round_scores, score_cards
from flip7.core.state import (
    GameState,
    PlayerState,
    PlayerStatus,
)


def test_score_cards_sums_number_cards_and_additive_modifiers() -> None:
    breakdown = score_cards(
        (NumberCard(6), NumberCard(8), ModifierCard(ModifierCardName.PLUS_4))
    )

    assert breakdown == ScoreBreakdown(
        number_sum=14,
        doubled_number_sum=14,
        modifier_total=4,
        flip7_bonus=0,
        round_score=18,
        busted=False,
    )


def test_score_cards_applies_x2_before_additive_modifiers() -> None:
    breakdown = score_cards(
        (
            NumberCard(4),
            NumberCard(8),
            NumberCard(12),
            ModifierCard(ModifierCardName.TIMES_2),
            ModifierCard(ModifierCardName.PLUS_6),
            ModifierCard(ModifierCardName.PLUS_10),
        )
    )

    assert breakdown.number_sum == 24
    assert breakdown.doubled_number_sum == 48
    assert breakdown.modifier_total == 16
    assert breakdown.round_score == 64


def test_score_cards_adds_flip7_bonus_after_modifiers() -> None:
    cards = (
        NumberCard(1),
        NumberCard(2),
        NumberCard(3),
        NumberCard(4),
        NumberCard(5),
        NumberCard(6),
        NumberCard(7),
        ModifierCard(ModifierCardName.TIMES_2),
        ModifierCard(ModifierCardName.PLUS_2),
    )

    breakdown = score_cards(cards, flip7=True)

    assert breakdown.number_sum == 28
    assert breakdown.doubled_number_sum == 56
    assert breakdown.modifier_total == 2
    assert breakdown.flip7_bonus == 15
    assert breakdown.round_score == 73


def test_score_cards_treats_modifier_only_and_x2_only_hands() -> None:
    additive_only = score_cards((ModifierCard(ModifierCardName.PLUS_4),))
    times_only = score_cards((ModifierCard(ModifierCardName.TIMES_2),))

    assert additive_only.round_score == 4
    assert times_only.round_score == 0
    assert times_only.modifier_total == 0


def test_score_cards_returns_zero_for_busted_hands() -> None:
    breakdown = score_cards(
        (NumberCard(3), NumberCard(5), NumberCard(10), NumberCard(5)),
        busted=True,
        flip7=True,
    )

    assert breakdown == ScoreBreakdown(
        number_sum=0,
        doubled_number_sum=0,
        modifier_total=0,
        flip7_bonus=0,
        round_score=0,
        busted=True,
    )


def test_calculate_round_scores_uses_status_and_flip7_player() -> None:
    state = GameState(
        players=(
            PlayerState(
                player_id=0,
                status=PlayerStatus.STAYED,
                cards=(
                    NumberCard(9),
                    NumberCard(12),
                    ModifierCard(ModifierCardName.PLUS_6),
                ),
            ),
            PlayerState(
                player_id=1,
                status=PlayerStatus.BUSTED,
                cards=(NumberCard(3), NumberCard(5), NumberCard(10)),
            ),
            PlayerState(
                player_id=2,
                status=PlayerStatus.FROZEN,
                cards=(
                    NumberCard(1),
                    NumberCard(2),
                    NumberCard(3),
                    NumberCard(4),
                    NumberCard(5),
                    NumberCard(6),
                    NumberCard(7),
                    ActionCard(ActionCardName.SECOND_CHANCE),
                ),
            ),
        ),
        dealer_id=0,
        flip7_player_id=2,
    )

    scores = {
        item.player_id: item.round_score for item in calculate_round_scores(state)
    }

    assert scores == {0: 27, 1: 0, 2: 43}
