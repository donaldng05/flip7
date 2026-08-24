"""Tests for authoritative Flip 7 card and deck mechanics."""

import random
from collections import Counter

import pytest

from flip7.core import (
    ActionCard,
    ActionCardName,
    Deck,
    ModifierCard,
    ModifierCardName,
    NumberCard,
    card_counts,
    create_standard_deck,
)


def test_standard_deck_has_authoritative_94_card_composition() -> None:
    cards = create_standard_deck()

    assert len(cards) == 94
    assert card_counts(cards) == Counter(
        {
            NumberCard(0): 1,
            NumberCard(1): 1,
            NumberCard(2): 2,
            NumberCard(3): 3,
            NumberCard(4): 4,
            NumberCard(5): 5,
            NumberCard(6): 6,
            NumberCard(7): 7,
            NumberCard(8): 8,
            NumberCard(9): 9,
            NumberCard(10): 10,
            NumberCard(11): 11,
            NumberCard(12): 12,
            ActionCard(ActionCardName.FREEZE): 3,
            ActionCard(ActionCardName.FLIP_THREE): 3,
            ActionCard(ActionCardName.SECOND_CHANCE): 3,
            ModifierCard(ModifierCardName.PLUS_2): 1,
            ModifierCard(ModifierCardName.PLUS_4): 1,
            ModifierCard(ModifierCardName.PLUS_6): 1,
            ModifierCard(ModifierCardName.PLUS_8): 1,
            ModifierCard(ModifierCardName.PLUS_10): 1,
            ModifierCard(ModifierCardName.TIMES_2): 1,
        }
    )


def test_card_definitions_are_immutable_and_validate_number_range() -> None:
    number_card = NumberCard(12)

    with pytest.raises(AttributeError):
        number_card.value = 7  # type: ignore[reportAttributeAccessIssue]

    with pytest.raises(ValueError, match="Number cards must be between 0 and 12"):
        NumberCard(13)


def test_deck_draws_without_replacement_using_injected_rng() -> None:
    first = Deck.standard(random.Random(11))
    second = Deck.standard(random.Random(11))

    first_draws = [first.draw() for _ in range(10)]
    second_draws = [second.draw() for _ in range(10)]

    assert first_draws == second_draws
    assert card_counts(first_draws) <= card_counts(create_standard_deck())
    assert first.draw_count == 84
    assert first.discard_count == 0


def test_deck_exposes_draw_and_discard_card_counts() -> None:
    deck = Deck(
        random.Random(3),
        draw_pile=(NumberCard(1), NumberCard(1), ModifierCard(ModifierCardName.PLUS_4)),
    )

    drawn = deck.draw()
    deck.discard([drawn, ActionCard(ActionCardName.FREEZE)])

    assert deck.draw_counts == Counter(
        {NumberCard(1): 1, ModifierCard(ModifierCardName.PLUS_4): 1}
    )
    assert deck.discard_counts == Counter(
        {NumberCard(1): 1, ActionCard(ActionCardName.FREEZE): 1}
    )


def test_deck_reshuffles_discards_only_when_draw_pile_is_empty() -> None:
    deck = Deck(
        random.Random(5),
        draw_pile=(NumberCard(3),),
        discard_pile=(NumberCard(7), NumberCard(8)),
    )

    assert deck.draw() == NumberCard(3)
    assert deck.discard_count == 2

    reshuffled_card = deck.draw()

    assert reshuffled_card in {NumberCard(7), NumberCard(8)}
    assert deck.draw_count == 1
    assert deck.discard_count == 0


def test_deck_reports_empty_when_no_draw_or_discard_cards_remain() -> None:
    deck = Deck(random.Random(1), draw_pile=())

    with pytest.raises(IndexError, match="Cannot draw from an empty deck"):
        deck.draw()
