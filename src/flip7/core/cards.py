"""Typed immutable Flip 7 card definitions."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum


class ActionCardName(StrEnum):
    """Official Flip 7 action card faces."""

    FREEZE = "freeze"
    FLIP_THREE = "flip_three"
    SECOND_CHANCE = "second_chance"


class ModifierCardName(StrEnum):
    """Official Flip 7 modifier card faces."""

    PLUS_2 = "+2"
    PLUS_4 = "+4"
    PLUS_6 = "+6"
    PLUS_8 = "+8"
    PLUS_10 = "+10"
    TIMES_2 = "x2"


@dataclass(frozen=True, slots=True)
class NumberCard:
    """A Flip 7 number card with value 0 through 12."""

    value: int

    def __post_init__(self) -> None:
        if not 0 <= self.value <= 12:
            msg = "Number cards must be between 0 and 12"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ActionCard:
    """A Flip 7 action card."""

    name: ActionCardName


@dataclass(frozen=True, slots=True)
class ModifierCard:
    """A Flip 7 modifier card."""

    name: ModifierCardName


type Card = NumberCard | ActionCard | ModifierCard


def create_standard_deck() -> tuple[Card, ...]:
    """Create the authoritative unshuffled 94-card Flip 7 deck."""
    cards: list[Card] = []

    for value in range(13):
        cards.extend(NumberCard(value) for _ in range(max(1, value)))

    for action in ActionCardName:
        cards.extend(ActionCard(action) for _ in range(3))

    cards.extend(ModifierCard(modifier) for modifier in ModifierCardName)

    return tuple(cards)


def card_counts(cards: Iterable[Card]) -> Counter[Card]:
    """Count cards by immutable face definition."""
    return Counter(cards)
