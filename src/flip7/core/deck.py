"""Deterministic Flip 7 draw, discard, and reshuffle mechanics."""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Iterable

from flip7.core.cards import Card, card_counts, create_standard_deck


class EmptyDeckError(IndexError):
    """Raised when a draw is required and no unplayed cards remain."""


class Deck:
    """A deterministic Flip 7 deck using caller-supplied randomness only."""

    def __init__(
        self,
        rng: random.Random,
        *,
        draw_pile: Iterable[Card],
        discard_pile: Iterable[Card] = (),
    ) -> None:
        self._rng = rng
        self._draw_pile = list(draw_pile)
        self._discard_pile = list(discard_pile)

    @classmethod
    def standard(cls, rng: random.Random) -> Deck:
        """Create a shuffled standard Flip 7 deck."""
        draw_pile = list(create_standard_deck())
        rng.shuffle(draw_pile)
        return cls(rng, draw_pile=draw_pile)

    @property
    def draw_count(self) -> int:
        """Number of cards currently available in the draw pile."""
        return len(self._draw_pile)

    @property
    def discard_count(self) -> int:
        """Number of discarded cards waiting for a required reshuffle."""
        return len(self._discard_pile)

    @property
    def draw_counts(self) -> Counter[Card]:
        """Card counts currently in the draw pile."""
        return card_counts(self._draw_pile)

    @property
    def discard_counts(self) -> Counter[Card]:
        """Card counts currently in the discard pile."""
        return card_counts(self._discard_pile)

    @property
    def draw_cards(self) -> tuple[Card, ...]:
        """Current draw-pile order, next card first."""
        return tuple(self._draw_pile)

    @property
    def discard_cards(self) -> tuple[Card, ...]:
        """Cards waiting for a required reshuffle."""
        return tuple(self._discard_pile)

    def draw(self) -> Card:
        """Draw the next card, reshuffling discards if the draw pile is empty."""
        if not self._draw_pile:
            self._reshuffle_discards()

        if not self._draw_pile:
            msg = "Cannot draw from an empty deck"
            raise EmptyDeckError(msg)

        return self._draw_pile.pop(0)

    def discard(self, cards: Iterable[Card]) -> None:
        """Add supplied cards to the discard pile."""
        self._discard_pile.extend(cards)

    def _reshuffle_discards(self) -> None:
        if not self._discard_pile:
            return

        self._draw_pile = self._discard_pile
        self._discard_pile = []
        self._rng.shuffle(self._draw_pile)
