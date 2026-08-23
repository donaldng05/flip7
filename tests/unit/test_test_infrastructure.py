"""Tests for shared deterministic test infrastructure."""

import random


def test_seeded_rng_fixture_repeats_sequence(seeded_rng: random.Random) -> None:
    first = [seeded_rng.random() for _ in range(3)]
    repeat = random.Random(7)
    assert first == [repeat.random() for _ in range(3)]
