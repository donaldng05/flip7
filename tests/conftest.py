"""Shared test configuration."""

import random

import pytest


@pytest.fixture
def seeded_rng() -> random.Random:
    """Provide an isolated deterministic RNG for simulation tests."""
    return random.Random(7)
