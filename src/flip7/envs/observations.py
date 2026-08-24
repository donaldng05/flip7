"""Public-information observation encodings for Flip 7 environments."""

from __future__ import annotations

from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from flip7.core.cards import (
    ActionCard,
    ActionCardName,
    ModifierCard,
    ModifierCardName,
    NumberCard,
    card_counts,
)
from flip7.core.state import GameState, PlayerState, PlayerStatus

NUMBER_FEATURE_COUNT = 13
SECOND_CHANCE_INDEX = 13
MODIFIER_FEATURE_OFFSET = 14
MODIFIER_FEATURE_COUNT = 6
STATUS_FEATURE_OFFSET = 20
STATUS_FEATURE_COUNT = 4
UNIQUE_COUNT_INDEX = 24
SCORE_INDEX = 25
PLAYER_FEATURE_SIZE = 26

PENDING_ACTION_COUNT = 4
ROUND_SCALE = 30.0
SCORE_SCALE = 200.0
FLIP7_SCALE = 7.0
FLIP_THREE_SCALE = 3.0
DECK_SIZE = 94
HISTOGRAM_SIZE = 22
DRAW_COUNT_OFFSET = 22
DISCARD_COUNT_OFFSET = 23
DECK_FEATURE_SIZE = HISTOGRAM_SIZE + 2

BASIC_SIZE = PLAYER_FEATURE_SIZE + 1 + PENDING_ACTION_COUNT + 1 + 1

_STATUS_ORDER = (
    PlayerStatus.ACTIVE,
    PlayerStatus.STAYED,
    PlayerStatus.FROZEN,
    PlayerStatus.BUSTED,
)
_MODIFIER_ORDER = tuple(ModifierCardName)
_ACTION_ORDER = tuple(ActionCardName)


class ObservationFamily(StrEnum):
    """Named public-information observation layouts."""

    BASIC = "basic"
    COMPETITIVE = "competitive"
    DECK_AWARE = "deck_aware"


def observation_size(player_count: int, family: ObservationFamily) -> int:
    """Return the observation vector length for a seated game and family."""
    if family is ObservationFamily.BASIC:
        return BASIC_SIZE
    competitive = PLAYER_FEATURE_SIZE * player_count + 4 * player_count + 6
    if family is ObservationFamily.COMPETITIVE:
        return competitive
    if family is ObservationFamily.DECK_AWARE:
        return competitive + DECK_FEATURE_SIZE
    msg = f"unsupported observation family: {family}"
    raise ValueError(msg)


def encode_observation(
    state: GameState,
    player_id: int,
    family: ObservationFamily,
) -> NDArray[np.float32]:
    """Encode one player's observation without draw-pile order."""
    if family is ObservationFamily.BASIC:
        return _encode_basic(state, player_id)
    if family is ObservationFamily.COMPETITIVE:
        return _encode_competitive(state, player_id)
    if family is ObservationFamily.DECK_AWARE:
        return _encode_deck_aware(state, player_id)
    msg = f"unsupported observation family: {family}"
    raise ValueError(msg)


def histogram_index(card: NumberCard | ActionCard | ModifierCard) -> int:
    """Return the stable remaining-count histogram index for a card face."""
    if isinstance(card, NumberCard):
        return card.value
    if isinstance(card, ActionCard):
        return NUMBER_FEATURE_COUNT + _ACTION_ORDER.index(card.name)
    return NUMBER_FEATURE_COUNT + len(_ACTION_ORDER) + _MODIFIER_ORDER.index(card.name)


def _encode_basic(state: GameState, player_id: int) -> NDArray[np.float32]:
    observation = np.zeros(BASIC_SIZE, dtype=np.float32)
    observation[:PLAYER_FEATURE_SIZE] = _player_features(state.player(player_id))
    observation[PLAYER_FEATURE_SIZE] = state.round_number / ROUND_SCALE
    pending_offset = PLAYER_FEATURE_SIZE + 1
    observation[pending_offset : pending_offset + PENDING_ACTION_COUNT] = (
        _pending_action_one_hot(state)
    )
    observation[pending_offset + PENDING_ACTION_COUNT] = float(
        state.dealer_id == player_id
    )
    observation[pending_offset + PENDING_ACTION_COUNT + 1] = _flip_three_remaining(
        state
    )
    return observation


def _encode_competitive(state: GameState, player_id: int) -> NDArray[np.float32]:
    player_count = len(state.players)
    observation = np.zeros(
        observation_size(player_count, ObservationFamily.COMPETITIVE),
        dtype=np.float32,
    )
    for seat, player in enumerate(state.players):
        start = seat * PLAYER_FEATURE_SIZE
        observation[start : start + PLAYER_FEATURE_SIZE] = _player_features(player)
    offset = PLAYER_FEATURE_SIZE * player_count
    observation[offset + player_id] = 1.0
    offset += player_count
    if state.current_player_id is not None:
        observation[offset + state.current_player_id] = 1.0
    offset += player_count
    observation[offset + state.dealer_id] = 1.0
    offset += player_count
    if state.pending_action is not None:
        observation[offset + state.pending_action.resolving_player_id] = 1.0
    offset += player_count
    observation[offset] = state.round_number / ROUND_SCALE
    offset += 1
    observation[offset : offset + PENDING_ACTION_COUNT] = _pending_action_one_hot(state)
    offset += PENDING_ACTION_COUNT
    observation[offset] = _flip_three_remaining(state)
    return observation


def _encode_deck_aware(state: GameState, player_id: int) -> NDArray[np.float32]:
    competitive = _encode_competitive(state, player_id)
    deck = np.zeros(DECK_FEATURE_SIZE, dtype=np.float32)
    for card, count in card_counts(state.draw_pile).items():
        deck[histogram_index(card)] = float(count)
    deck[DRAW_COUNT_OFFSET] = float(len(state.draw_pile))
    deck[DISCARD_COUNT_OFFSET] = float(len(state.discard_pile))
    return np.concatenate([competitive, deck])


def _player_features(player: PlayerState) -> NDArray[np.float32]:
    features = np.zeros(PLAYER_FEATURE_SIZE, dtype=np.float32)
    unique_numbers: set[int] = set()
    for card in player.cards:
        if isinstance(card, NumberCard):
            features[card.value] = 1.0
            unique_numbers.add(card.value)
        elif isinstance(card, ActionCard) and card.name is ActionCardName.SECOND_CHANCE:
            features[SECOND_CHANCE_INDEX] = 1.0
        elif isinstance(card, ModifierCard):
            features[MODIFIER_FEATURE_OFFSET + _MODIFIER_ORDER.index(card.name)] = 1.0
    features[STATUS_FEATURE_OFFSET + _STATUS_ORDER.index(player.status)] = 1.0
    features[UNIQUE_COUNT_INDEX] = len(unique_numbers) / FLIP7_SCALE
    features[SCORE_INDEX] = player.cumulative_score / SCORE_SCALE
    return features


def _pending_action_one_hot(state: GameState) -> NDArray[np.float32]:
    one_hot = np.zeros(PENDING_ACTION_COUNT, dtype=np.float32)
    if state.pending_action is None:
        one_hot[0] = 1.0
        return one_hot
    one_hot[1 + _ACTION_ORDER.index(state.pending_action.action_card)] = 1.0
    return one_hot


def _flip_three_remaining(state: GameState) -> float:
    if state.pending_flip_three is None:
        return 0.0
    return state.pending_flip_three.cards_remaining / FLIP_THREE_SCALE
