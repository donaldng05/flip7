"""Integer action encodings and agent-id helpers for Flip 7 environments."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from flip7.core.actions import (
    HitAction,
    InvalidActionError,
    LegalAction,
    StayAction,
    TargetAction,
)
from flip7.core.state import GameState

HIT_INDEX = 0
STAY_INDEX = 1
TARGET_OFFSET = 2


def agent_name(player_id: int) -> str:
    """Return the PettingZoo agent id for a seated player."""
    return f"player_{player_id}"


def player_id_from_agent(agent: str) -> int:
    """Parse a PettingZoo agent id into a seated player id."""
    prefix, separator, suffix = agent.partition("_")
    if separator != "_" or prefix != "player" or not suffix.isdigit():
        msg = f"invalid agent id: {agent}"
        raise ValueError(msg)
    player_id = int(suffix)
    if player_id < 0 or str(player_id) != suffix:
        msg = f"invalid agent id: {agent}"
        raise ValueError(msg)
    return player_id


def action_space_size(player_count: int) -> int:
    """Return the discrete action-space size for a seated game."""
    return TARGET_OFFSET + player_count


def encode_action(action: LegalAction) -> int:
    """Map a core legal action to a discrete environment action index."""
    if isinstance(action, HitAction):
        return HIT_INDEX
    if isinstance(action, StayAction):
        return STAY_INDEX
    return TARGET_OFFSET + action.target_player_id


def decode_action(action: int, state: GameState, player_id: int) -> LegalAction:
    """Map a discrete environment action index to a core legal action."""
    size = action_space_size(len(state.players))
    if action < 0 or action >= size:
        msg = "action index is outside the discrete action space"
        raise InvalidActionError(msg)
    if action == HIT_INDEX:
        return HitAction(player_id=player_id)
    if action == STAY_INDEX:
        return StayAction(player_id=player_id)
    if state.pending_action is None:
        msg = "a hit or stay action is required"
        raise InvalidActionError(msg)
    return TargetAction(
        player_id=player_id,
        action_card=state.pending_action.action_card,
        target_player_id=action - TARGET_OFFSET,
    )


def action_mask(
    legal_actions: tuple[LegalAction, ...], player_count: int
) -> NDArray[np.int8]:
    """Return an int8 mask with 1s at legal discrete action indices."""
    mask = np.zeros(action_space_size(player_count), dtype=np.int8)
    for action in legal_actions:
        mask[encode_action(action)] = 1
    return mask
