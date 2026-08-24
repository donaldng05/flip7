"""Typed Flip 7 legal action models and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from flip7.core.cards import ActionCard, ActionCardName
from flip7.core.state import GameState, InvalidPhaseError, PlayerState, RoundPhase


class Flip7ActionError(ValueError):
    """Base class for invalid Flip 7 player actions."""


class InvalidActionError(Flip7ActionError):
    """Raised when a player attempts an illegal action."""


class InvalidTargetError(Flip7ActionError):
    """Raised when an Action card target is illegal."""


class ActionKind(StrEnum):
    """Legal action categories exposed by the core game model."""

    HIT = "hit"
    STAY = "stay"
    TARGET = "target"


@dataclass(frozen=True, slots=True)
class HitAction:
    """Voluntary request for the current player to receive one card."""

    player_id: int
    kind: ActionKind = field(default=ActionKind.HIT, init=False)


@dataclass(frozen=True, slots=True)
class StayAction:
    """Voluntary request for the current player to bank cards and leave the round."""

    player_id: int
    kind: ActionKind = field(default=ActionKind.STAY, init=False)


@dataclass(frozen=True, slots=True)
class TargetAction:
    """Assignment of a targetable Action card to an active player."""

    player_id: int
    action_card: ActionCardName
    target_player_id: int
    kind: ActionKind = field(default=ActionKind.TARGET, init=False)


type TurnAction = HitAction | StayAction
type LegalAction = TurnAction | TargetAction

_TARGETABLE_ACTION_CARDS = frozenset(
    {
        ActionCardName.FREEZE,
        ActionCardName.FLIP_THREE,
        ActionCardName.SECOND_CHANCE,
    }
)


def legal_turn_actions(state: GameState, player_id: int) -> tuple[TurnAction, ...]:
    """Return legal Hit/Stay choices for an offered normal turn."""
    player = _validate_turn_actor(state, player_id)

    actions: list[TurnAction] = [HitAction(player_id=player_id)]
    if player.has_cards:
        actions.append(StayAction(player_id=player_id))
    return tuple(actions)


def validate_turn_action(state: GameState, action: TurnAction) -> TurnAction:
    """Validate and return a Hit or Stay action for the current turn."""
    player = _validate_turn_actor(state, action.player_id)

    if isinstance(action, StayAction) and not player.has_cards:
        msg = "Stay requires at least one card"
        raise InvalidActionError(msg)

    return action


def legal_targets(
    state: GameState,
    player_id: int,
    action_card: ActionCardName,
) -> tuple[int, ...]:
    """Return legal active targets for a targetable Action card."""
    _validate_targeting_actor(state, player_id, action_card)
    if action_card is ActionCardName.SECOND_CHANCE:
        return tuple(
            player.player_id
            for player in state.players
            if player.is_active
            and player.player_id != player_id
            and not _has_second_chance(player)
        )
    return tuple(player.player_id for player in state.players if player.is_active)


def validate_target_action(state: GameState, action: TargetAction) -> TargetAction:
    """Validate and return a target assignment action."""
    _validate_targeting_actor(state, action.player_id, action.action_card)
    target = state.player(action.target_player_id)
    if not target.is_active:
        msg = "Action cards require an active target"
        raise InvalidTargetError(msg)
    if action.action_card is ActionCardName.SECOND_CHANCE:
        if action.target_player_id == action.player_id:
            msg = "cannot keep a second Second Chance"
            raise InvalidTargetError(msg)
        if _has_second_chance(target):
            msg = "target already has Second Chance"
            raise InvalidTargetError(msg)
    return action


def _validate_turn_actor(state: GameState, player_id: int) -> PlayerState:
    if state.round_phase is not RoundPhase.TURN:
        msg = "turn actions require turn phase"
        raise InvalidPhaseError(msg)
    if state.is_round_terminal:
        msg = "cannot act in a terminal round"
        raise InvalidActionError(msg)
    if state.current_player_id != player_id:
        msg = "turn action must be selected by the current player"
        raise InvalidActionError(msg)

    player = state.player(player_id)
    if not player.is_active:
        msg = "turn action requires an active player"
        raise InvalidActionError(msg)
    return player


def _validate_targeting_actor(
    state: GameState,
    player_id: int,
    action_card: ActionCardName,
) -> PlayerState:
    if action_card not in _TARGETABLE_ACTION_CARDS:
        msg = f"{action_card.value} does not target players"
        raise InvalidActionError(msg)
    if state.is_round_terminal:
        msg = "cannot target players in a terminal round"
        raise InvalidActionError(msg)

    player = state.player(player_id)
    if not player.is_active:
        msg = "target action requires an active player"
        raise InvalidActionError(msg)
    return player


def _has_second_chance(player: PlayerState) -> bool:
    return any(
        isinstance(card, ActionCard) and card.name is ActionCardName.SECOND_CHANCE
        for card in player.cards
    )
