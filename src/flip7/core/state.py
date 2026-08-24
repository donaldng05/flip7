"""Typed Flip 7 game-state models for the pure rules engine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from flip7.core.cards import ActionCard, ActionCardName, Card


class Flip7StateError(ValueError):
    """Base class for invalid Flip 7 state model data."""


class InvalidPlayerCountError(Flip7StateError):
    """Raised when a baseline game has an unsupported player count."""


class InvalidPlayerError(Flip7StateError):
    """Raised when state references a player outside the seated game."""


class InvalidPhaseError(Flip7StateError):
    """Raised when an action or state transition is used in the wrong phase."""


class GamePhase(StrEnum):
    """Coarse lifecycle for a complete Flip 7 game."""

    SETUP = "setup"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"


class RoundPhase(StrEnum):
    """Rule-level phases within a Flip 7 round."""

    SETUP = "setup"
    INITIAL_DEAL = "initial_deal"
    TURN = "turn"
    CARD_RESOLUTION = "card_resolution"
    SCORING = "scoring"
    CLEANUP = "cleanup"
    COMPLETE = "complete"


class PlayerStatus(StrEnum):
    """A player's current participation status in a round."""

    ACTIVE = "active"
    STAYED = "stayed"
    FROZEN = "frozen"
    BUSTED = "busted"


@dataclass(frozen=True, slots=True)
class PlayerState:
    """Visible round cards, status, and cumulative score for one seated player."""

    player_id: int
    status: PlayerStatus = PlayerStatus.ACTIVE
    cards: tuple[Card, ...] = ()
    cumulative_score: int = 0

    def __post_init__(self) -> None:
        if self.player_id < 0:
            msg = "player_id must be non-negative"
            raise InvalidPlayerError(msg)
        if self.cumulative_score < 0:
            msg = "cumulative_score must be non-negative"
            raise ValueError(msg)

    @property
    def is_active(self) -> bool:
        """Whether the player may currently receive cards or act."""
        return self.status is PlayerStatus.ACTIVE

    @property
    def has_cards(self) -> bool:
        """Whether the player has any round cards in front of them."""
        return bool(self.cards)


@dataclass(frozen=True, slots=True)
class PendingAction:
    """Targeting information for an Action card awaiting resolution."""

    action_card: ActionCardName
    resolving_player_id: int
    target_player_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        _validate_non_negative_player_id(
            self.resolving_player_id, "resolving_player_id"
        )
        for target_player_id in self.target_player_ids:
            _validate_non_negative_player_id(target_player_id, "target_player_ids")


@dataclass(frozen=True, slots=True)
class PendingFlipThree:
    """Progress for a Flip Three sequence and deferred action cards."""

    resolving_player_id: int
    target_player_id: int
    cards_remaining: int = 3
    deferred_actions: tuple[ActionCard, ...] = ()

    def __post_init__(self) -> None:
        _validate_non_negative_player_id(
            self.resolving_player_id, "resolving_player_id"
        )
        _validate_non_negative_player_id(self.target_player_id, "target_player_id")
        if not 0 <= self.cards_remaining <= 3:
            msg = "cards_remaining must be between 0 and 3"
            raise ValueError(msg)
        for action in self.deferred_actions:
            if action.name not in {
                ActionCardName.FREEZE,
                ActionCardName.FLIP_THREE,
            }:
                msg = "deferred_actions may only contain Freeze or Flip Three cards"
                raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class GameState:
    """Complete public engine state needed by future pure transitions."""

    players: tuple[PlayerState, ...]
    dealer_id: int
    current_player_id: int | None = None
    game_phase: GamePhase = GamePhase.IN_PROGRESS
    round_phase: RoundPhase = RoundPhase.SETUP
    round_number: int = 1
    draw_pile: tuple[Card, ...] = ()
    discard_pile: tuple[Card, ...] = ()
    pending_action: PendingAction | None = None
    pending_flip_three: PendingFlipThree | None = None
    flip7_player_id: int | None = None
    is_round_terminal: bool = False
    is_game_terminal: bool = False
    winning_player_ids: tuple[int, ...] = ()
    players_needing_initial_card: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        _validate_baseline_player_count(len(self.players))
        player_ids = tuple(player.player_id for player in self.players)
        expected_player_ids = tuple(range(len(self.players)))
        if player_ids != expected_player_ids:
            msg = "players must be seated with contiguous ids starting at 0"
            raise InvalidPlayerError(msg)

        self._validate_player_reference(self.dealer_id, "dealer_id")
        self._validate_optional_player_reference(
            self.current_player_id,
            "current_player_id",
        )
        self._validate_optional_player_reference(
            self.flip7_player_id, "flip7_player_id"
        )
        for winning_player_id in self.winning_player_ids:
            self._validate_player_reference(winning_player_id, "winning_player_ids")
        for player_id in self.players_needing_initial_card:
            self._validate_player_reference(player_id, "players_needing_initial_card")

        if self.round_number < 1:
            msg = "round_number must be at least 1"
            raise ValueError(msg)

        if self.pending_action is not None:
            self._validate_player_reference(
                self.pending_action.resolving_player_id,
                "pending_action.resolving_player_id",
            )
            for target_player_id in self.pending_action.target_player_ids:
                self._validate_player_reference(
                    target_player_id,
                    "pending_action.target_player_ids",
                )

        if self.pending_flip_three is not None:
            self._validate_player_reference(
                self.pending_flip_three.resolving_player_id,
                "pending_flip_three.resolving_player_id",
            )
            self._validate_player_reference(
                self.pending_flip_three.target_player_id,
                "pending_flip_three.target_player_id",
            )

    def player(self, player_id: int) -> PlayerState:
        """Return a player by id or raise a domain error."""
        self._validate_player_reference(player_id, "player_id")
        return self.players[player_id]

    def _validate_optional_player_reference(
        self,
        player_id: int | None,
        field_name: str,
    ) -> None:
        if player_id is None:
            return
        self._validate_player_reference(player_id, field_name)

    def _validate_player_reference(self, player_id: int, field_name: str) -> None:
        if not 0 <= player_id < len(self.players):
            msg = f"{field_name} must reference a seated player"
            raise InvalidPlayerError(msg)


def create_initial_state(player_count: int, *, dealer_id: int = 0) -> GameState:
    """Create a baseline game state with seated active players and no round cards."""
    _validate_baseline_player_count(player_count)
    players = tuple(
        PlayerState(player_id=player_id) for player_id in range(player_count)
    )
    return GameState(players=players, dealer_id=dealer_id)


def _validate_baseline_player_count(player_count: int) -> None:
    if not 3 <= player_count <= 18:
        msg = "baseline Flip 7 supports 3 to 18 players"
        raise InvalidPlayerCountError(msg)


def _validate_non_negative_player_id(player_id: int, field_name: str) -> None:
    if player_id < 0:
        msg = f"{field_name} must be non-negative"
        raise InvalidPlayerError(msg)
