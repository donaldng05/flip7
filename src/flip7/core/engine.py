"""Deterministic Flip 7 round and game orchestration."""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum

from flip7.core.actions import (
    HitAction,
    InvalidActionError,
    LegalAction,
    StayAction,
    TargetAction,
    TurnAction,
    legal_targets,
    legal_turn_actions,
    validate_target_action,
    validate_turn_action,
)
from flip7.core.cards import ActionCard, ActionCardName, Card, NumberCard
from flip7.core.deck import Deck, EmptyDeckError
from flip7.core.scoring import calculate_round_scores
from flip7.core.state import (
    GamePhase,
    GameState,
    InvalidPlayerCountError,
    PendingAction,
    PendingFlipThree,
    PlayerState,
    PlayerStatus,
    RoundPhase,
    create_initial_state,
)

GAME_END_THRESHOLD = 200
_SETTLE_LIMIT = 500


class EventKind(StrEnum):
    """Structured engine events produced by a state transition."""

    CARD_DEALT = "card_dealt"
    CARD_DRAWN = "card_drawn"
    STAYED = "stayed"
    BUSTED = "busted"
    SECOND_CHANCE_USED = "second_chance_used"
    SECOND_CHANCE_TRANSFERRED = "second_chance_transferred"
    SECOND_CHANCE_DISCARDED = "second_chance_discarded"
    FROZEN = "frozen"
    FLIP_THREE_STARTED = "flip_three_started"
    ACTION_DEFERRED = "action_deferred"
    ACTION_DISCARDED = "action_discarded"
    FLIP7 = "flip7"
    RESHUFFLED = "reshuffled"
    ROUND_SCORED = "round_scored"
    ROUND_ENDED = "round_ended"
    GAME_ENDED = "game_ended"


@dataclass(frozen=True, slots=True)
class GameEvent:
    """One inspectable rule event from a transition."""

    kind: EventKind
    player_id: int | None = None
    card: Card | None = None
    target_player_id: int | None = None
    score: int | None = None


@dataclass(frozen=True, slots=True)
class TransitionResult:
    """Result of applying one legal player action."""

    state: GameState
    events: tuple[GameEvent, ...]
    drawn_cards: tuple[Card, ...]
    score_changes: tuple[tuple[int, int], ...]
    is_round_terminal: bool
    is_game_terminal: bool


class Flip7Engine:
    """Seeded Flip 7 rules engine with structured transitions."""

    def __init__(
        self,
        rng: random.Random,
        player_count: int,
        *,
        dealer_id: int = 0,
        draw_pile: Iterable[Card] | None = None,
        discard_pile: Iterable[Card] = (),
        starting_scores: Sequence[int] | None = None,
    ) -> None:
        if not 3 <= player_count <= 18:
            msg = "baseline Flip 7 supports 3 to 18 players"
            raise InvalidPlayerCountError(msg)

        initial = create_initial_state(player_count, dealer_id=dealer_id)
        scores = (
            tuple(starting_scores)
            if starting_scores is not None
            else (0,) * player_count
        )
        if len(scores) != player_count:
            msg = "starting_scores must include every seated player"
            raise ValueError(msg)

        self._rng = rng
        self._deck = (
            Deck(rng, draw_pile=draw_pile, discard_pile=discard_pile)
            if draw_pile is not None
            else Deck.standard(rng)
        )
        self._players = [
            replace(player, cumulative_score=score)
            for player, score in zip(initial.players, scores, strict=True)
        ]
        self._dealer_id = dealer_id
        self._current_player_id: int | None = None
        self._game_phase = GamePhase.IN_PROGRESS
        self._round_phase = RoundPhase.SETUP
        self._round_number = 1
        self._pending_action: PendingAction | None = None
        self._pending_flip_three: PendingFlipThree | None = None
        self._flip7_player_id: int | None = None
        self._is_round_terminal = False
        self._is_game_terminal = False
        self._winning_player_ids: tuple[int, ...] = ()
        self._players_needing_initial_card: list[int] = []
        self._deferred_queue: list[tuple[int, ActionCard]] = []
        self._held_aside: list[Card] = []
        self._turn_player_id: int | None = None
        self._events: list[GameEvent] = []
        self._drawn: list[Card] = []
        self._score_changes: list[tuple[int, int]] = []

        self._begin_round()
        self._resolve_until_input()
        self._clear_transition_buffers()

    @property
    def state(self) -> GameState:
        """Current inspectable engine snapshot."""
        return self._snapshot()

    def legal_actions(self) -> tuple[LegalAction, ...]:
        """Return the legal actions for the current decision point."""
        if self._is_game_terminal:
            return ()
        if self._pending_action is not None:
            return tuple(
                TargetAction(
                    player_id=self._pending_action.resolving_player_id,
                    action_card=self._pending_action.action_card,
                    target_player_id=target_id,
                )
                for target_id in legal_targets(
                    self._snapshot(),
                    self._pending_action.resolving_player_id,
                    self._pending_action.action_card,
                )
            )
        if self._round_phase is RoundPhase.TURN and self._current_player_id is not None:
            return legal_turn_actions(self._snapshot(), self._current_player_id)
        return ()

    def apply(self, action: LegalAction) -> TransitionResult:
        """Apply one legal action and resolve automatic consequences."""
        if self._is_game_terminal:
            msg = "cannot act after the game has ended"
            raise InvalidActionError(msg)

        self._clear_transition_buffers()
        if self._pending_action is not None:
            if not isinstance(action, TargetAction):
                msg = "a target assignment is required"
                raise InvalidActionError(msg)
            self._apply_target(action)
        else:
            if not isinstance(action, HitAction | StayAction):
                msg = "a hit or stay action is required"
                raise InvalidActionError(msg)
            self._apply_turn(action)

        self._resolve_until_input()
        state = self._snapshot()
        return TransitionResult(
            state=state,
            events=tuple(self._events),
            drawn_cards=tuple(self._drawn),
            score_changes=tuple(self._score_changes),
            is_round_terminal=any(
                event.kind is EventKind.ROUND_ENDED for event in self._events
            )
            or state.is_game_terminal,
            is_game_terminal=state.is_game_terminal,
        )

    def _apply_turn(self, action: TurnAction) -> None:
        validate_turn_action(self._snapshot(), action)
        if isinstance(action, StayAction):
            self._replace_player(action.player_id, status=PlayerStatus.STAYED)
            self._events.append(
                GameEvent(kind=EventKind.STAYED, player_id=action.player_id)
            )
            self._current_player_id = self._next_active_after(action.player_id)
            if self._current_player_id is None:
                self._is_round_terminal = True
            return

        self._turn_player_id = action.player_id
        card = self._draw()
        self._events.append(
            GameEvent(kind=EventKind.CARD_DRAWN, player_id=action.player_id, card=card)
        )
        self._give_card(action.player_id, card)

    def _apply_target(self, action: TargetAction) -> None:
        pending = self._pending_action
        if pending is None:
            msg = "no action card is waiting for a target"
            raise InvalidActionError(msg)
        if (
            action.player_id != pending.resolving_player_id
            or action.action_card is not pending.action_card
        ):
            msg = "target action does not match the pending Action card"
            raise InvalidActionError(msg)
        validate_target_action(self._snapshot(), action)
        self._pending_action = None
        self._take_held(action.action_card)

        target = self._players[action.target_player_id]
        if not target.is_active:
            self._discard_cards([ActionCard(action.action_card)])
            self._events.append(
                GameEvent(
                    kind=EventKind.ACTION_DISCARDED,
                    player_id=action.player_id,
                    card=ActionCard(action.action_card),
                    target_player_id=action.target_player_id,
                )
            )
            return

        if action.action_card is ActionCardName.FREEZE:
            self._replace_player(action.target_player_id, status=PlayerStatus.FROZEN)
            self._discard_cards([ActionCard(ActionCardName.FREEZE)])
            self._events.append(
                GameEvent(
                    kind=EventKind.FROZEN,
                    player_id=action.target_player_id,
                )
            )
            return

        if action.action_card is ActionCardName.FLIP_THREE:
            self._held_aside.append(ActionCard(ActionCardName.FLIP_THREE))
            self._pending_flip_three = PendingFlipThree(
                resolving_player_id=action.player_id,
                target_player_id=action.target_player_id,
            )
            self._events.append(
                GameEvent(
                    kind=EventKind.FLIP_THREE_STARTED,
                    player_id=action.player_id,
                    card=ActionCard(ActionCardName.FLIP_THREE),
                    target_player_id=action.target_player_id,
                )
            )
            return

        self._add_card(
            action.target_player_id, ActionCard(ActionCardName.SECOND_CHANCE)
        )
        self._events.append(
            GameEvent(
                kind=EventKind.SECOND_CHANCE_TRANSFERRED,
                player_id=action.player_id,
                card=ActionCard(ActionCardName.SECOND_CHANCE),
                target_player_id=action.target_player_id,
            )
        )

    def _begin_round(self) -> None:
        self._players = [
            replace(player, status=PlayerStatus.ACTIVE, cards=())
            for player in self._players
        ]
        self._players_needing_initial_card = [
            player.player_id for player in self._players
        ]
        self._current_player_id = None
        self._round_phase = RoundPhase.INITIAL_DEAL
        self._pending_action = None
        self._pending_flip_three = None
        self._flip7_player_id = None
        self._is_round_terminal = False
        self._deferred_queue = []
        self._held_aside = []
        self._turn_player_id = None
        self._game_phase = GamePhase.IN_PROGRESS

    def _resolve_until_input(self) -> None:
        for _ in range(_SETTLE_LIMIT):
            if self._needs_input() or self._is_game_terminal:
                return
            self._auto_step()
        msg = "engine failed to settle on a decision point"
        raise RuntimeError(msg)

    def _needs_input(self) -> bool:
        if self._is_game_terminal:
            return True
        if self._pending_action is not None:
            return True
        if self._turn_player_id is not None:
            return False
        return (
            self._round_phase is RoundPhase.TURN
            and not self._is_round_terminal
            and self._current_player_id is not None
            and self._players[self._current_player_id].is_active
            and self._pending_flip_three is None
            and not self._deferred_queue
            and not self._players_needing_initial_card
        )

    def _auto_step(self) -> None:
        if self._is_round_terminal:
            self._finish_round()
            return
        if self._pending_action is not None:
            return
        if self._pending_flip_three is not None:
            if self._pending_flip_three.cards_remaining > 0:
                self._draw_flip_three_card()
                return
            if (
                self._flip7_player_id is None
                and self._players[self._pending_flip_three.target_player_id].is_active
            ):
                self._complete_flip_three_sequence()
            else:
                self._abort_flip_three()
            return
        if self._deferred_queue:
            self._promote_deferred()
            return
        if self._players_needing_initial_card:
            self._deal_next_initial()
            return
        if self._no_active_players() or self._flip7_player_id is not None:
            self._is_round_terminal = True
            return
        self._round_phase = RoundPhase.TURN
        if self._turn_player_id is not None:
            self._current_player_id = self._next_active_after(self._turn_player_id)
            self._turn_player_id = None
        elif (
            self._current_player_id is None
            or not self._players[self._current_player_id].is_active
        ):
            self._current_player_id = self._next_active_after(self._dealer_id)
        if self._current_player_id is None:
            self._is_round_terminal = True

    def _deal_next_initial(self) -> None:
        self._players_needing_initial_card = [
            player_id
            for player_id in self._players_needing_initial_card
            if self._players[player_id].is_active
        ]
        player_id = self._next_deal_player()
        if player_id is None:
            self._players_needing_initial_card = []
            return
        self._players_needing_initial_card = [
            item for item in self._players_needing_initial_card if item != player_id
        ]
        card = self._draw()
        self._events.append(
            GameEvent(kind=EventKind.CARD_DEALT, player_id=player_id, card=card)
        )
        self._round_phase = RoundPhase.INITIAL_DEAL
        self._give_card(player_id, card)

    def _draw_flip_three_card(self) -> None:
        pending = self._pending_flip_three
        if pending is None:
            return
        card = self._draw()
        self._events.append(
            GameEvent(
                kind=EventKind.CARD_DRAWN, player_id=pending.target_player_id, card=card
            )
        )
        remaining = pending.cards_remaining - 1
        target_id = pending.target_player_id
        self._pending_flip_three = replace(pending, cards_remaining=remaining)
        self._give_card(target_id, card, flip_three=True)
        if (
            remaining == 0
            and self._flip7_player_id is None
            and self._players[target_id].is_active
        ):
            self._complete_flip_three_sequence()

    def _complete_flip_three_sequence(self) -> None:
        pending = self._pending_flip_three
        self._pending_flip_three = None
        if pending is not None:
            deferred = [
                (pending.resolving_player_id, action)
                for action in pending.deferred_actions
            ]
            self._deferred_queue = deferred + self._deferred_queue
        self._discard_held(ActionCardName.FLIP_THREE)

    def _abort_flip_three(self) -> None:
        pending = self._pending_flip_three
        self._pending_flip_three = None
        if pending is not None:
            for card in pending.deferred_actions:
                self._discard_cards([card])
                self._events.append(
                    GameEvent(
                        kind=EventKind.ACTION_DISCARDED,
                        player_id=pending.target_player_id,
                        card=card,
                    )
                )
        self._discard_held(ActionCardName.FLIP_THREE)

    def _promote_deferred(self) -> None:
        resolver_id, card = self._deferred_queue.pop(0)
        if not self._players[resolver_id].is_active or self._no_active_players():
            self._discard_cards([card])
            self._events.append(
                GameEvent(
                    kind=EventKind.ACTION_DISCARDED,
                    player_id=resolver_id,
                    card=card,
                )
            )
            return
        self._held_aside.append(card)
        self._pending_action = PendingAction(
            action_card=card.name,
            resolving_player_id=resolver_id,
        )
        self._round_phase = RoundPhase.CARD_RESOLUTION
        self._current_player_id = resolver_id

    def _give_card(
        self, player_id: int, card: Card, *, flip_three: bool = False
    ) -> None:
        if isinstance(card, NumberCard):
            self._receive_number(player_id, card, flip_three=flip_three)
            return
        if isinstance(card, ActionCard):
            self._receive_action(player_id, card, flip_three=flip_three)
            return
        self._add_card(player_id, card)

    def _receive_number(
        self, player_id: int, card: NumberCard, *, flip_three: bool
    ) -> None:
        if self._has_number(player_id, card.value):
            if self._has_second_chance(player_id):
                self._remove_second_chance(player_id)
                self._discard_cards([card, ActionCard(ActionCardName.SECOND_CHANCE)])
                self._events.append(
                    GameEvent(
                        kind=EventKind.SECOND_CHANCE_USED,
                        player_id=player_id,
                        card=card,
                    )
                )
                return
            self._add_card(player_id, card)
            self._replace_player(player_id, status=PlayerStatus.BUSTED)
            self._events.append(
                GameEvent(kind=EventKind.BUSTED, player_id=player_id, card=card)
            )
            if flip_three:
                self._abort_flip_three()
            return

        self._add_card(player_id, card)
        if self._unique_number_count(player_id) >= 7:
            self._flip7_player_id = player_id
            self._is_round_terminal = True
            self._events.append(
                GameEvent(kind=EventKind.FLIP7, player_id=player_id, card=card)
            )
            if flip_three:
                self._abort_flip_three()

    def _receive_action(
        self, player_id: int, card: ActionCard, *, flip_three: bool
    ) -> None:
        if flip_three and card.name in {
            ActionCardName.FREEZE,
            ActionCardName.FLIP_THREE,
        }:
            pending = self._pending_flip_three
            if pending is not None:
                self._pending_flip_three = replace(
                    pending,
                    deferred_actions=(*pending.deferred_actions, card),
                )
            self._events.append(
                GameEvent(
                    kind=EventKind.ACTION_DEFERRED, player_id=player_id, card=card
                )
            )
            return

        if card.name is ActionCardName.SECOND_CHANCE:
            self._receive_second_chance(player_id, card)
            return

        self._held_aside.append(card)
        self._pending_action = PendingAction(
            action_card=card.name,
            resolving_player_id=player_id,
        )
        self._round_phase = RoundPhase.CARD_RESOLUTION
        self._current_player_id = player_id

    def _receive_second_chance(self, player_id: int, card: ActionCard) -> None:
        if not self._has_second_chance(player_id):
            self._add_card(player_id, card)
            return
        if not legal_targets(self._snapshot(), player_id, ActionCardName.SECOND_CHANCE):
            self._discard_cards([card])
            self._events.append(
                GameEvent(
                    kind=EventKind.SECOND_CHANCE_DISCARDED,
                    player_id=player_id,
                    card=card,
                )
            )
            return
        self._held_aside.append(card)
        self._pending_action = PendingAction(
            action_card=ActionCardName.SECOND_CHANCE,
            resolving_player_id=player_id,
        )
        self._round_phase = RoundPhase.CARD_RESOLUTION
        self._current_player_id = player_id

    def _finish_round(self) -> None:
        breakdowns = calculate_round_scores(self._snapshot())
        discarded: list[Card] = []
        updated: list[PlayerState] = []
        for player, breakdown in zip(self._players, breakdowns, strict=True):
            discarded.extend(player.cards)
            updated.append(
                replace(
                    player,
                    cards=(),
                    status=PlayerStatus.ACTIVE,
                    cumulative_score=player.cumulative_score + breakdown.round_score,
                )
            )
            self._score_changes.append((player.player_id, breakdown.round_score))
            self._events.append(
                GameEvent(
                    kind=EventKind.ROUND_SCORED,
                    player_id=player.player_id,
                    score=breakdown.round_score,
                )
            )
        self._players = updated
        discarded.extend(self._held_aside)
        self._held_aside = []
        self._discard_cards(discarded)
        self._pending_action = None
        self._pending_flip_three = None
        self._deferred_queue = []
        self._players_needing_initial_card = []
        self._turn_player_id = None

        highest = max(player.cumulative_score for player in self._players)
        if any(
            player.cumulative_score >= GAME_END_THRESHOLD for player in self._players
        ):
            self._winning_player_ids = tuple(
                player.player_id
                for player in self._players
                if player.cumulative_score == highest
            )
            self._is_game_terminal = True
            self._is_round_terminal = True
            self._game_phase = GamePhase.COMPLETE
            self._round_phase = RoundPhase.COMPLETE
            self._current_player_id = None
            self._events.append(GameEvent(kind=EventKind.GAME_ENDED))
            return

        self._events.append(GameEvent(kind=EventKind.ROUND_ENDED))
        self._dealer_id = (self._dealer_id + 1) % len(self._players)
        self._round_number += 1
        self._begin_round()

    def _draw(self) -> Card:
        if self._deck.draw_count == 0:
            if self._deck.discard_count == 0:
                raise EmptyDeckError("Cannot draw from an empty deck")
            self._events.append(GameEvent(kind=EventKind.RESHUFFLED))
        card = self._deck.draw()
        self._drawn.append(card)
        return card

    def _snapshot(self) -> GameState:
        return GameState(
            players=tuple(self._players),
            dealer_id=self._dealer_id,
            current_player_id=self._current_player_id,
            game_phase=self._game_phase,
            round_phase=self._round_phase,
            round_number=self._round_number,
            draw_pile=self._deck.draw_cards,
            discard_pile=self._deck.discard_cards,
            pending_action=self._pending_action,
            pending_flip_three=self._pending_flip_three,
            flip7_player_id=self._flip7_player_id,
            is_round_terminal=self._is_round_terminal,
            is_game_terminal=self._is_game_terminal,
            winning_player_ids=self._winning_player_ids,
            players_needing_initial_card=tuple(self._players_needing_initial_card),
        )

    def _next_active_after(self, player_id: int) -> int | None:
        count = len(self._players)
        for offset in range(1, count + 1):
            candidate = (player_id + offset) % count
            if self._players[candidate].is_active:
                return candidate
        return None

    def _next_deal_player(self) -> int | None:
        needing = set(self._players_needing_initial_card)
        if not needing:
            return None
        count = len(self._players)
        for offset in range(1, count + 1):
            candidate = (self._dealer_id + offset) % count
            if candidate in needing:
                return candidate
        return None

    def _no_active_players(self) -> bool:
        return all(not player.is_active for player in self._players)

    def _has_number(self, player_id: int, value: int) -> bool:
        return any(
            isinstance(card, NumberCard) and card.value == value
            for card in self._players[player_id].cards
        )

    def _has_second_chance(self, player_id: int) -> bool:
        return any(
            isinstance(card, ActionCard) and card.name is ActionCardName.SECOND_CHANCE
            for card in self._players[player_id].cards
        )

    def _unique_number_count(self, player_id: int) -> int:
        return len(
            {
                card.value
                for card in self._players[player_id].cards
                if isinstance(card, NumberCard)
            }
        )

    def _add_card(self, player_id: int, card: Card) -> None:
        player = self._players[player_id]
        self._replace_player(player_id, cards=(*player.cards, card))

    def _remove_second_chance(self, player_id: int) -> None:
        player = self._players[player_id]
        remaining = list(player.cards)
        for index, card in enumerate(remaining):
            if (
                isinstance(card, ActionCard)
                and card.name is ActionCardName.SECOND_CHANCE
            ):
                del remaining[index]
                break
        self._replace_player(player_id, cards=tuple(remaining))

    def _replace_player(self, player_id: int, **changes: object) -> None:
        self._players[player_id] = replace(self._players[player_id], **changes)

    def _discard_cards(self, cards: Iterable[Card]) -> None:
        self._deck.discard(cards)

    def _take_held(self, action_card: ActionCardName) -> Card:
        for index, card in enumerate(self._held_aside):
            if isinstance(card, ActionCard) and card.name is action_card:
                return self._held_aside.pop(index)
        return ActionCard(action_card)

    def _discard_held(self, action_card: ActionCardName) -> None:
        for index, card in enumerate(self._held_aside):
            if isinstance(card, ActionCard) and card.name is action_card:
                self._discard_cards([self._held_aside.pop(index)])
                return

    def _clear_transition_buffers(self) -> None:
        self._events = []
        self._drawn = []
        self._score_changes = []
