"""Deterministic complete-round and complete-game integration tests."""

import random

from flip7.core.actions import LegalAction, StayAction, TargetAction
from flip7.core.cards import ActionCardName, Card, NumberCard
from flip7.core.engine import Flip7Engine
from flip7.core.state import GamePhase


def _choose_action(engine: Flip7Engine) -> LegalAction:
    legal = engine.legal_actions()
    stay = next((action for action in legal if isinstance(action, StayAction)), None)
    if stay is not None:
        return stay
    return legal[0]


def test_scripted_two_round_game_rotates_dealer_and_accumulates_scores() -> None:
    draw_pile: tuple[Card, ...] = (
        NumberCard(1),
        NumberCard(2),
        NumberCard(3),
        NumberCard(4),
        NumberCard(5),
        NumberCard(6),
        NumberCard(7),
    )
    engine = Flip7Engine(random.Random(0), 3, draw_pile=draw_pile)

    engine.apply(StayAction(player_id=1))
    engine.apply(StayAction(player_id=2))
    engine.apply(StayAction(player_id=0))

    assert engine.state.round_number == 2
    assert engine.state.dealer_id == 1
    assert [player.cumulative_score for player in engine.state.players] == [3, 1, 2]
    assert [player.cards for player in engine.state.players] == [
        (NumberCard(5),),
        (NumberCard(6),),
        (NumberCard(4),),
    ]

    engine.apply(StayAction(player_id=2))
    engine.apply(StayAction(player_id=0))
    engine.apply(StayAction(player_id=1))

    assert engine.state.round_number == 3
    assert engine.state.dealer_id == 2
    assert [player.cumulative_score for player in engine.state.players] == [8, 7, 6]


def test_complete_seeded_game_reaches_threshold_and_reproduces() -> None:
    first = Flip7Engine(random.Random(99), 3)
    second = Flip7Engine(random.Random(99), 3)
    action_count = 0

    while not first.state.is_game_terminal:
        action = _choose_action(first)
        first_result = first.apply(action)
        second_result = second.apply(action)
        action_count += 1
        assert first_result.state == second_result.state
        assert first.state == second.state

    assert action_count > 0
    assert first.state.game_phase is GamePhase.COMPLETE
    assert max(player.cumulative_score for player in first.state.players) >= 200
    assert first.state.winning_player_ids
    winning_score = max(player.cumulative_score for player in first.state.players)
    assert all(
        first.state.player(player_id).cumulative_score == winning_score
        for player_id in first.state.winning_player_ids
    )


def test_four_player_game_supports_action_targets_until_completion() -> None:
    engine = Flip7Engine(random.Random(5), 4)

    while not engine.state.is_game_terminal:
        legal = engine.legal_actions()
        assert legal
        action = legal[0]
        if (
            isinstance(action, TargetAction)
            and action.action_card is ActionCardName.FREEZE
        ):
            others = [
                item
                for item in legal
                if isinstance(item, TargetAction)
                and item.target_player_id != action.player_id
            ]
            action = others[0] if others else action
        engine.apply(action)

    assert engine.state.is_game_terminal is True
    assert engine.state.round_number >= 1
