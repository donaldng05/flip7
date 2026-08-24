"""Tests for Flip 7 round and game orchestration."""

import random

import pytest

from flip7.core.actions import (
    HitAction,
    InvalidActionError,
    StayAction,
    TargetAction,
)
from flip7.core.cards import (
    ActionCard,
    ActionCardName,
    Card,
    ModifierCard,
    ModifierCardName,
    NumberCard,
)
from flip7.core.engine import EmptyDeckError, EventKind, Flip7Engine
from flip7.core.state import (
    GamePhase,
    InvalidPlayerCountError,
    PlayerStatus,
    RoundPhase,
)


def _pile(*faces: int | ActionCardName | ModifierCardName | Card) -> tuple[Card, ...]:
    cards: list[Card] = []
    for face in faces:
        if isinstance(face, int):
            cards.append(NumberCard(face))
        elif isinstance(face, ActionCardName):
            cards.append(ActionCard(face))
        elif isinstance(face, ModifierCardName):
            cards.append(ModifierCard(face))
        else:
            cards.append(face)
    return tuple(cards)


def make_engine(
    draw_pile: tuple[Card, ...],
    *,
    player_count: int = 3,
    dealer_id: int = 0,
    starting_scores: tuple[int, ...] | None = None,
    discard_pile: tuple[Card, ...] = (),
    rng: random.Random | None = None,
) -> Flip7Engine:
    return Flip7Engine(
        rng or random.Random(0),
        player_count,
        dealer_id=dealer_id,
        draw_pile=draw_pile,
        discard_pile=discard_pile,
        starting_scores=starting_scores,
    )


def test_engine_rejects_out_of_scope_player_counts() -> None:
    with pytest.raises(InvalidPlayerCountError):
        make_engine(_pile(1, 2, 3), player_count=2)


def test_initial_deal_starts_left_of_dealer_and_waits_for_first_turn() -> None:
    engine = make_engine(_pile(4, 8, 6, 10))

    assert [player.cards for player in engine.state.players] == [
        (NumberCard(6),),
        (NumberCard(4),),
        (NumberCard(8),),
    ]
    assert engine.state.round_phase is RoundPhase.TURN
    assert engine.state.current_player_id == 1
    assert engine.legal_actions() == (
        HitAction(player_id=1),
        StayAction(player_id=1),
    )


def test_hit_and_stay_end_the_round_when_no_active_players_remain() -> None:
    engine = make_engine(_pile(4, 8, 6, ModifierCardName.PLUS_4))

    result = engine.apply(HitAction(player_id=1))
    assert result.drawn_cards == (ModifierCard(ModifierCardName.PLUS_4),)
    assert engine.state.player(1).cards == (
        NumberCard(4),
        ModifierCard(ModifierCardName.PLUS_4),
    )
    assert engine.state.current_player_id == 2

    engine.apply(StayAction(player_id=2))
    engine.apply(StayAction(player_id=0))
    final = engine.apply(StayAction(player_id=1))

    assert engine.state.player(1).cumulative_score == 8
    assert engine.state.player(2).cumulative_score == 8
    assert engine.state.player(0).cumulative_score == 6
    assert engine.state.dealer_id == 1
    assert engine.state.round_number == 2
    assert EventKind.ROUND_SCORED in {event.kind for event in final.events}


def test_duplicate_number_busts_and_scores_zero() -> None:
    engine = make_engine(_pile(5, 8, 6, 5))

    engine.apply(HitAction(player_id=1))

    assert engine.state.player(1).status is PlayerStatus.BUSTED
    assert engine.state.current_player_id == 2

    engine.apply(StayAction(player_id=2))
    engine.apply(StayAction(player_id=0))

    assert engine.state.player(1).cumulative_score == 0
    assert engine.state.player(2).cumulative_score == 8


def test_second_chance_discards_duplicate_and_does_not_bust() -> None:
    engine = make_engine(_pile(ActionCardName.SECOND_CHANCE, 8, 6, 5, 5))

    engine.apply(HitAction(player_id=1))
    engine.apply(StayAction(player_id=2))
    engine.apply(StayAction(player_id=0))
    result = engine.apply(HitAction(player_id=1))

    assert engine.state.player(1).status is PlayerStatus.ACTIVE
    assert engine.state.player(1).cards == (NumberCard(5),)
    assert EventKind.SECOND_CHANCE_USED in {event.kind for event in result.events}


def test_second_chance_transfer_requires_another_active_player() -> None:
    engine = make_engine(
        _pile(ActionCardName.SECOND_CHANCE, 2, 3, ActionCardName.SECOND_CHANCE)
    )

    engine.apply(HitAction(player_id=1))
    actions = engine.legal_actions()
    assert actions == (
        TargetAction(
            player_id=1,
            action_card=ActionCardName.SECOND_CHANCE,
            target_player_id=0,
        ),
        TargetAction(
            player_id=1,
            action_card=ActionCardName.SECOND_CHANCE,
            target_player_id=2,
        ),
    )

    engine.apply(actions[1])

    assert ActionCard(ActionCardName.SECOND_CHANCE) in engine.state.player(1).cards
    assert ActionCard(ActionCardName.SECOND_CHANCE) in engine.state.player(2).cards
    assert engine.state.current_player_id == 2


def test_freeze_makes_target_inactive_and_keeps_cards_for_scoring() -> None:
    engine = make_engine(_pile(4, 5, 6, ActionCardName.FREEZE))

    actions = engine.legal_actions()
    assert actions[0] == HitAction(player_id=1)
    engine.apply(HitAction(player_id=1))

    freeze = TargetAction(
        player_id=1,
        action_card=ActionCardName.FREEZE,
        target_player_id=2,
    )
    engine.apply(freeze)

    assert engine.state.player(2).status is PlayerStatus.FROZEN
    assert engine.state.player(2).cards == (NumberCard(5),)
    assert engine.state.current_player_id == 0

    engine.apply(StayAction(player_id=0))
    engine.apply(StayAction(player_id=1))

    assert engine.state.player(2).cumulative_score == 5


def test_initial_deal_freeze_skips_target_without_an_initial_card() -> None:
    engine = make_engine(_pile(ActionCardName.FREEZE, 9, 11))

    assert engine.legal_actions()[0].kind.value == "target"
    engine.apply(
        TargetAction(
            player_id=1,
            action_card=ActionCardName.FREEZE,
            target_player_id=2,
        )
    )

    assert engine.state.player(1).cards == ()
    assert engine.state.player(2).status is PlayerStatus.FROZEN
    assert engine.state.player(2).cards == ()
    assert engine.state.player(0).cards == (NumberCard(9),)
    assert engine.state.round_phase is RoundPhase.TURN
    assert engine.state.current_player_id == 1
    assert engine.legal_actions() == (HitAction(player_id=1),)


def test_flip_three_defers_freeze_until_sequence_completes() -> None:
    engine = make_engine(
        _pile(4, 5, 6, ActionCardName.FLIP_THREE, 7, ActionCardName.FREEZE, 8)
    )

    engine.apply(HitAction(player_id=1))
    engine.apply(
        TargetAction(
            player_id=1,
            action_card=ActionCardName.FLIP_THREE,
            target_player_id=2,
        )
    )

    assert engine.state.player(2).cards == (NumberCard(5), NumberCard(7), NumberCard(8))
    assert engine.legal_actions() == (
        TargetAction(
            player_id=1,
            action_card=ActionCardName.FREEZE,
            target_player_id=0,
        ),
        TargetAction(
            player_id=1,
            action_card=ActionCardName.FREEZE,
            target_player_id=1,
        ),
        TargetAction(
            player_id=1,
            action_card=ActionCardName.FREEZE,
            target_player_id=2,
        ),
    )

    engine.apply(
        TargetAction(
            player_id=1,
            action_card=ActionCardName.FREEZE,
            target_player_id=0,
        )
    )

    assert engine.state.player(0).status is PlayerStatus.FROZEN
    assert engine.state.current_player_id == 2


def test_flip_three_completes_when_final_card_is_deferred_freeze() -> None:
    engine = make_engine(
        _pile(4, 5, 6, ActionCardName.FLIP_THREE, 7, 8, ActionCardName.FREEZE)
    )

    engine.apply(HitAction(player_id=1))
    engine.apply(
        TargetAction(
            player_id=1,
            action_card=ActionCardName.FLIP_THREE,
            target_player_id=2,
        )
    )

    assert engine.state.player(2).cards == (NumberCard(5), NumberCard(7), NumberCard(8))
    assert {
        action.action_card
        for action in engine.legal_actions()
        if isinstance(action, TargetAction)
    } == {ActionCardName.FREEZE}


def test_flip_three_discards_deferred_action_when_target_busts() -> None:
    engine = make_engine(
        _pile(4, 5, 6, ActionCardName.FLIP_THREE, ActionCardName.FREEZE, 5)
    )

    engine.apply(HitAction(player_id=1))
    result = engine.apply(
        TargetAction(
            player_id=1,
            action_card=ActionCardName.FLIP_THREE,
            target_player_id=2,
        )
    )

    assert engine.state.player(2).status is PlayerStatus.BUSTED
    assert engine.state.player(0).status is PlayerStatus.ACTIVE
    assert all(event.kind is not EventKind.FROZEN for event in result.events)
    assert engine.state.pending_action is None
    assert engine.state.current_player_id == 0


def test_flip7_ends_round_immediately_and_awards_bonus() -> None:
    engine = make_engine(_pile(1, 2, 3, 4, 5, 6, 7, 8, 9))

    engine.apply(HitAction(player_id=1))
    engine.apply(StayAction(player_id=2))
    engine.apply(StayAction(player_id=0))
    engine.apply(HitAction(player_id=1))
    engine.apply(HitAction(player_id=1))
    engine.apply(HitAction(player_id=1))
    engine.apply(HitAction(player_id=1))
    result = engine.apply(HitAction(player_id=1))

    assert engine.state.player(1).cumulative_score == 55
    assert engine.state.player(2).cumulative_score == 2
    assert engine.state.player(0).cumulative_score == 3
    assert EventKind.FLIP7 in {event.kind for event in result.events}


def test_game_ends_after_round_when_threshold_is_reached() -> None:
    engine = make_engine(_pile(1, 2, 12), starting_scores=(199, 0, 0))

    engine.apply(StayAction(player_id=1))
    engine.apply(StayAction(player_id=2))
    result = engine.apply(StayAction(player_id=0))

    assert engine.state.is_game_terminal is True
    assert engine.state.game_phase is GamePhase.COMPLETE
    assert engine.state.winning_player_ids == (0,)
    assert engine.state.player(0).cumulative_score == 211
    assert EventKind.GAME_ENDED in {event.kind for event in result.events}
    assert engine.legal_actions() == ()


def test_tied_winners_are_reported_without_a_tie_breaker() -> None:
    engine = make_engine(_pile(10, 1, 10), starting_scores=(190, 190, 0))

    engine.apply(StayAction(player_id=1))
    engine.apply(StayAction(player_id=2))
    engine.apply(StayAction(player_id=0))

    assert engine.state.is_game_terminal is True
    assert engine.state.winning_player_ids == (0, 1)


def test_illegal_action_does_not_advance_the_game() -> None:
    engine = make_engine(_pile(4, 5, 6))
    before = engine.state

    with pytest.raises(InvalidActionError, match="current player"):
        engine.apply(StayAction(player_id=2))

    assert engine.state == before


def test_mid_round_reshuffle_leaves_cards_in_front_of_players() -> None:
    engine = make_engine(
        _pile(4, 5, 6),
        discard_pile=_pile(7, 8, 9, 10),
    )

    result = engine.apply(HitAction(player_id=1))

    assert EventKind.RESHUFFLED in {event.kind for event in result.events}
    assert engine.state.player(0).cards == (NumberCard(6),)
    assert engine.state.player(2).cards == (NumberCard(5),)
    assert len(engine.state.player(1).cards) == 2
    assert NumberCard(4) in engine.state.player(1).cards


def test_empty_draw_and_discard_piles_raise_a_domain_error() -> None:
    with pytest.raises(EmptyDeckError):
        make_engine(_pile())


def test_identical_seeds_and_actions_reproduce_the_same_state() -> None:
    first = Flip7Engine(random.Random(21), 3)
    second = Flip7Engine(random.Random(21), 3)

    while not first.state.is_game_terminal:
        legal = first.legal_actions()
        action = next(
            (item for item in legal if isinstance(item, StayAction)), legal[0]
        )
        first.apply(action)
        second.apply(action)

    assert first.state == second.state
    assert first.state.is_game_terminal is True
