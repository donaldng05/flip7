"""Tests for Flip 7 observation families and information constraints."""

import numpy as np

from flip7.core.cards import (
    ActionCard,
    ActionCardName,
    ModifierCard,
    ModifierCardName,
    NumberCard,
)
from flip7.core.state import (
    GameState,
    PendingAction,
    PlayerState,
    PlayerStatus,
    RoundPhase,
)
from flip7.envs.observations import (
    BASIC_SIZE,
    ObservationFamily,
    encode_observation,
    histogram_index,
    observation_size,
)


def _three_player_state(
    draw_pile: tuple[NumberCard, ...],
    discard_pile: tuple[NumberCard, ...] = (),
) -> GameState:
    return GameState(
        players=(
            PlayerState(player_id=0, cards=(NumberCard(6),), cumulative_score=40),
            PlayerState(player_id=1, cards=(NumberCard(4),), cumulative_score=80),
            PlayerState(
                player_id=2,
                status=PlayerStatus.STAYED,
                cards=(NumberCard(8), ModifierCard(ModifierCardName.PLUS_4)),
            ),
        ),
        dealer_id=0,
        current_player_id=1,
        round_phase=RoundPhase.TURN,
        round_number=3,
        draw_pile=draw_pile,
        discard_pile=discard_pile,
    )


def test_observation_sizes_match_documented_layouts() -> None:
    assert observation_size(3, ObservationFamily.BASIC) == BASIC_SIZE
    assert observation_size(3, ObservationFamily.COMPETITIVE) == 96
    assert observation_size(3, ObservationFamily.DECK_AWARE) == 120
    assert observation_size(4, ObservationFamily.COMPETITIVE) == 126


def test_basic_observation_includes_only_the_observing_player_cards() -> None:
    state = _three_player_state((NumberCard(10), NumberCard(11)))
    observation = encode_observation(state, 1, ObservationFamily.BASIC)

    assert observation.shape == (BASIC_SIZE,)
    assert observation[4] == 1.0
    assert observation[6] == 0.0
    assert observation[8] == 0.0
    assert observation.dtype == np.float32


def test_competitive_observation_marks_ego_and_public_seats() -> None:
    state = _three_player_state((NumberCard(10), NumberCard(11)))
    observation = encode_observation(state, 1, ObservationFamily.COMPETITIVE)

    assert observation.shape == (96,)
    assert observation[4] == 0.0
    assert observation[26 + 4] == 1.0
    assert observation[78 + 1] == 1.0
    assert observation[81 + 1] == 1.0
    assert observation[84 + 0] == 1.0


def test_deck_aware_counts_are_order_independent_and_omit_draw_order() -> None:
    first = _three_player_state((NumberCard(10), NumberCard(11)), (NumberCard(1),))
    second = _three_player_state((NumberCard(11), NumberCard(10)), (NumberCard(1),))

    first_obs = encode_observation(first, 1, ObservationFamily.DECK_AWARE)
    second_obs = encode_observation(second, 1, ObservationFamily.DECK_AWARE)
    competitive = encode_observation(first, 1, ObservationFamily.COMPETITIVE)

    assert np.array_equal(first_obs, second_obs)
    assert first_obs[96 + histogram_index(NumberCard(10))] == 1.0
    assert first_obs[96 + histogram_index(NumberCard(11))] == 1.0
    assert first_obs[96 + 22] == 2.0
    assert first_obs[96 + 23] == 1.0
    np.testing.assert_array_equal(first_obs[:96], competitive)


def test_pending_action_and_second_chance_are_encoded() -> None:
    state = GameState(
        players=(
            PlayerState(
                player_id=0,
                cards=(ActionCard(ActionCardName.SECOND_CHANCE),),
            ),
            PlayerState(player_id=1, cards=(NumberCard(5),)),
            PlayerState(player_id=2, cards=(NumberCard(7),)),
        ),
        dealer_id=2,
        current_player_id=0,
        round_phase=RoundPhase.CARD_RESOLUTION,
        pending_action=PendingAction(
            action_card=ActionCardName.FREEZE,
            resolving_player_id=0,
        ),
    )
    observation = encode_observation(state, 0, ObservationFamily.BASIC)

    assert observation[13] == 1.0
    assert observation[27] == 0.0
    assert observation[28] == 1.0
    assert observation[31] == 0.0
