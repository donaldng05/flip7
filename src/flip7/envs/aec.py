"""PettingZoo AEC adapter for the Flip 7 rules engine."""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from typing import Any, cast

import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray
from pettingzoo.utils.env import AECEnv  # type: ignore[import-untyped]

from flip7.core.cards import Card
from flip7.core.engine import Flip7Engine
from flip7.core.state import InvalidPlayerCountError
from flip7.envs.encodings import (
    action_mask,
    action_space_size,
    agent_name,
    decode_action,
    player_id_from_agent,
)
from flip7.envs.observations import (
    ObservationFamily,
    encode_observation,
    observation_size,
)
from flip7.envs.rewards import RewardMode, rewards_from_result

type AgentID = str
type Observation = NDArray[np.float32]


class Flip7AECEnv(AECEnv[AgentID, Observation, int | None]):
    """Sequential multi-agent Flip 7 environment."""

    metadata = {
        "name": "flip7_v0",
        "render_modes": [],
        "is_parallelizable": False,
    }

    def __init__(
        self,
        player_count: int = 3,
        *,
        observation: ObservationFamily | str = ObservationFamily.COMPETITIVE,
        reward: RewardMode | str = RewardMode.SPARSE_WIN,
    ) -> None:
        super().__init__()
        if not 3 <= player_count <= 18:
            msg = "baseline Flip 7 supports 3 to 18 players"
            raise InvalidPlayerCountError(msg)

        self.player_count = player_count
        self.observation_family = ObservationFamily(observation)
        self.reward_mode = RewardMode(reward)
        self.possible_agents = [
            agent_name(player_id) for player_id in range(player_count)
        ]
        self.agents: list[AgentID] = []
        self._engine: Flip7Engine | None = None
        obs_size = observation_size(player_count, self.observation_family)
        self._observation_space = spaces.Box(
            low=0.0,
            high=np.finfo(np.float32).max,
            shape=(obs_size,),
            dtype=np.float32,
        )
        self._action_space: spaces.Space[int | None] = cast(
            spaces.Space[int | None],
            spaces.Discrete(action_space_size(player_count)),
        )
        self.observation_spaces = {
            agent: self._observation_space for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: self._action_space for agent in self.possible_agents
        }

    @property
    def engine(self) -> Flip7Engine:
        """Current rules engine, available after reset."""
        if self._engine is None:
            msg = "environment has not been reset"
            raise RuntimeError(msg)
        return self._engine

    def observation_space(self, agent: AgentID) -> spaces.Box:
        """Return the shared observation space for a seated agent."""
        del agent
        return self._observation_space

    def action_space(self, agent: AgentID) -> spaces.Space[int | None]:
        """Return the shared discrete action space for a seated agent."""
        del agent
        return self._action_space

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> None:
        """Start a new seeded game and select the first deciding agent."""
        options = options or {}
        self.agents = list(self.possible_agents)
        self._engine = Flip7Engine(
            random.Random(seed),
            self.player_count,
            dealer_id=int(options.get("dealer_id", 0)),
            draw_pile=_optional_cards(options.get("draw_pile")),
            discard_pile=_optional_cards(options.get("discard_pile")) or (),
            starting_scores=_optional_scores(options.get("starting_scores")),
        )
        self.rewards = dict.fromkeys(self.agents, 0.0)
        self._cumulative_rewards = dict.fromkeys(self.agents, 0.0)
        self.terminations = dict.fromkeys(self.agents, False)
        self.truncations = dict.fromkeys(self.agents, False)
        self.infos = {agent: self._info(agent) for agent in self.agents}
        self.agent_selection = self._current_agent()

    def step(self, action: int | None) -> None:
        """Apply the selected agent's action, or retire a terminated agent."""
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return
        if action is None:
            msg = "None is only valid after termination"
            raise ValueError(msg)

        agent = self.agent_selection
        player_id = player_id_from_agent(agent)
        decoded = decode_action(action, self.engine.state, player_id)
        result = self.engine.apply(decoded)

        self._cumulative_rewards[agent] = 0
        self._clear_rewards()
        assigned = rewards_from_result(result, self.player_count, self.reward_mode)
        for reward_player_id, reward in assigned.items():
            self.rewards[agent_name(reward_player_id)] = reward

        self.infos = {item: self._info(item) for item in self.agents}
        if result.is_game_terminal:
            self.terminations = dict.fromkeys(self.agents, True)
            self._accumulate_rewards()
            self._deads_step_first()
            return

        self.agent_selection = self._current_agent()
        self._accumulate_rewards()

    def observe(self, agent: AgentID) -> Observation:
        """Return the named observation for one seated agent."""
        return encode_observation(
            self.engine.state,
            player_id_from_agent(agent),
            self.observation_family,
        )

    def render(self) -> None:
        """Phase 3 does not define a human rendering mode."""
        return None

    def close(self) -> None:
        """Release adapter resources. The engine holds no external handles."""
        self._engine = None

    def _current_agent(self) -> AgentID:
        current_player_id = self.engine.state.current_player_id
        if current_player_id is None:
            msg = "engine settled without a deciding player"
            raise RuntimeError(msg)
        return agent_name(current_player_id)

    def _info(self, agent: AgentID) -> dict[str, NDArray[np.int8]]:
        if self.engine.state.is_game_terminal or agent != self._acting_agent():
            mask = np.zeros(action_space_size(self.player_count), dtype=np.int8)
        else:
            mask = action_mask(self.engine.legal_actions(), self.player_count)
        return {"action_mask": mask}

    def _acting_agent(self) -> AgentID | None:
        current_player_id = self.engine.state.current_player_id
        if current_player_id is None:
            return None
        return agent_name(current_player_id)


def _optional_cards(value: object) -> Iterable[Card] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        msg = "draw_pile and discard_pile must be iterables of cards"
        raise TypeError(msg)
    typed_cards = cast(Iterable[Card], value)
    return tuple(typed_cards)


def _optional_scores(value: object) -> Sequence[int] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        msg = "starting_scores must be a sequence of integers"
        raise TypeError(msg)
    typed_scores = cast(Sequence[int], value)
    return tuple(int(score) for score in typed_scores)
