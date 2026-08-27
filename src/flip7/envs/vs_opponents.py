"""Gymnasium single-learner wrapper around the Flip 7 AEC environment."""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping
from typing import Any, cast

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from flip7.envs.aec import Flip7AECEnv
from flip7.envs.encodings import action_space_size, agent_name, player_id_from_agent
from flip7.envs.observations import ObservationFamily, encode_observation
from flip7.envs.rewards import RewardMode

type ActionMask = NDArray[np.int8]
type Observation = NDArray[np.float32]
type Opponent = Callable[[Observation, ActionMask], int]


class RandomLegalPolicy:
    """Select uniformly among currently legal discrete actions."""

    def __init__(self, rng: random.Random) -> None:
        self._rng = rng

    def __call__(self, observation: Observation, action_mask: ActionMask) -> int:
        del observation
        legal = np.flatnonzero(action_mask)
        if legal.size == 0:
            msg = "no legal actions remain for the random policy"
            raise RuntimeError(msg)
        return int(self._rng.choice(legal))


class Flip7VsOpponentsEnv(gym.Env[Observation, int]):
    """Gymnasium view of one learning seat against frozen opponents."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        player_count: int = 3,
        *,
        learner_id: int = 0,
        observation: ObservationFamily | str = ObservationFamily.COMPETITIVE,
        opponent_observation: ObservationFamily | str | None = None,
        opponent_observations: Mapping[str, ObservationFamily | str] | None = None,
        reward: RewardMode | str = RewardMode.SPARSE_WIN,
        opponents: Mapping[str, Opponent] | None = None,
    ) -> None:
        super().__init__()
        if not 0 <= learner_id < player_count:
            msg = "learner_id must reference a seated player"
            raise ValueError(msg)

        self.player_count = player_count
        self.learner_id = learner_id
        self.learner_agent = agent_name(learner_id)
        self._opponent_observation = ObservationFamily(
            observation if opponent_observation is None else opponent_observation
        )
        self._opponent_observations = (
            {
                agent: ObservationFamily(family)
                for agent, family in opponent_observations.items()
            }
            if opponent_observations is not None
            else None
        )
        if self._opponent_observations is not None:
            self._validate_opponent_observations(self._opponent_observations)
        self._supplied_opponents = dict(opponents) if opponents is not None else None
        self._opponents: dict[str, Opponent] = {}
        if self._supplied_opponents is not None:
            self._validate_opponents(self._supplied_opponents)

        self._aec = Flip7AECEnv(
            player_count,
            observation=observation,
            reward=reward,
        )
        self.observation_space = self._aec.observation_space(self.learner_agent)
        self.action_space = cast(
            spaces.Space[int],
            spaces.Discrete(action_space_size(player_count)),
        )

    @property
    def aec_env(self) -> Flip7AECEnv:
        """Underlying PettingZoo environment."""
        return self._aec

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Observation, dict[str, ActionMask]]:
        """Start a game and advance frozen opponents until the learner must act."""
        super().reset(seed=seed)
        self._aec.reset(seed=seed, options=options)
        self._opponents = self._policies_for_reset(seed)
        self._play_opponents()
        return self._observe_learner()

    def step(
        self, action: int
    ) -> tuple[Observation, float, bool, bool, dict[str, ActionMask]]:
        """Apply the learner action, then play opponents until the next decision."""
        self._aec.step(int(action))
        reward = float(self._aec.rewards.get(self.learner_agent, 0.0))
        reward += self._play_opponents()
        observation, info = self._observe_learner()
        terminated = self._aec.engine.state.is_game_terminal
        return observation, reward, terminated, False, info

    def close(self) -> None:
        """Close the underlying AEC environment."""
        self._aec.close()

    def _play_opponents(self) -> float:
        reward = 0.0
        while (
            not self._aec.engine.state.is_game_terminal
            and self._aec.agent_selection != self.learner_agent
        ):
            agent = self._aec.agent_selection
            _observation, _, _, _, info = self._aec.last()
            observation = encode_observation(
                self._aec.engine.state,
                player_id_from_agent(agent),
                self._observation_for_opponent(agent),
            )
            action = self._opponents[agent](observation, info["action_mask"])
            self._aec.step(action)
            reward += float(self._aec.rewards.get(self.learner_agent, 0.0))
        return reward

    def _observe_learner(self) -> tuple[Observation, dict[str, ActionMask]]:
        observation = self._aec.observe(self.learner_agent)
        if self._aec.engine.state.is_game_terminal:
            mask = np.zeros(action_space_size(self.player_count), dtype=np.int8)
        elif self._aec.agent_selection == self.learner_agent:
            mask = self._aec.infos[self.learner_agent]["action_mask"]
        else:
            mask = np.zeros(action_space_size(self.player_count), dtype=np.int8)
        return observation, {"action_mask": mask}

    def _policies_for_reset(self, seed: int | None) -> dict[str, Opponent]:
        if self._supplied_opponents is not None:
            return dict(self._supplied_opponents)
        rng = random.Random(None if seed is None else seed + 1_000_003)
        return {
            agent_name(player_id): RandomLegalPolicy(random.Random(rng.random()))
            for player_id in range(self.player_count)
            if player_id != self.learner_id
        }

    def _validate_opponents(self, opponents: Mapping[str, Opponent]) -> None:
        expected = {
            agent_name(player_id)
            for player_id in range(self.player_count)
            if player_id != self.learner_id
        }
        if set(opponents) != expected:
            msg = "opponents must include every seated player except the learner"
            raise ValueError(msg)

    def _observation_for_opponent(self, agent: str) -> ObservationFamily:
        if self._opponent_observations is None:
            return self._opponent_observation
        return self._opponent_observations[agent]

    def _validate_opponent_observations(
        self, opponent_observations: Mapping[str, ObservationFamily]
    ) -> None:
        expected = {
            agent_name(player_id)
            for player_id in range(self.player_count)
            if player_id != self.learner_id
        }
        if set(opponent_observations) != expected:
            msg = (
                "opponent_observations must include every seated player "
                "except the learner"
            )
            raise ValueError(msg)
