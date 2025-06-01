# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

import logging
from typing import Any, Dict, Optional, Union

import numpy as np
from pearl.api.action import Action
from pearl.api.action_result import ActionResult
from pearl.api.action_space import ActionSpace
from pearl.api.environment import Environment
from pearl.api.observation import Observation
from pearl.api.space import Space
from pearl.utils.instantiations.spaces.box import BoxSpace
from pearl.utils.instantiations.spaces.box_action import BoxActionSpace
from pearl.utils.instantiations.spaces.discrete import DiscreteSpace
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace
from torch import Tensor

try:
    import gymnasium as gym

    logging.info("Using 'gymnasium' package.")
except ModuleNotFoundError:
    import gym

    logging.warning("Using deprecated 'gym' package.")


def single_element_tensor_to_int(x: Tensor) -> int:
    return int(x)


def tensor_to_numpy(x: Tensor) -> np.ndarray:
    return x.numpy(force=True)


GYM_TO_PEARL_ACTION_SPACE = {
    "Discrete": DiscreteActionSpace,
    "Box": BoxActionSpace,
    # Add more here as needed
}
GYM_TO_PEARL_OBSERVATION_SPACE = {
    "Discrete": DiscreteSpace,
    "Box": BoxSpace,
    # Add more here as needed
}
PEARL_TO_GYM_ACTION = {
    "Discrete": single_element_tensor_to_int,
    "Box": tensor_to_numpy,
    # Add more here as needed
}


class GymEnvironment(Environment):
    """A wrapper for `gym.Env` to behave like Pearl's `Environment`."""

    def __init__(
        self, env: gym.Env
    ) -> None:
        """Constructs a `GymEnvironment` wrapper.

        Args:
            env: A gym.Env instance
        """
        self.env: gym.Env = env
        self._action_space: ActionSpace = _get_pearl_space(
            gym_space=self.env.action_space,
            gym_to_pearl_map=GYM_TO_PEARL_ACTION_SPACE,
        )
        self._observation_space: Space = _get_pearl_space(
            gym_space=self.env.observation_space,
            gym_to_pearl_map=GYM_TO_PEARL_OBSERVATION_SPACE,
        )

    @property
    def action_space(self) -> ActionSpace:
        """Returns the Pearl action space for this environment."""
        return self._action_space

    @property
    def observation_space(self) -> Space:
        return self._observation_space

    def reset(self, seed: Optional[int] = None) -> Observation:
        """Resets the environment and returns the initial observation and
        initial action space."""
        observation, info = self.env.reset(seed=seed)
        if observation.dtype == np.float64:
            observation = observation.astype(np.float32)
        return observation, info

    def step(self, action: Action) -> ActionResult:
        """Takes one step in the environment given the agent's action. Returns an
        `ActionResult` object containing the next observation, reward, and done flag."""
        # Convert action to the format expected by Gymnasium
        effective_action = _get_gym_action(
            pearl_action=action,
            gym_space=self.env.action_space,
        )
        # Take a step in the environment and receive an action result
        observation, reward, terminated, truncated, info = self.env.step(effective_action)

        if observation.dtype == np.float64:
            observation = observation.astype(np.float32)
        if isinstance(reward, np.float64):
            reward = reward.astype(np.float32)

        return observation, reward, terminated, truncated, info

    def render(self) -> None:
        self.env.render()

    def close(self) -> None:
        self.env.close()

    def __str__(self) -> str:
        if self.env.spec is not None:
            return self.env.spec.id
        else:
            return "CustomGymEnvironment"


def _get_gym_action(
    pearl_action: Action, gym_space: gym.Space
) -> Union[int, np.ndarray]:
    """A helper function to convert a Pearl `Action` to an action compatible with
    the Gym action space `gym_space`."""
    gym_space_name = gym_space.__class__.__name__
    try:
        pearl_to_gym_action_transform = PEARL_TO_GYM_ACTION[gym_space_name]
    except KeyError:
        raise NotImplementedError(
            f"The Gym space '{gym_space_name}' is not yet supported in Pearl."
        )
    return pearl_to_gym_action_transform(pearl_action)


def _get_pearl_space(
    gym_space: gym.Space, gym_to_pearl_map: Dict[str, Any]
) -> ActionSpace:
    """Returns the Pearl action space for this environment."""
    gym_space_name = gym_space.__class__.__name__
    try:
        pearl_action_space_cls = gym_to_pearl_map[gym_space_name]
    except KeyError:
        raise NotImplementedError(
            f"The Gym space '{gym_space_name}' is not yet supported in Pearl."
        )
    return pearl_action_space_cls.from_gym(gym_space)
