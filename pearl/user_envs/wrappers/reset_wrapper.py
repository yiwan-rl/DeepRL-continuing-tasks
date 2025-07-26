# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# pyre-ignore-all-errors

try:
    import gymnasium as gym
except ModuleNotFoundError:
    print("gymnasium module not found.")
import random

import numpy as np
from gymnasium import spaces


class AgentTerminationWrapper(gym.Wrapper):
    r"""
    This wrapper is used for learning resetting in continuing tasks. This wrapper should be used only when AdditionalActionWrapper is used.
    The agent controls the last dimension of the action space (continuous control) or chooses an action (discrete control) to choose whether to reset or not. 
    A cost is incurred if the agent resets.
    Args:
        env: the environment
        reset_cost: the cost of resetting the environment
    """

    def __init__(self, env):
        super(AgentTerminationWrapper, self).__init__(env)

    def step(self, action):
        if isinstance(self.action_space, spaces.Box):
            termination_prob = (action[-1] - self.action_space.low[-1]) / (
                self.action_space.high[-1] - self.action_space.low[-1]
            )
            terminated = random.random() < termination_prob
        else:
            terminated = action == self.action_space.n - 1 # last action is termination

        if terminated:
            obs, reward, _, truncated, info = self.env.step(action)
            return obs, reward, True, truncated, info
        else:
            return self.env.step(action)
    

class IgnoreTerminationTruncationWrapper(gym.Wrapper):
    r"""
    This wrapper is used for ignoring termination or truncation signals.
    It is used for converting an episodic task to a continuing task.
    Args:
        env: the environment
    """

    def __init__(self, env):
        super(IgnoreTerminationTruncationWrapper, self).__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        return obs, reward, False, False, info


class AdditionalActionWrapper(gym.Wrapper):
    r"""
    This wrapper is used to add an additional action dimension/action.
    This additional action dim/action can be used to control resetting.
    Args:
        env: the environment
    """

    def __init__(self, env):
        super(AdditionalActionWrapper, self).__init__(env)
        if isinstance(self.action_space, spaces.Box):
            # Add a dimension
            self.augmented_low = self.action_space.low[-1]
            self.augmented_high = self.action_space.high[-1]
            low = np.append(self.action_space.low, self.augmented_low)
            high = np.append(self.action_space.high, self.augmented_high)
            self.action_space = spaces.Box(low=low, high=high)
        else:
            self.action_space = spaces.Discrete(n=self.action_space.n + 1)

    def step(self, action):
        if isinstance(self.action_space, spaces.Box):
            obs, reward, terminated, truncated, info = self.env.step(action[:-1])
        else:
            obs, reward, terminated, truncated, info = self.env.step(action % (self.action_space.n - 1)) # if last action was chosen, choose the first action.
        return obs, reward, terminated, truncated, info


class RandomTerminationWrapper(gym.Wrapper):
    r"""
    This wrapper is used to create an environment that randomly terminates episodes with a probability of termination_prob.
    Args:
        env: the environment
        termination_prob: the probability of terminating the episode
    """

    def __init__(self, env, termination_prob=0.0):
        super(RandomTerminationWrapper, self).__init__(env)
        assert termination_prob is not None
        self.termination_prob = termination_prob

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if random.random() < self.termination_prob:
            terminated = True
        return obs, reward, terminated, truncated, info


class ResetWrapper(gym.Wrapper):
    r"""
    This wrapper is used to create an environment that resets when receiving a termination or truncation signal.
    This is used for any continuing or episodic tasks that send termination or truncation signals.
    Args:
        env: the environment
    """

    def __init__(self, env, reset_cost=0.0):
        super(ResetWrapper, self).__init__(env)
        self.reset_cost = reset_cost

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            obs, _ = self.env.reset()
            reward = -self.reset_cost
            info["num_resets"] = 1
            info["reset_cost"] = self.reset_cost
        return obs, reward, terminated, truncated, info