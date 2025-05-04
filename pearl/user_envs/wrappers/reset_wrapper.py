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


class EpisodicTaskAddCostWrapper(gym.Wrapper):
    r"""A wrapper for episodic tasks. This wrapper adds a reset cost to the reward when the episode terminates.
    If discount factor is one, adding this cost shifts all returns by the same amount, and thus will not change the order of policies.
    Args:
        reset_cost: a number subtracted from the final reward.
    """

    def __init__(self, env, reset_cost):
        super(EpisodicTaskAddCostWrapper, self).__init__(env)
        assert reset_cost is not None
        self.reset_cost = reset_cost

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            reward -= self.reset_cost
            info["reset_cost"] = self.reset_cost
        return obs, reward, terminated, truncated, info


class AgentResetWrapper(gym.Wrapper):
    r"""
    This wrapper is used for learning resetting in continuing tasks. This wrapper should be used only when AdditionalActionWrapper is used.
    The agent controls the last dimension of the action space (continuous control) or chooses an action (discrete control) to choose whether to reset or not. 
    A cost is incurred if the agent resets.
    Args:
        env: the environment
        reset_cost: the cost of resetting the environment
    """

    def __init__(self, env, reset_cost):
        super(AgentResetWrapper, self).__init__(env)
        # check if AdditionalActionWrapper is used
        assert isinstance(self.env, AdditionalActionWrapper)
        assert reset_cost is not None
        self.reset_cost = reset_cost
        self.step_cnt = 0        

    def step(self, action):
        self.step_cnt += 1
        if isinstance(self.action_space, spaces.Box):
            resetting_prob = (action[-1] - self.action_space.low[-1]) / (
                self.action_space.high[-1] - self.action_space.low[-1]
            )
            resetting = random.random() < resetting_prob
        else:
            resetting = action == self.action_space.n - 1 # last action is reset
        if resetting:
            obs, _ = self.env.reset(seed=self.step_cnt)
            reward = -self.reset_cost
            info = {
                "reset_cost": self.reset_cost,
            }
            return obs, reward, False, False, info
        else:
            obs, reward, terminated, truncated, info = self.env.step(action)
            # environment should not decide to reset
            assert terminated is False
            assert truncated is False
            return obs, reward, terminated, truncated, info
    

class EpisodicToContinuingWrapper(gym.Wrapper):
    r"""
    This wrapper is used for converting an episodic task to a continuing task. 
    A cost is incurred when the environment resets.
    Args:
        env: the environment
        reset_cost: the cost of resetting the environment
    """

    def __init__(self, env, reset_cost):
        super(EpisodicToContinuingWrapper, self).__init__(env)
        assert reset_cost is not None
        self.reset_cost = reset_cost
        self.step_cnt = 0        

    def step(self, action):
        self.step_cnt += 1
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            # environment decides to reset
            obs, _ = self.env.reset(seed=self.step_cnt)
            reward = -self.reset_cost
            info = {
                "reset_cost": self.reset_cost,
            }

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
        self.step_cnt = 0
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


class RandomResetWrapper(gym.Wrapper):
    r"""
    This wrapper is used to create an environment that randomly resets with a probability of reset_prob.
    Args:
        env: the environment
        reset_prob: the probability of resetting the environment
    """

    def __init__(self, env, reset_prob):
        super(RandomResetWrapper, self).__init__(env)
        self.reset_prob = reset_prob

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if random.random() < self.reset_prob:
            terminated = True
        return obs, reward, terminated, truncated, info
