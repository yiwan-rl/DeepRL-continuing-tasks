# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

import typing
from typing import Any, Dict, Optional

import numpy as np

import torch
from pearl.api.action import Action
from pearl.api.observation import Observation

from pearl.policy_learners.policy_learner import PolicyLearner
from pearl.policy_learners.sequential_decision_making.ppo import (
    ProximalPolicyOptimization,
)
from pearl.replay_buffers.replay_buffer import ReplayBuffer
from pearl.replay_buffers.transition import TransitionBatch

from pearl.utils.device import get_pearl_device
from pearl.utils.functional_utils.learning.reward_centering import MA_RC


class PearlAgent:
    """
    A Agent gathering the most common aspects of production-ready agents.
    It is meant as a catch-all agent whose functionality is defined by flags
    (and possibly factories down the line)
    """

    # TODO: define a data structure that hosts the configs for a Pearl Agent
    def __init__(
        self,
        policy_learner: PolicyLearner,
        replay_buffer: ReplayBuffer,
        device_id: int = -1,
    ) -> None:
        """
        Initializes the PearlAgent.

        Args:
            policy_learner (PolicyLearner): An instance of PolicyLearner.
            replay_buffer (ReplayBuffer, optional): A replay buffer. Defaults to a single-transition
                replay buffer (note: this default is likely to change).
        """
        self.policy_learner: PolicyLearner = policy_learner
        self._device_id: int = device_id
        self.device: torch.device = get_pearl_device(device_id)

        self.replay_buffer: ReplayBuffer = replay_buffer
        # set here so replay_buffer and policy_learner are in sync
        self.replay_buffer._is_action_continuous = (
            self.policy_learner._is_action_continuous
        )
        self.policy_learner.device = self.device
        self.replay_buffer.device_for_batches = self.device

        self._latest_observation: Optional[Observation] = None
        self._latest_action: Optional[Action] = None
        self.policy_learner.to(self.device)
        if self.policy_learner.reward_rate.device != self.device:
            self.policy_learner.reward_rate = self.policy_learner.reward_rate.to(
                self.device
            )

    def act(self, exploit: bool = False) -> Action:
        action = self.policy_learner.act(
            self._latest_observation.to(self.device), exploit=exploit  # pyre-fixme[6]
        )
        self._latest_action = action.detach().cpu()

        if hasattr(self.policy_learner, "action_post_processing") and callable(
            self.policy_learner.action_post_processing
        ):
            return self.policy_learner.action_post_processing(action).detach().cpu().numpy()
        else:
            return self._latest_action.numpy()

    def observe(
        self,
        obs, reward, terminated, truncated, info,
    ) -> None:
        assert self._latest_action is not None
        if isinstance(self.policy_learner.reward_centering, MA_RC):
            ma_rate = self.policy_learner.reward_centering.ma_rate
            self.policy_learner.reward_rate = (
                self.policy_learner.reward_rate * ma_rate
                + reward * (1 - ma_rate)
            )

        assert self._latest_observation is not None
        assert self._latest_action is not None
        next_obs = torch.from_numpy(obs)
        self.replay_buffer.push(
            obs=self._latest_observation,
            action=self._latest_action,
            reward=torch.from_numpy(reward),
            terminated=torch.from_numpy(terminated),
            truncated=torch.from_numpy(truncated),
            next_obs=next_obs,
        )
        self._latest_observation = next_obs

    def learn(self) -> Dict[str, Any]:
        report = self.policy_learner.learn(self.replay_buffer)

        if isinstance(self.policy_learner, ProximalPolicyOptimization):
            self.replay_buffer.clear()

        return report

    def reset(
        self, observation: Observation
    ) -> None:
        self._latest_action = None
        self._latest_observation = torch.from_numpy(observation)

    def __str__(self) -> str:
        items = []
        items.append(self.policy_learner)
        items.append(self.replay_buffer)
        return "PearlAgent" + (
            " with " + ", ".join(str(item) for item in items) if items else ""
        )
