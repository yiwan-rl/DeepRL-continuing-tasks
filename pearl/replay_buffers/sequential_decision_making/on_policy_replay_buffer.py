# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict
from dataclasses import dataclass

from typing import Optional

import numpy as np

import torch

from pearl.api.observation import Observation
from pearl.api.action import Action
from pearl.api.reward import Reward
from pearl.replay_buffers.replay_buffer import ReplayBuffer
from pearl.replay_buffers.transition import Transition, TransitionBatch
from pearl.utils.device import get_default_device


@dataclass(frozen=False)
class OnPolicyTransition(Transition):
    gae: Optional[torch.Tensor] = None  # generalized advantage estimation
    lam_return: Optional[torch.Tensor] = None  # lambda return
    action_log_probs: Optional[torch.Tensor] = None  # action probs
    cum_reward: Optional[torch.Tensor] = None  # cumulative reward
    value: Optional[torch.Tensor] = None  # value


@dataclass(frozen=False)
class OnPolicyTransitionBatch(TransitionBatch):
    gae: Optional[torch.Tensor] = None  # generalized advantage estimation
    lam_return: Optional[torch.Tensor] = None  # lambda return
    action_log_probs: Optional[torch.Tensor] = None  # action probs
    cum_reward: Optional[torch.Tensor] = None  # cumulative reward
    value: Optional[torch.Tensor] = None  # value

    @classmethod
    def from_parent(
        cls,
        parent_obj: TransitionBatch,
        gae: Optional[torch.Tensor] = None,
        lam_return: Optional[torch.Tensor] = None,
        action_log_probs: Optional[torch.Tensor] = None,
        cum_reward: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
    ) -> "OnPolicyTransitionBatch":
        # Extract attributes from parent_obj using __dict__ and create a new Child object
        child_obj = cls(
            **parent_obj.__dict__,
            gae=gae,
            lam_return=lam_return,
            action_log_probs=action_log_probs,
            cum_reward=cum_reward,
            value=value,
        )
        return child_obj


class OnPolicyReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        capacity: int,
    ) -> None:
        super(ReplayBuffer, self).__init__()
        self.capacity = capacity
        self.observations: Optional[torch.Tensor] = None
        self.actions: Optional[torch.Tensor] = None
        self.rewards: Optional[torch.Tensor] = None
        self.terminateds: Optional[torch.Tensor] = None
        self.truncateds: Optional[torch.Tensor] = None
        self.next_observations: Optional[torch.Tensor] = None
        self.pos = 0
        self.full = False
        self._device_for_batches: torch.device = get_default_device()
        self._count: int = 0
        self._indices: Optional[np.ndarray] = None
    
    def add(
        self,
        obs: Observation,
        action: Action,
        reward: Reward,
        terminated: bool,
        truncated: bool,
        next_obs: Observation,
    ) -> None:
        if self.capacity == 0:
            return
        if self.observations is None:
            self.observations = torch.zeros((self.capacity,) + obs.shape, dtype=obs.dtype)
            # pyre-fixme
            self.actions = torch.zeros((self.capacity,) + action.shape, dtype=action.dtype)
            self.rewards = torch.zeros((self.capacity,) + reward.shape, dtype=reward.dtype)
            self.terminateds = torch.zeros((self.capacity,) + terminated.shape, dtype=terminated.dtype)
            self.truncateds = torch.zeros((self.capacity,) + truncated.shape, dtype=truncated.dtype)
            self.next_observations = torch.zeros((self.capacity,) + next_obs.shape, dtype=next_obs.dtype)
        # pyre-fixme
        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.terminateds[self.pos] = terminated
        self.truncateds[self.pos] = truncated
        self.next_observations[self.pos] = next_obs
        self.pos += 1
        if self.pos == self.capacity:
            self.pos = 0
            self.full = True

    @property
    def device_for_batches(self) -> torch.device:
        return self._device_for_batches

    @device_for_batches.setter
    def device_for_batches(self, new_device_for_batches: torch.device) -> None:
        self._device_for_batches = new_device_for_batches

    def push(
        self,
        obs: Observation,
        action: Action,
        reward: Reward,
        next_obs: Observation,
        terminated: bool,
        truncated: bool,
    ) -> None:

        self.add(
            obs=obs,
            action=action,
            reward=reward,
            next_obs=next_obs,
            terminated=terminated,
            truncated=truncated,
        )

    def init_indices(self) -> None:
        """
        Before sampling from the replay buffer,
        first randomly initialize the sequence of indices to be sampled.
        """
        self._count = 0
        self._indices = np.arange(self.pos)
        np.random.shuffle(self._indices)

    def sample(self, batch_size: int) -> TransitionBatch:
        """
        The OnPolicyReplayBuffer modifies the sample method from TensorBasedReplayBuffer.
        Instead of independently drawing random samples each time,
        it shuffles all transitions and samples the first batch_size data.
        On subsequent calls, it samples the next batch_size data until all transitions are sampled once.
        Then, it reshuffles and repeats the process, ensuring a random and even sampling of transitions.
        """
        if self._count == self.pos or self._indices is None:
            self.init_indices()
            self._count = 0
        assert self._indices is not None
        batch_inds = self._indices[self._count : self._count + batch_size]

        batch = OnPolicyTransitionBatch(
            # pyre-fixme
            state=self.observations[batch_inds, :].transpose(0, 1),  # (num_exps, batch_size, state_dim)
            action=self.actions[batch_inds, :].transpose(0, 1),
            reward=self.rewards[batch_inds].transpose(0, 1),
            terminated=self.terminateds[batch_inds].transpose(0, 1),
            truncated=self.truncateds[batch_inds].transpose(0, 1),
            next_state=self.next_observations[batch_inds, :].transpose(0, 1),
            # pyre-fixme[16]: `Optional` has no attribute `__setitem__`.
            action_log_probs=self.action_log_probs[:, batch_inds],
            gae=self.gae[:, batch_inds],
            lam_return=self.lam_return[:, batch_inds],
            value=self.value[:, batch_inds],
        ).to(self.device_for_batches)
        self._count += batch_size

        return batch

    def sample_all(self) -> TransitionBatch:
        """
        The OnPolicyReplayBuffer modifies the sample method from TensorBasedReplayBuffer.
        Instead of independently drawing random samples each time,
        it shuffles all transitions and samples the first batch_size data.
        On subsequent calls, it samples the next batch_size data until all transitions are sampled once.
        Then, it reshuffles and repeats the process, ensuring a random and even sampling of transitions.
        """

        batch = TransitionBatch(
            # pyre-fixme
            state=self.observations[: self.pos, :].transpose(0, 1),  # (num_exps, batch_size, state_dim)
            action=self.actions[: self.pos, :].transpose(0, 1),
            reward=self.rewards[: self.pos].transpose(0, 1),
            terminated=self.terminateds[: self.pos].transpose(0, 1),
            truncated=self.truncateds[: self.pos].transpose(0, 1),
            next_state=self.next_observations[: self.pos, :].transpose(0, 1),
        ).to(self.device_for_batches)

        return batch

    def __len__(self) -> int:
        if self.full is True:
            return self.capacity
        else:
            return self.pos

    def clear(self) -> None:
        self.observations = None
        self.pos = 0
        self.full = False
