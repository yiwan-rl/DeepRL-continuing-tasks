# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

from typing import Optional, List

import torch

from pearl.policy_learners.exploration_modules.exploration_module import (
    ExplorationModule,
)
from pearl.policy_learners.sequential_decision_making.actor_critic_base import (
    ActorCriticBase,
)
from pearl.replay_buffers.transition import TransitionBatch
from pearl.utils.functional_utils.learning.reward_centering import MA_RC, RVI_RC, TD_RC
from pearl.utils.instantiations.spaces import VectorBoxSpace
from torch import nn
from torch import vmap
import torch.nn.functional as F
from torch.func import grad, functional_call


def actor_loss_fn(
        actor_network_instance: nn.Module, 
        actor_params, 
        actor_buffers,
        critic_network_instance: nn.Module, 
        critic_params, 
        critic_buffers,
        state,
        action_space_low,
        action_space_high,
        action_space_mask,
    ):
    # Compute scalar actor loss for ONE experiment
    # state: (batch_size, state_dim)
    # actor_params: parameters for one experiment

    action_batch = actor_network_instance.sample_action(
        x=state, 
        params=actor_params, 
        buffers=actor_buffers, 
        action_space_low=action_space_low, 
        action_space_high=action_space_high, 
        action_space_mask=action_space_mask
    )  # (batch_size, action_dim)

    q = functional_call(critic_network_instance, (critic_params, critic_buffers), (state, action_batch))  # (batch_size)

    loss = -q.mean()
    return loss


def critic_loss_fn(
    actor_network_instance: nn.Module,
    actor_target_params,
    actor_target_buffers,
    critic_network_instance: nn.Module,
    critic_params,
    critic_buffers,
    critic_target_params,
    critic_target_buffers,
    state, # (batch_size, state_dim)
    action, # (batch_size, action_dim)
    terminated, # (batch_size)
    reward, # (batch_size)
    next_state, # (batch_size, state_dim)
    discount_factor,
    action_space_low,
    action_space_high,
    action_space_mask,
):
    # Compute scalar critic loss for ONE experiment

    with torch.no_grad():
        next_action = actor_network_instance.sample_action(
            x=next_state, 
            params=actor_target_params, 
            buffers=actor_target_buffers, 
            action_space_low=action_space_low, 
            action_space_high=action_space_high, 
            action_space_mask=action_space_mask
        )  # (batch_size, action_dim)
        next_q = functional_call(critic_network_instance, (critic_target_params, critic_target_buffers), (next_state, next_action))  # (batch_size)

        expected_state_action_values = (
            next_q * discount_factor * (1 - terminated.float())
        ) + reward  # (batch_size)

    q = functional_call(critic_network_instance, (critic_params, critic_buffers), (state, action))  # (batch_size)

    loss = F.mse_loss(q, expected_state_action_values.detach())
    return loss


class DeepDeterministicPolicyGradient(ActorCriticBase):
    """
    A Class for Deep Deterministic Deep Policy Gradient policy learner.
    paper: https://arxiv.org/pdf/1509.02971.pdf
    """

    def __init__(
        self,
        action_space: VectorBoxSpace,
        actor_network_instances: List[nn.Module],
        critic_network_instances: List[nn.Module],
        actor_optimizer,
        critic_optimizer,
        exploration_module: ExplorationModule,
        actor_soft_update_tau: float = 0.005,
        critic_soft_update_tau: float = 0.005,
        discount_factor: float = 0.99,
        training_rounds: int = 1,
        batch_size: int = 256,
        reward_rate: torch.Tensor = torch.tensor(0.0),
        reward_centering: Optional[TD_RC | RVI_RC | MA_RC] = None,
    ) -> None:
        super(DeepDeterministicPolicyGradient, self).__init__(
            action_space=action_space,
            use_actor_target=True,
            use_critic_target=True,
            ensemble_critic_size=1,
            actor_soft_update_tau=actor_soft_update_tau,
            critic_soft_update_tau=critic_soft_update_tau,
            exploration_module=exploration_module,
            discount_factor=discount_factor,
            training_rounds=training_rounds,
            batch_size=batch_size,
            is_action_continuous=True,
            actor_network_instances=actor_network_instances,
            critic_network_instances=critic_network_instances,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            reward_rate=reward_rate,
            reward_centering=reward_centering,
        )

        # Wrap grad with vmap to compute gradients per experiment
        self._actor_grad_fn = vmap(
            grad(actor_loss_fn, argnums=1),
            in_dims=(
                None,  # actor_network_instance
                0,  # actor_params
                0,  # actor_buffers
                None,  # critic_network_instance
                0,  # critic_params
                0,  # critic_buffers
                1,  # state
                0,  # action_space_low
                0,  # action_space_high
                0  # action_space_mask
            ),
        )
        self._critic_grad_fn = vmap(
            grad(critic_loss_fn, argnums=4),
            in_dims=(
                None,  # actor_network_instance
                0,  # actor_target_params batched over experiments
                0,  # actor_target_buffers batched over experiments
                None,  # critic_network_instance
                0,  # critic_params batched over experiments
                0,  # critic_buffers batched over experiments
                0,  # critic_target_params batched over experiments
                0,  # critic_target_buffers batched over experiments
                1,  # state
                1,  # action
                1,  # terminated
                1,  # reward
                1,  # next_state
                None,  # discount_factor
                0,  # action_space_low
                0,  # action_space_high
                0,  # action_space_mask
            ),
        )

    def _get_actor_gradient(self, batch: TransitionBatch) -> torch.Tensor:
        # batch.state shape: (batch_size, num_exps, state_dim)
        # Make sure batch.state shape is (batch_size, num_exps, state_dim)
        grads = self._actor_grad_fn(
            self._actor,
            self._actor_params,
            self._actor_buffers,
            self._critic,
            self._critic_params,
            self._critic_buffers,
            batch.state,
            self._action_space.low,
            self._action_space.high,
            self._action_space.mask,
        )
        return torch.utils._pytree.tree_map(lambda g: g.detach(), grads)

    def _get_critic_gradient(self, batch: TransitionBatch) -> torch.Tensor:
        grads = self._critic_grad_fn(
            self._actor,
            self._actor_target_params,
            self._actor_target_buffers,
            self._critic,
            self._critic_params,
            self._critic_buffers,
            self._critic_target_params,
            self._critic_target_buffers,
            batch.state,
            batch.action,
            batch.terminated,
            batch.reward,
            batch.next_state,
            self._discount_factor,
            self._action_space.low,
            self._action_space.high,
            self._action_space.mask,
        )
        return torch.utils._pytree.tree_map(lambda g: g.detach(), grads)