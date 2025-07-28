# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

from typing import Any, Dict, Optional

import torch

from pearl.policy_learners.exploration_modules.exploration_module import (
    ExplorationModule,
)
from pearl.policy_learners.sequential_decision_making.actor_critic_base import (
    ActorCriticBase,
)
from pearl.replay_buffers.transition import TransitionBatch

from pearl.utils.functional_utils.learning.reward_centering import MA_RC, RVI_RC, TD_RC
from torch import nn
from pearl.utils.instantiations.spaces import VectorBoxSpace
from pearl.neural_networks.common.utils import update_target_params
import torchopt
from torch.func import grad, functional_call
import torch.nn.functional as F
from torch import vmap


# forward pass for one critic subnetwork
def critic_subnetwork_forward(params, buffers, subnetwork_module, state, action):
    return functional_call(subnetwork_module, (params, buffers), (state, action))


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

    # Critic ensemble forward pass
    q = vmap(critic_subnetwork_forward, in_dims=(0, 0, None, None, None))(
        critic_params, critic_buffers, critic_network_instance, state, action_batch
    )  # (ensemble_size, batch_size)

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
    state,
    action,
    terminated,
    reward,
    next_state,
    discount_factor,
    action_space_low,
    action_space_high,
    action_space_mask,
    actor_noise_mean,
    actor_noise_std,
    actor_noise_clip,
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
        
        # sample clipped gaussian noise
        noise = torch.normal(
            mean=actor_noise_mean,
            std=actor_noise_std,
        ) # shape (action_dim)

        noise = torch.clamp(
            noise,
            -actor_noise_clip,
            actor_noise_clip,
        )  # shape (action_dim)

        # rescale the noise
        noise = noise * (action_space_high - action_space_low) / 2 # shape (action_dim)

        # add clipped noise to next_action
        next_action = torch.clamp(
            next_action + noise, action_space_low, action_space_high
        )  # shape (batch_size, action_dim)

        next_q = vmap(critic_subnetwork_forward, in_dims=(0, 0, None, None, None))(
            critic_target_params, critic_target_buffers, critic_network_instance, next_state, next_action
        )  # shape (ensemble_size, batch_size)

        expected_state_action_values = (
            next_q * discount_factor * (1 - terminated.float())
        ) + reward  # shape (ensemble_size,batch_size)

    q = vmap(critic_subnetwork_forward, in_dims=(0, 0, None, None, None))(critic_params, critic_buffers, critic_network_instance, state, action)  # shape (ensemble_size, batch_size)
    loss = F.mse_loss(q, expected_state_action_values.detach())
    return loss


class TD3(ActorCriticBase):
    """
    TD3 uses a deterministic actor, Twin critics, and a delayed actor update.
        - An exploration module is used with deterministic actors.
        - To avoid exploration, use NoExploration module.
    """

    def __init__(
        self,
        action_space: VectorBoxSpace,
        actor_network_instances: nn.ModuleList,
        critic_network_instances: nn.ModuleList,
        actor_optimizer,
        critic_optimizer,
        exploration_module: ExplorationModule,
        actor_soft_update_tau: float = 0.005,
        critic_soft_update_tau: float = 0.005,
        discount_factor: float = 0.99,
        training_rounds: int = 1,
        batch_size: int = 256,
        actor_update_freq: int = 2,
        actor_noise_std: float | torch.Tensor = 0.2,
        actor_noise_clip: float = 0.5,
        ensemble_critic_size: int = 2,  # for TD3, ensemble_critic_size is 2
        reward_rate: torch.Tensor = torch.tensor(0.0),
        reward_centering: Optional[TD_RC | RVI_RC | MA_RC] = None,
    ) -> None:
        super(TD3, self).__init__(
            action_space=action_space,
            use_actor_target=True,
            use_critic_target=True,
            ensemble_critic_size=ensemble_critic_size,
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
        self._actor_update_freq = actor_update_freq
        self.register_buffer("_actor_noise_mean", torch.zeros(actor_noise_std.shape))
        self.register_buffer("_actor_noise_std", actor_noise_std)
        self._actor_noise_clip = actor_noise_clip

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
                None,  # actor_noise_mean
                None,  # actor_noise_std
                None,  # actor_noise_clip
            ),
            randomness='different',  # Each vmap call gets different random noise
        )

    def learn_batch(self, batch: TransitionBatch) -> Dict[str, Any]:

        if isinstance(self.reward_centering, TD_RC):
            if self.reward_centering.initialize_reward_rate:
                self.reward_rate.data.fill_(batch.reward.mean(-1))
                self.reward_centering.initialize_reward_rate = False

        # Compute per-experiment gradients

        # delayed actor update
        if self._training_steps % self._actor_update_freq == 0:
            actor_gradients = self._get_actor_gradient(batch)

            # Update actor parameters
            updates, self._actor_optimizer_state = self._actor_optimizer.update(
                actor_gradients, self._actor_optimizer_state
            )
            self._actor_params = torchopt.apply_updates(self._actor_params, updates)

        critic_gradients = self._get_critic_gradient(batch)
        # Update critic parameters
        updates, self._critic_optimizer_state = self._critic_optimizer.update(
            critic_gradients, self._critic_optimizer_state
        )
        self._critic_params = torchopt.apply_updates(self._critic_params, updates)

        if isinstance(self.reward_centering, TD_RC):
            self.reward_centering.optimizer.step()

        # Soft update targets
        if self._use_critic_target and self._training_steps % self._critic_target_update_freq == 0:
            update_target_params(
                self._critic_target_params,
                self._critic_params,
                self._critic_soft_update_tau,
            )
        if self._use_actor_target and self._training_steps % self._actor_target_update_freq == 0:
            update_target_params(
                self._actor_target_params,
                self._actor_params,
                self._actor_soft_update_tau,
            )

        return {}

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
            self._actor_noise_mean,
            self._actor_noise_std,
            self._actor_noise_clip,
        )
        return torch.utils._pytree.tree_map(lambda g: g.detach(), grads)
