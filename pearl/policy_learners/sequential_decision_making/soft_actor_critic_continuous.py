# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

from typing import Any, Dict, Optional, Union

import torch
from pearl.policy_learners.exploration_modules.exploration_module import (
    ExplorationModule,
)
from pearl.policy_learners.sequential_decision_making.actor_critic_base import (
    ActorCriticBase,
)
from pearl.replay_buffers.transition import TransitionBatch
from pearl.utils.functional_utils.learning.reward_centering import MA_RC, RVI_RC, TD_RC
from torch import nn, optim
from pearl.utils.instantiations.spaces import VectorBoxSpace
from torch.func import functional_call, grad
from torch import vmap
import torch.nn.functional as F


def critic_subnetwork_forward(params, buffers, subnetwork_module, state, action):
    return functional_call(subnetwork_module, (params, buffers), (state, action))


def actor_loss_fn(
        action_batch,
        action_batch_log_prob,
        critic_network_instance: nn.Module, 
        critic_params, 
        critic_buffers,
        state,
        entropy_coef,
    ):
    # Critic ensemble forward pass
    qs = vmap(critic_subnetwork_forward, in_dims=(0, 0, None, None, None))(
        critic_params, critic_buffers, critic_network_instance, state, action_batch
    )  # (ensemble_size, batch_size)

    # clipped double q learning (reduce overestimation bias)
    q = torch.min(qs, dim=0).values  # shape: (batch_size)

    loss = (entropy_coef * action_batch_log_prob - q).mean()

    return loss



def critic_loss_fn(
    state, # (batch_size, state_dim)
    action, # (batch_size, action_dim)
    terminated, # (batch_size)
    reward, # (batch_size)
    next_state, # (batch_size, state_dim)
    next_action,
    next_action_log_prob,
    entropy_coef,
    critic_network_instance: nn.Module,
    critic_params,
    critic_buffers,
    critic_target_params,
    critic_target_buffers,
    discount_factor,
):

    next_qs = vmap(critic_subnetwork_forward, in_dims=(0, 0, None, None, None))(
        critic_target_params, critic_target_buffers, critic_network_instance, next_state, next_action
    )  # (ensemble_size, batch_size)

    # clipped double q-learning (reduce overestimation bias)
    next_q = torch.min(next_qs, dim=0).values  # shape: (batch_size)
    # add entropy regularization

    next_state_action_values = next_q - (entropy_coef * next_action_log_prob)  # shape: (batch_size x 1)

    expected_state_action_values = (
        next_state_action_values
        * discount_factor
        * (1 - terminated.float())
    ) + reward  # shape of expected_state_action_values: (batch_size)

    qs = vmap(critic_subnetwork_forward, in_dims=(0, 0, None, None, None))(
        critic_params, critic_buffers, critic_network_instance, state, action
    )  # (ensemble_size, batch_size)

    loss = F.mse_loss(qs, expected_state_action_values.detach().unsqueeze(0).expand(qs.shape[0], -1))
    return loss


class ContinuousSoftActorCritic(ActorCriticBase):
    """
    Soft Actor Critic Policy Learner.
    """

    def __init__(
        self,
        action_space: VectorBoxSpace,
        actor_network_instances: nn.ModuleList,
        critic_network_instances: nn.ModuleList,
        actor_optimizer: optim.Optimizer,
        critic_optimizer: optim.Optimizer,
        exploration_module: ExplorationModule,
        critic_soft_update_tau: float = 0.005,
        discount_factor: float = 0.99,
        training_rounds: int = 100,
        batch_size: int = 256,
        entropy_coef: float = 0.2,
        target_entropy_offset: float = 0.0,
        entropy_learning_rate: float = 1e-3,
        entropy_autotune: bool = True,
        ensemble_critic_size: int = 2,
        reward_rate: torch.Tensor = torch.tensor(0.0),
        reward_centering: Optional[TD_RC | RVI_RC | MA_RC] = None,
    ) -> None:
        super(ContinuousSoftActorCritic, self).__init__(
            action_space=action_space,
            use_actor_target=False,
            use_critic_target=True,
            actor_soft_update_tau=0.0,
            critic_soft_update_tau=critic_soft_update_tau,
            ensemble_critic_size=ensemble_critic_size,
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

        self._entropy_autotune = entropy_autotune
        if entropy_autotune:
            # initialize the entropy coefficient to 0
            self.register_parameter(
                "_log_entropy",
                torch.nn.Parameter(torch.zeros(action_space.num_elements(), requires_grad=True)),
            )
            self._entropy_optimizer: torch.optim.Optimizer = optim.Adam(
                [self._log_entropy], lr=entropy_learning_rate
            )
            self.register_buffer("_entropy_coef", torch.exp(self._log_entropy).detach())
            self.register_buffer(
                "_target_entropy",
                - self._action_space.actual_sizes
                + target_entropy_offset,
            )
        else:
            self.register_buffer("_entropy_coef", torch.ones(len(critic_network_instances)) * entropy_coef)
        
        # Wrap grad with vmap to compute gradients per experiment
        self._actor_grad_fn = vmap(
            grad(actor_loss_fn, argnums=(0, 1)),
            in_dims=(
                0,  # action_batch
                0,  # action_batch_log_prob
                None,  # critic_network_instance
                0,  # critic_params
                0,  # critic_buffers
                0,  # state
                0,  # entropy_coef
            ),
        )
        self._critic_grad_fn = vmap(
            grad(critic_loss_fn, argnums=9),
            in_dims=(
                0,  # state
                0,  # action
                0,  # terminated
                0,  # reward
                0,  # next_state
                0,  # next_action
                0,  # next_action_log_prob
                0,  # entropy_coef
                None,  # critic_network_instance
                0,  # critic_params
                0,  # critic_buffers
                0,  # critic_target_params
                0,  # critic_target_buffers
                None,  # discount_factor
            ),
            randomness='different',  # Each vmap call gets different random noise
        )
        

    def learn_batch(self, batch: TransitionBatch) -> Dict[str, Any]:
        # shape of batch: (num_exps, batch_size)

        state_and_next_state = torch.cat((batch.state, batch.next_state), dim=1)

        # sample both action and next action in one vmap call to save time
        action_and_next_action_batch, action_and_next_action_batch_log_prob = vmap(
            lambda x, params, buffers, low, high, mask, get_log_prob : self._actor.sample_action(x, params, buffers, low, high, mask, get_log_prob),
            in_dims=(0, 0, 0, 0, 0, 0, None),
            randomness="different"
        )(state_and_next_state, self._actor_params, self._actor_buffers, self._action_space.low, self._action_space.high, self._action_space.mask, torch.tensor(1, device=state_and_next_state.device))
        
        batch_size = batch.state.shape[1]

        self.sampled_action = action_and_next_action_batch[:, :batch_size]
        self.sampled_next_action = action_and_next_action_batch[:, batch_size:]
        self.sampled_action_log_prob = action_and_next_action_batch_log_prob[:, :batch_size]
        self.sampled_next_action_log_prob = action_and_next_action_batch_log_prob[:, batch_size:]

        actor_critic_loss = super().learn_batch(batch)

        if self._entropy_autotune:
            entropy_optimizer_loss = (
                -torch.exp(self._log_entropy.unsqueeze(-1))
                * (self.sampled_action_log_prob + self._target_entropy.unsqueeze(-1)).detach()
            ).mean()

            self._entropy_optimizer.zero_grad()
            entropy_optimizer_loss.backward()
            self._entropy_optimizer.step()

            self._entropy_coef = torch.exp(self._log_entropy).detach()
            {**actor_critic_loss, **{"entropy_coef": entropy_optimizer_loss}}

        return actor_critic_loss
    

    def _get_actor_gradient(self, batch: TransitionBatch) -> torch.Tensor:
        # batch.state shape: (batch_size, num_exps, state_dim)
        # Make sure batch.state shape is (batch_size, num_exps, state_dim)
        grads_wrt_action_batch_and_action_batch_log_prob = self._actor_grad_fn(
            self.sampled_action,
            self.sampled_action_log_prob,
            self._critic,
            self._critic_params,
            self._critic_buffers,
            batch.state,
            self._entropy_coef,
        )
        grads = torch.autograd.grad(
            outputs=(self.sampled_action, self.sampled_action_log_prob),
            inputs=self._actor_params.values(),  # list of tensors in w1
            grad_outputs=grads_wrt_action_batch_and_action_batch_log_prob,
            retain_graph=False
        )
        grads = {k: v for k, v in zip(self._actor_params.keys(), grads)}
        return torch.utils._pytree.tree_map(lambda g: g.detach(), grads)

    def _get_critic_gradient(self, batch: TransitionBatch) -> torch.Tensor:
        grads = self._critic_grad_fn(
            batch.state, # (batch_size, state_dim)
            batch.action, # (batch_size, action_dim)
            batch.terminated, # (batch_size)
            batch.reward, # (batch_size)
            batch.next_state, # (batch_size, state_dim)
            self.sampled_next_action,
            self.sampled_next_action_log_prob,
            self._entropy_coef,
            self._critic,
            self._critic_params,
            self._critic_buffers,
            self._critic_target_params,
            self._critic_target_buffers,
            self._discount_factor,
        )
        return torch.utils._pytree.tree_map(lambda g: g.detach(), grads)