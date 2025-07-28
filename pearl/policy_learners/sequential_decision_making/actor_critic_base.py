# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

import copy
from abc import abstractmethod
from typing import Any, Dict, Optional
import torchopt
import torch

from pearl.api.action import Action

from pearl.utils.instantiations.spaces import VectorDiscreteSpace, VectorBoxSpace
from pearl.api.state import SubjectiveState
from pearl.neural_networks.common.utils import update_target_params



from pearl.policy_learners.exploration_modules.exploration_module import (
    ExplorationModule,
)
from pearl.policy_learners.policy_learner import PolicyLearner
from pearl.replay_buffers.transition import TransitionBatch
from pearl.utils.functional_utils.learning.reward_centering import MA_RC, RVI_RC, TD_RC
from pearl.utils.instantiations.spaces.discrete import VectorDiscreteSpace
from pearl.utils.instantiations.spaces.box import VectorBoxSpace
from torch import nn
from torch import vmap
from torch.func import stack_module_state

class ActorCriticBase(PolicyLearner):
    """
    A base class for all actor-critic based policy learners.

    Many components that are common to all actor-critic methods have been put in this base class.
    These include:

    - actor and critic network initializations (optionally with corresponding target networks).
    - `act`, `reset` and `learn_batch` methods.
    - Utility functions used by many actor-critic methods.
    """

    def __init__(
        self,
        action_space: VectorDiscreteSpace | VectorBoxSpace,
        actor_network_instances: nn.ModuleList,
        critic_network_instances: nn.ModuleList,
        actor_optimizer,
        critic_optimizer,
        exploration_module: ExplorationModule,
        use_actor_target: bool = False,
        use_critic_target: bool = False,
        actor_soft_update_tau: float = 0.005,
        critic_soft_update_tau: float = 0.005,
        actor_target_update_freq: int = 1,
        critic_target_update_freq: int = 1,
        ensemble_critic_size: int = 1,  # number of critics, used only for EnsembleQValueNetwork
        discount_factor: float = 0.99,
        training_rounds: int = 1,
        batch_size: int = 256,
        is_action_continuous: bool = False,
        reward_rate: torch.Tensor = torch.tensor(0.0),
        reward_centering: Optional[TD_RC | RVI_RC | MA_RC] = None,
    ) -> None:
        super(ActorCriticBase, self).__init__(
            action_space=action_space,
            is_action_continuous=is_action_continuous,
            training_rounds=training_rounds,
            batch_size=batch_size,
            exploration_module=exploration_module,
            reward_rate=reward_rate,
            reward_centering=reward_centering,
        )
        """
        Constructs a base actor-critic policy learner.
        """

        # Reshape batched params and buffers from (80, ...) to (10, 8, ...)
        def reshape_batched(batched):
            return batched.reshape(len(actor_network_instances), ensemble_critic_size, *batched.shape[1:])

        self._use_actor_target = use_actor_target
        self._use_critic_target = use_critic_target

        self._actor: nn.Module = actor_network_instances[0]
        # Stack parameters and buffers into batched state
        self._actor_params, self._actor_buffers = stack_module_state(actor_network_instances)  # NamedTuple(params, buffers)
        self._actor_optimizer = actor_optimizer
        self._actor_optimizer_state = self._actor_optimizer.init(self._actor_params)
        self._actor_target_update_freq = actor_target_update_freq
        self._actor_soft_update_tau = actor_soft_update_tau

        # make a copy of the actor network to be used as the actor target network
        if self._use_actor_target:
            self._actor_target_params = torch.utils._pytree.tree_map(lambda p: p.clone().detach().requires_grad_(False), self._actor_params)
            self._actor_target_buffers = torch.utils._pytree.tree_map(lambda p: p.clone().detach().requires_grad_(False), self._actor_buffers)

        self._critic_target_update_freq = critic_target_update_freq
        self._critic_soft_update_tau = critic_soft_update_tau
        self._critic: nn.Module = critic_network_instances[0]

        # Stack parameters and buffers into batched state
        self._critic_params, self._critic_buffers = stack_module_state(critic_network_instances)  # NamedTuple(params, buffers)

        if ensemble_critic_size > 1:
            # reshape every critic parameter and buffer to (num_exps, ensemble_critic_size, ...)
            self._critic_params = {k: reshape_batched(v) for k, v in self._critic_params.items()}
            self._critic_buffers = {k: reshape_batched(v) for k, v in self._critic_buffers.items()}

        self._critic_optimizer = critic_optimizer
        self._critic_optimizer_state = self._critic_optimizer.init(self._critic_params)
        if self._use_critic_target:
            self._critic_target_params = torch.utils._pytree.tree_map(lambda p: p.clone().detach().requires_grad_(False), self._critic_params)
            self._critic_target_buffers = torch.utils._pytree.tree_map(lambda p: p.clone().detach().requires_grad_(False), self._critic_buffers)

        self._discount_factor = discount_factor
        self._current_steps = 0
        self._test_time = False

    def act(
        self,
        subjective_state: SubjectiveState,
        exploit: bool = False,
    ) -> Action:
        """
        Determines an action based on the policy network and optionally the exploration module.
        This function can operate in two modes: exploit or explore. The mode is determined by the
        `exploit` parameter.

        - If `exploit` is True, the function returns an action determined solely by the policy
        network.
        - If `exploit` is False, the function first calculates an `exploit_action` using the policy
        network. This action is then passed to the exploration module, along with additional
        arguments specific to the exploration module in use. The exploration module then generates
        an action that strikes a balance between exploration and exploitation.

        Args:
            subjective_state (SubjectiveState): Subjective state of the agent.
            exploit (bool, optional): Determines the mode of operation. If True, the function
            operates in exploit mode. If False, it operates in explore mode. Defaults to False.
        Returns:
            Action: An action (decision made by the agent in the given subjective state)
            that balances between exploration and exploitation, depending on the mode
            specified by the user. The returned action is from the available action space.
        """
        # Step 1: compute exploit_action
        # (action computed by actor network; and without any exploration)
        if self._test_time is False:
            self._current_steps += 1
        with torch.no_grad():
            if self._is_action_continuous:
                exploit_action = vmap(
                    lambda x, params, buffers, low, high, mask : self._actor.sample_action(x, params, buffers, low, high, mask)
                )(subjective_state, self._actor_params, self._actor_buffers, self._action_space.low, self._action_space.high, self._action_space.mask)
                action_probabilities = None
            else:
                action_probabilities = self._actor.get_policy_distribution(
                    state_batch=subjective_state,
                )
                # (num_exps x num_actions)
                exploit_action_index = torch.argmax(action_probabilities, dim=-1)
                exploit_action = self._actor.action_space.actions[exploit_action_index]

        # Step 2: return exploit action if no exploration,
        # else pass through the exploration module
        if exploit:
            return exploit_action

        # TODO: carefully check if safe action space is integrated with the exploration module
        return self._exploration_module.act(
            exploit_action=exploit_action,
            action_space=self._action_space,
            subjective_state=subjective_state,
            values=action_probabilities,
        )


    def learn_batch(self, batch: TransitionBatch) -> Dict[str, Any]:
        if isinstance(self.reward_centering, TD_RC):
            if self.reward_centering.initialize_reward_rate:
                self.reward_rate.data.fill_(batch.reward.mean(-1))
                self.reward_centering.initialize_reward_rate = False

        # Compute per-experiment gradients
        actor_gradients = self._get_actor_gradient(batch)
        critic_gradients = self._get_critic_gradient(batch)

        # Update actor parameters
        updates, self._actor_optimizer_state = self._actor_optimizer.update(
            actor_gradients, self._actor_optimizer_state
        )
        self._actor_params = torchopt.apply_updates(self._actor_params, updates)

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


    @abstractmethod
    def _get_actor_gradient(self, batch: TransitionBatch) -> torch.Tensor:
        """
        Abstract method for implementing the algorithm-specific logic for updating the actor
        network. This method must be implemented by any concrete subclass to provide the specific
        logic for updating the actor network based on the algorithm implemented by the subclass.
        Args:
            batch (TransitionBatch): A batch of transitions used for updating the actor network.
        Returns:
            loss (Tensor): The actor loss.
        """
        pass

    @abstractmethod
    def _get_critic_gradient(self, batch: TransitionBatch) -> torch.Tensor:
        """
        Abstract method for implementing the algorithm-specific logic for updating the critic
        network. This method must be implemented by any concrete subclass to provide the specific
        logic for updating the critic network based on the algorithm implemented by the subclass.
        Args:
            batch (TransitionBatch): A batch of transitions used for updating the actor network.
        Returns:
            loss (Tensor): The critic loss.
        """
        pass

    def save_model(self, path: str) -> None:
        torch.save(self._critic, path + "_critic")
        torch.save(self._actor, path + "_actor")

    def load_model(self, path: str) -> None:
        self._critic = torch.load(
            path + "_critic", map_location="cpu", weights_only=False
        ).to(self.device)
        self._actor = torch.load(
            path + "_actor", map_location="cpu", weights_only=False
        ).to(self.device)

    def compute_f_value(self, batch: TransitionBatch) -> torch.Tensor:
        """
        Computes the f value for a batch of transitions.
        Args:
            batch (TransitionBatch): A batch of transitions.
        Returns:
            f_value (Tensor): The value function for the batch of transitions.
        """
        qs = self._critic_target.get_q_values(
            batch.state, batch.action, get_all_values=True
        )
        return torch.mean(qs).detach()
    
    def to(self, device: torch.device) -> None:
        super().to(device)
        self._actor_params = torch.utils._pytree.tree_map(lambda p: p.to(device), self._actor_params)
        self._actor_buffers = torch.utils._pytree.tree_map(lambda p: p.to(device), self._actor_buffers)
        self._critic_params = torch.utils._pytree.tree_map(lambda p: p.to(device), self._critic_params)
        self._critic_buffers = torch.utils._pytree.tree_map(lambda p: p.to(device), self._critic_buffers)
        if self._use_actor_target:
            self._actor_target_params = torch.utils._pytree.tree_map(lambda p: p.to(device), self._actor_target_params)
            self._actor_target_buffers = torch.utils._pytree.tree_map(lambda p: p.to(device), self._actor_target_buffers)
        if self._use_critic_target:
            self._critic_target_params = torch.utils._pytree.tree_map(lambda p: p.to(device), self._critic_target_params)
            self._critic_target_buffers = torch.utils._pytree.tree_map(lambda p: p.to(device), self._critic_target_buffers)
        self._actor_optimizer_state = torch.utils._pytree.tree_map(lambda p: p.to(device), self._actor_optimizer_state)
        self._critic_optimizer_state = torch.utils._pytree.tree_map(lambda p: p.to(device), self._critic_optimizer_state)
