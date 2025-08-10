# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

import math
from typing import Any, Dict, Optional

import torch
from pearl.neural_networks.sequential_decision_making.actor_networks import (
    action_scaling,
)
from pearl.policy_learners.exploration_modules.exploration_module import (
    ExplorationModule,
)
from pearl.policy_learners.sequential_decision_making.actor_critic_base import (
    ActorCriticBase,
)
from pearl.replay_buffers.replay_buffer import ReplayBuffer
from pearl.replay_buffers.sequential_decision_making.on_policy_replay_buffer import (
    OnPolicyReplayBuffer,
    OnPolicyTransitionBatch,
)
from pearl.replay_buffers.transition import TransitionBatch
from pearl.utils.functional_utils.learning.preprocessing import RunningMeanStd
from pearl.utils.functional_utils.learning.reward_centering import MA_RC, RVI_RC, TD_RC
from torch import nn
from pearl.utils.instantiations.spaces import VectorDiscreteSpace, VectorBoxSpace
from torch.func import grad
from torch import vmap
from torch.func import functional_call
import torchopt


def actor_loss_fn(
        actor_network_instance: nn.Module, 
        actor_params, 
        actor_buffers,
        state,
        action,
        action_log_probs_old,
        action_space_mask,
        epsilon,
        norm_adv,
        entropy_bonus_scaling,
        gae,
    ):
    action_log_probs, entropy = actor_network_instance.get_action_log_prob_and_entropy(
        state_batch=state,
        action_batch=action,
        params=actor_params,
        buffers=actor_buffers,
        action_space_mask=action_space_mask,
    )  # shape (batch_size, 1), (batch_size, 1)

    r_thelta = torch.exp(
        action_log_probs - action_log_probs_old
    )  # shape (batch_size, 1)

    clip = torch.clamp(
        r_thelta, min=1.0 - epsilon, max=1.0 + epsilon
    )  # shape (batch_size, 1)
    if norm_adv:
        adv = (gae - gae.mean()) / (gae.std() + 1e-8)
        loss = torch.mean(-torch.min(r_thelta * adv, clip * adv))
    else:
        loss = torch.mean(-torch.min(r_thelta * gae, clip * gae))
    loss -= entropy_bonus_scaling * torch.mean(entropy)
    return loss


def critic_loss_fn(
    critic_network_instance: nn.Module,
    critic_params,
    critic_buffers,
    state, # (batch_size, state_dim)
    norm_return,
    lam_return,
    clip_value,
    epsilon,
    value_old,
    ret_rms_var,
):
    if norm_return:  # normalize the return
        assert lam_return is not None
        lam_return = lam_return / math.sqrt(float(ret_rms_var) + 1e-8)
    else:
        lam_return = lam_return
    value = critic_value_fn(critic_network_instance, critic_params, critic_buffers, state)  # shape (batch_size)

    if clip_value:
        assert value_old is not None
        v_clip = value_old + (value - value_old).clamp(
            -epsilon,
            epsilon,
        )
        vf1 = (lam_return - value).pow(2)
        vf2 = (lam_return - v_clip).pow(2)
        return torch.max(vf1, vf2).mean()
    else:
        return (lam_return - value).pow(2).mean()


def critic_value_fn(
    critic_network_instance: nn.Module,
    critic_params,
    critic_buffers,
    state,
):
    return functional_call(critic_network_instance, (critic_params, critic_buffers), (state))


class ProximalPolicyOptimization(ActorCriticBase):
    """
    paper: https://arxiv.org/pdf/1707.06347.pdf.
    This class implements both discrete and continuous control versions of PPO.
    """

    def __init__(
        self,
        action_space: VectorDiscreteSpace | VectorBoxSpace,
        actor_network_instances: nn.ModuleList,
        critic_network_instances: nn.ModuleList,
        actor_optimizer,
        critic_optimizer,
        exploration_module: ExplorationModule,
        is_action_continuous: bool,
        discount_factor: float = 0.99,
        training_rounds: int = 100,
        batch_size: int = 128,
        epsilon: float = 0.0,
        trace_decay_param: float = 0.95,
        entropy_bonus_scaling: float = 0.0,
        norm_return: bool = False,
        norm_adv: bool = False,
        critic_weight: float = 0.5,
        max_grad_norm: Optional[float] = None,
        anneal_lr: bool = True,
        max_steps: Optional[int] = None,
        clip_value: bool = False,
        reprocessing_buffer: bool = True,
        reward_rate: torch.Tensor = torch.tensor(0.0),
        reward_centering: Optional[TD_RC | RVI_RC | MA_RC] = None,
    ) -> None:
        super(ProximalPolicyOptimization, self).__init__(
            action_space=action_space,
            use_actor_target=False,
            use_critic_target=False,
            actor_soft_update_tau=0.0,  # not used
            critic_soft_update_tau=0.0,  # not used
            ensemble_critic_size=1,  # not used
            exploration_module=exploration_module,
            discount_factor=discount_factor,
            training_rounds=training_rounds,
            batch_size=batch_size,
            is_action_continuous=is_action_continuous,
            actor_network_instances=actor_network_instances,
            critic_network_instances=critic_network_instances,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            reward_rate=reward_rate,
            reward_centering=reward_centering,
        )
        self._epsilon = epsilon
        self._trace_decay_param = trace_decay_param
        self._entropy_bonus_scaling = entropy_bonus_scaling
        self._norm_return = norm_return
        self._norm_adv = norm_adv
        self._critic_weight = critic_weight
        self._max_grad_norm = max_grad_norm
        self._ret_rms: RunningMeanStd = RunningMeanStd(shape=(1,))
        self._anneal_lr = anneal_lr
        self._max_steps = max_steps
        self._clip_value = clip_value
        self._reprocessing_buffer = reprocessing_buffer

        # Wrap grad with vmap to compute gradients per experiment
        self._actor_grad_fn = vmap(
            grad(actor_loss_fn, argnums=1),
            in_dims=(
                None,  # actor_network_instance
                0,  # actor_params
                0,  # actor_buffers
                0,  # state
                0,  # action
                0,  # action_log_probs_old
                0,  # action_space_mask
                None,  # epsilon
                None,  # norm_adv
                None,  # entropy_bonus_scaling
                0,  # gae
            ),
            randomness="different"
        )
        self._critic_grad_fn = vmap(
            grad(critic_loss_fn, argnums=1),
            in_dims=(
                None,  # critic_network_instance
                0,  # critic_params batched over experiments
                0,  # critic_buffers batched over experiments
                0,  # state
                None,  # norm_return
                0,  # lam_return
                None,  # clip_value
                None,  # epsilon
                0,  # value_old
                None,  # ret_rms_var
            ),
            randomness="different"
        )
    
    def _get_actor_gradient(self, batch: TransitionBatch) -> torch.Tensor:
        # batch.state shape: (batch_size, num_exps, state_dim)
        # Make sure batch.state shape is (batch_size, num_exps, state_dim)
        grads = self._actor_grad_fn(
            self._actor,
            self._actor_params,
            self._actor_buffers,
            batch.state,
            batch.action,
            batch.action_log_probs,
            self._action_space.mask,
            torch.tensor(self._epsilon, device=batch.state.device),
            torch.tensor(self._norm_adv, device=batch.state.device),
            torch.tensor(self._entropy_bonus_scaling, device=batch.state.device),
            batch.gae,
        )
        return torch.utils._pytree.tree_map(lambda g: g.detach(), grads)

    def _get_critic_gradient(self, batch: TransitionBatch) -> torch.Tensor:
        grads = self._critic_grad_fn(
            self._critic,
            self._critic_params,
            self._critic_buffers,
            batch.state,
            torch.tensor(self._norm_return),
            batch.lam_return,
            torch.tensor(self._clip_value),
            torch.tensor(self._epsilon).to(batch.state.device),
            batch.value,
            torch.tensor(self._ret_rms.var).to(batch.state.device),
        )
        return torch.utils._pytree.tree_map(lambda g: g.detach(), grads)



    def learn_batch(self, batch: TransitionBatch) -> Dict[str, Any]:
        # Compute actor losses per experiment
        actor_loss_fn_batch = vmap(actor_loss_fn, in_dims=(None, 0, 0, 0, 0, 0, 0, None, None, None, 0), randomness="different")
        actor_losses = actor_loss_fn_batch(
            self._actor,
            self._actor_params,
            self._actor_buffers,
            batch.state,
            batch.action,
            batch.action_log_probs,
            self._action_space.mask,
            self._epsilon,
            self._norm_adv,
            self._entropy_bonus_scaling,
            batch.gae,
        )
        
        # Compute critic losses per experiment
        critic_loss_fn_batch = vmap(critic_loss_fn, in_dims=(None, 0, 0, 0, None, 0, None, None, 0, None), randomness="different")
        critic_losses = critic_loss_fn_batch(
            self._critic,
            self._critic_params,
            self._critic_buffers,
            batch.state,
            self._norm_return,
            batch.lam_return,
            self._clip_value,
            self._epsilon,
            batch.value,
            self._ret_rms.var,
        )

        actor_gradients = self._get_actor_gradient(batch)
        critic_gradients = self._get_critic_gradient(batch)

        # Update parameters
        actor_updates, self._actor_optimizer_state = self._actor_optimizer.update(
            actor_gradients, self._actor_optimizer_state
        )
        critic_updates, self._critic_optimizer_state = self._critic_optimizer.update(
            critic_gradients, self._critic_optimizer_state
        )
        self._actor_params = torchopt.apply_updates(self._actor_params, actor_updates)
        self._critic_params = torchopt.apply_updates(self._critic_params, critic_updates)

        return {"actor_loss": actor_losses.detach().cpu(), "critic_loss": critic_losses.detach().cpu()}

    def learn(self, replay_buffer: ReplayBuffer) -> Dict[str, Any]:
        if isinstance(self.reward_centering, RVI_RC):
            freq = self.reward_centering.ref_states_update_freq
            if self._training_steps % freq == 0:
                # pyre-fixme
                batch = replay_buffer.create_f_batch(
                    batch_size=self._batch_size, last_k_steps=freq
                )
        if len(replay_buffer) == 0:
            return {}

        if self._batch_size == -1 or len(replay_buffer) < self._batch_size:
            batch_size = len(replay_buffer)
        else:
            batch_size = self._batch_size

        report = {}
        for tr in range(self._training_rounds):
            if tr == 0:
                self.preprocess_replay_buffer(
                    replay_buffer, update_action_log_prob=True
                )
            elif (
                self._reprocessing_buffer
                and tr % (len(replay_buffer) // batch_size) == 0
            ):
                self.preprocess_replay_buffer(
                    replay_buffer, update_action_log_prob=False
                )
            self._training_steps += 1
            batch = replay_buffer.sample(batch_size)
            single_report = {}
            if isinstance(batch, TransitionBatch):
                single_report = self.learn_batch(batch)

            for k, v in single_report.items():
                if k in report:
                    report[k].append(v)
                else:
                    report[k] = [v]
        return report

    def preprocess_replay_buffer(
        self, replay_buffer: ReplayBuffer, update_action_log_prob: bool = True
    ) -> None:
        """
        Preprocess the replay buffer by calculating
        and adding the generalized advantage estimates (gae),
        truncated lambda returns (lam_return) and action log probabilities (action_log_probs)
        under the current policy.
        See https://arxiv.org/abs/1707.06347 equation (11) for the definition of gae.
        See "Reinforcement Learning: An Introduction" by Sutton and Barto (2018) equation (12.10)
        for the definition of truncated lambda return.
        """
        if isinstance(self.reward_centering, RVI_RC):
            self.reward_rate = self.compute_f_value(self.reward_centering.f_batch)
        assert type(replay_buffer) is OnPolicyReplayBuffer
        replay_buffer.init_indices()
        batch = replay_buffer.sample_all()
        state_values = (
            vmap(
                critic_value_fn, 
                in_dims=(None, 0, 0, 0),
                randomness="different"
            )(self._critic, self._critic_params, self._critic_buffers, batch.state).detach().cpu()
        )  # shape (num_exps, batch_size, 1)

        next_state_values = (
            vmap(
                critic_value_fn, 
                in_dims=(None, 0, 0, 0),
                randomness="different"
            )(self._critic, self._critic_params, self._critic_buffers, batch.next_state).detach().cpu()
        )  # shape (num_exps, batch_size, 1)

        if self._norm_return:  # unnormalize state_values
            state_values = state_values * math.sqrt(float(self._ret_rms.var) + 1e-8)
            next_state_values = next_state_values * math.sqrt(
                float(self._ret_rms.var) + 1e-8
            )
        if update_action_log_prob:
            action_log_probs, _ = vmap(
                self._actor.get_action_log_prob_and_entropy, 
                in_dims=(0, 0, 0, 0, 0),
                randomness="different"
            )(batch.state, batch.action, self._actor_params, self._actor_buffers, self._action_space.mask)  # shape (num_exps, batch_size, 1)
            action_log_probs = action_log_probs.detach().cpu()  # shape (num_exps, batch_size, 1)
            replay_buffer.action_log_probs = action_log_probs

        reward = batch.reward.unsqueeze(-1).cpu()  # shape (num_exps, batch_size, 1)
        terminated = batch.terminated.unsqueeze(-1).cpu()  # shape (num_exps, batch_size, 1)
        truncated = batch.truncated.unsqueeze(-1).cpu()  # shape (num_exps, batch_size, 1)
        if isinstance(self.reward_centering, TD_RC):
            if self.reward_centering.initialize_reward_rate == True:
                self.reward_rate.data.fill_(batch.reward.mean())
                # pyre-fixme
                self.reward_centering.initialize_reward_rate = False
            # pyre-fixme
            self.reward_centering.optimizer.zero_grad()
            td_errors = (
                reward
                - self.reward_rate.cpu()
                + self._discount_factor
                * next_state_values
                * torch.logical_not(terminated)
                - state_values
            )  # shape (num_exps, batch_size, 1)
            reward_rate_error = td_errors.pow(2).mean()
            reward_rate_error.backward()
            # pyre-fixme
            self.reward_centering.optimizer.step()

        td_errors = (
            reward
            - self.reward_rate.detach().cpu()
            + self._discount_factor * next_state_values * torch.logical_not(terminated)
            - state_values
        )  # shape (num_exps, batch_size, 1)

        discounting = (
            torch.logical_not(torch.logical_or(terminated, truncated))
            * self._discount_factor
            * self._trace_decay_param
        )  # shape (num_exps, batch_size, 1)

        replay_buffer.gae = torch.zeros_like(td_errors)
        replay_buffer.gae[:, -1] = td_errors[:, -1]

        for i in range(replay_buffer.pos - 2, -1, -1):
            # pyre-fixme[16]: `Optional` has no attribute `__setitem__`.
            replay_buffer.gae[:, i] = (
                # pyre-fixme[16]: `Optional` has no attribute `__setitem__`.
                td_errors[:, i]
                # pyre-fixme
                + discounting[:, i] * replay_buffer.gae[:, i + 1]
            )  # shape (num_exps, batch_size, 1)
        # pyre-fixme
        replay_buffer.lam_return = replay_buffer.gae + state_values
        replay_buffer.value = state_values
        if self._norm_return:
            self._ret_rms.update(replay_buffer.lam_return.cpu().numpy())

    def action_post_processing(
        self,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Post-process the action before sending to the environment.
        """
        if self._is_action_continuous:
            action = torch.clamp(action, min=-1.0, max=1.0)
            action = action_scaling(self._action_space.low, self._action_space.high, action)
            action = action * self._action_space.mask
        return action

    def compute_f_value(self, batch: TransitionBatch) -> torch.Tensor:
        """
        Computes the f value for a batch of transitions.
        Args:
            batch (TransitionBatch): A batch of transitions.
        Returns:
            f_value (Tensor): The value function for the batch of transitions.
        """
        vs = self._critic(batch.state)
        f_value = torch.mean(vs).detach()
        if self._norm_return:  # unnormalize state_values
            f_value = f_value * math.sqrt(float(self._ret_rms.var) + 1e-8)
        return f_value
