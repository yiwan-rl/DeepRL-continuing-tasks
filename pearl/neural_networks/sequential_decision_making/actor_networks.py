# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

"""
This module defines several types of actor neural networks.
"""


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from pearl.neural_networks.common.utils import (
    compute_output_dim_model_cnn,
    conv_block,
    mlp_block,
)
from torch import Tensor
from torch.distributions import Normal


def action_scaling(
    action_space_low: torch.Tensor, action_space_high: torch.Tensor, input_action: torch.Tensor
) -> torch.Tensor:
    """
    Center and scale input action from [-1, 1]^{action_dim} to [low, high]^{action_dim}.
    Use cases:
        - For continuous action spaces, actor networks output "normalized_actions",
            i.e. actions in the range [-1, 1]^{action_dim}.

    Note: the action space is not assumed to be symmetric (low = -high).

    Args:
        action_space_low: the action space low, shape (action_dim)
        action_space_high: the action space high, shape (action_dim)
        input_action: the input action vector to be scaled, shape (batch_size, action_dim)
    Returns:
        scaled_action: centered and scaled input action vector, according to the action space
    """
    centered_and_scaled_action = (((action_space_high - action_space_low) * (input_action + 1.0)) / 2) + action_space_low
    return centered_and_scaled_action


def action_unscaling(
    action_space_low: torch.Tensor, action_space_high: torch.Tensor, input_action: torch.Tensor
) -> torch.Tensor:
    """
    The reverse operation of action_scaling
    """
    unscaled_action = (((input_action - action_space_low) / (action_space_high - action_space_low)) * 2.0) - 1.0
    return unscaled_action


def noise_scaling(action_space_low: torch.Tensor, action_space_high: torch.Tensor, input_noise: torch.Tensor) -> torch.Tensor:
    """
    This function rescales any input vector from [-1, 1]^{action_dim} to [low, high]^{action_dim}.
    Use case:
        - For noise based exploration, we need to scale the noise (for example, from the standard
            normal distribution) according to the action space.

    Args:
        action_space_low: the action space low
        action_space_high: the action space high
        input_vector: the input vector to be scaled
    Returns:
        torch.Tensor: scaled input vector, according to the action space
    """
    scaled_noise = ((action_space_high - action_space_low) / 2) * input_noise
    return scaled_noise


class VanillaActorNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]],
        output_dim: int,
        hidden_activation: str = "relu",
    ) -> None:
        """A Vanilla Actor Network is meant to be used with discrete action spaces.
           For an input state (batch of states), it outputs a probability distribution over
           all the actions.

        Args:
            input_dim: input state dimension (or dim of the state representation)
            hidden_dims: list of hidden layer dimensions
            output_dim: number of actions (action_space.n when used with discrete action space)
        """
        super().__init__()
        self._model: nn.Module = mlp_block(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            hidden_activation=hidden_activation,
            last_activation="softmax",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)

    def get_policy_distribution(
        self,
        state_batch: torch.Tensor,
        params,
    ) -> torch.Tensor:
        """
        Gets a policy distribution from a discrete actor network.
        The policy distribution is defined by the softmax of the output of the network.

        Args:
            state_batch: batch of states with shape (batch_size, state_dim) or (state_dim)
        """
        if len(state_batch.shape) == 1:
            state_batch = state_batch.unsqueeze(0)
            reshape_state_batch = True
        else:
            reshape_state_batch = False
        policy_distribution = torch.func.functional_call(
            self, params, (state_batch)
        )  # shape (batch_size, num_actions)
        if reshape_state_batch:
            policy_distribution = policy_distribution.squeeze(0)
        return policy_distribution  # shape (batch_size, num_actions) or (num_actions)

    def get_action_log_prob_and_entropy(
        self,
        state_batch: torch.Tensor,
        action_batch: torch.Tensor,
        params,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gets probabilities of different actions from a discrete actor network.
        Assumes that the input batch of actions is one-hot encoded
            (generalize it later).

        Args:
            state_batch: batch of states with shape (batch_size, input_dim)
            action_batch: batch of actions with shape (batch_size, output_dim)
        Returns:
            action_log_probs: probabilities of each action in the batch with shape (batch_size)
            entropy: entropy of the policy distribution with shape (batch_size)
        """
        all_action_probs = torch.func.functional_call(self, params, (state_batch))  # shape: (batch_size, output_dim)

        log_all_action_probs = torch.log(all_action_probs + 1e-8)
        log_action_probs = torch.sum(
            log_all_action_probs * action_batch, dim=1, keepdim=True
        )
        entropy = -torch.sum(
            all_action_probs * log_all_action_probs, dim=1, keepdim=True
        )
        return (
            log_action_probs,
            entropy,
        )  # shape (batch_size, 1), (batch_size, 1)


class CNNActorNetwork(nn.Module):
    def __init__(
        self,
        input_width: int,
        input_height: int,
        input_channels_count: int,
        kernel_sizes: List[int],
        output_channels_list: List[int],
        strides: List[int],
        paddings: List[int],
        hidden_dims_fully_connected: Optional[List[int]] = None,
        output_dim: int = 1,
        use_batch_norm_conv: bool = False,
        use_batch_norm_fully_connected: bool = False,
    ) -> None:
        """A CNN Actor Network is meant to be used with CNN to deal with images.
           For an input state (batch of states), it outputs a probability distribution over
           all the actions.

        Args:
            input_dim: input state dimension (or dim of the state representation)
            hidden_dims: list of hidden layer dimensions
            output_dim: number of actions (action_space.n when used with discrete action space)
        """
        super().__init__()
        self._input_channels = input_channels_count
        self._input_height = input_height
        self._input_width = input_width
        self._output_channels = output_channels_list
        self._kernel_sizes = kernel_sizes
        self._strides = strides
        self._paddings = paddings
        if hidden_dims_fully_connected is None:
            self._hidden_dims_fully_connected: List[int] = []
        else:
            self._hidden_dims_fully_connected: List[int] = hidden_dims_fully_connected

        self._use_batch_norm_conv = use_batch_norm_conv
        self._use_batch_norm_fully_connected = use_batch_norm_fully_connected
        self._output_dim = output_dim

        self._model_cnn: nn.Module = conv_block(
            input_channels_count=self._input_channels,
            output_channels_list=self._output_channels,
            kernel_sizes=self._kernel_sizes,
            strides=self._strides,
            paddings=self._paddings,
            use_batch_norm=self._use_batch_norm_conv,
        )
        # we concatenate actions to state representations in the mlp block of the Q-value network
        self._mlp_input_dims: int = compute_output_dim_model_cnn(
            input_channels=input_channels_count,
            input_width=input_width,
            input_height=input_height,
            model_cnn=self._model_cnn,
        )
        self._model_fc: nn.Module = mlp_block(
            input_dim=self._mlp_input_dims,
            hidden_dims=self._hidden_dims_fully_connected,
            output_dim=self._output_dim,
            use_batch_norm=self._use_batch_norm_fully_connected,
            last_activation="softmax",
        )
        self._state_dim: int = input_channels_count * input_height * input_width

    def forward(
        self,
        state_batch: torch.Tensor,  # shape: (batch_size, input_channels, input_height, input_width)
    ) -> torch.Tensor:
        batch_size = state_batch.shape[0]
        state_representation_batch = self._model_cnn(
            state_batch / 255.0
        )  # (batch_size x output_channels[-1] x output_height x output_width)
        state_representation_batch = state_representation_batch.view(
            batch_size, -1
        )  # (batch_size x state dim)
        policy = self._model_fc(
            state_representation_batch
        )  # (batch_size x num actions)
        return policy

    def get_policy_distribution(
        self,
        state_batch: torch.Tensor,  # shape: (batch_size, input_channels, input_height, input_width)
    ) -> torch.Tensor:
        """
        Gets a policy distribution from a discrete actor network.
        The policy distribution is defined by the softmax of the output of the network.

        Args:
            state_batch: batch of states with shape (batch_size, state_dim) or (state_dim)
        """
        if len(state_batch.shape) == 3:
            state_batch = state_batch.unsqueeze(0)
            reshape_state_batch = True
        else:
            reshape_state_batch = False
        assert len(state_batch.shape) == 4
        policy_distribution = self.forward(
            state_batch
        )  # shape (batch_size, num_actions)
        if reshape_state_batch:
            policy_distribution = policy_distribution.squeeze(0)
        return policy_distribution

    def get_action_log_prob_and_entropy(
        self,
        state_batch: torch.Tensor,
        action_batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gets probabilities of different actions from a discrete actor network.
        Assumes that the input batch of actions is one-hot encoded
            (generalize it later).

        Args:
            state_batch: batch of states with shape (batch_size, input_dim)
            action_batch: batch of actions with shape (batch_size, output_dim)
        Returns:
            action_log_probs: probabilities of each action in the batch with shape (batch_size)
            entropy: entropy of the policy distribution with shape (batch_size)
        """
        all_action_probs = self.forward(state_batch)  # shape: (batch_size, output_dim)
        log_all_action_probs = torch.log(all_action_probs + 1e-8)
        log_action_probs = torch.sum(
            log_all_action_probs * action_batch, dim=1, keepdim=True
        )
        entropy = -torch.sum(
            all_action_probs * log_all_action_probs, dim=1, keepdim=True
        )
        return (
            log_action_probs,
            entropy,
        )  # shape (batch_size, 1), (batch_size, 1)


class VanillaContinuousActorNetwork(nn.Module):
    """
    This is vanilla version of deterministic actor network
    Given input state, output an action vector
    Args
        output_dim: action dimension
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]],
        output_dim: int,
        effective_input_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._model: nn.Module = mlp_block(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            last_activation="tanh",
            effective_input_dim=effective_input_dim,
        )
            

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)

    def sample_action(
            self, 
            x: torch.Tensor, 
            params: torch.Tensor, 
            buffers: torch.Tensor,
            action_space_low: torch.Tensor, 
            action_space_high: torch.Tensor, 
            action_space_mask: torch.Tensor,
        ) -> torch.Tensor:
        """
        Sample an action from the actor network.
        Args:
            x: input state
        Returns:
            action: sampled action, scaled to the action space bounds
        """
        normalized_action = torch.func.functional_call(self, (params, buffers), (x))
        action = action_scaling(action_space_low, action_space_high, normalized_action)
        action = action * action_space_mask
        return action


class GaussianActorNetwork(nn.Module):
    """
    A multivariate gaussian actor network: parameterize the policy (action distirbution)
    as a multivariate gaussian. Given input state, the network outputs a pair of
    (mu, sigma), where mu is the mean of the Gaussian distribution, and sigma is its
    standard deviation along different dimensions.
       - Note: action distribution is assumed to be independent across different
         dimensions
       - Note: we implement both state-independent and state-dependent standard deviations
    Args:
        input_dim: input state dimension
        hidden_dims: list of hidden layer dimensions; cannot pass an empty list
        output_dim: action dimension
        state_conditioned_std: if True,
            the standard deviation shares the same network body with the mean,
            otherwise it is a set of k trainable parameters, where k is the number of actions
        hidden_activation: activation function for hidden layers
        log_std_init_offset: initial value of log of standard deviation,
            used only for state-independent std
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        hidden_activation: str = "relu",
        effective_input_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if len(hidden_dims) < 1:
            raise ValueError(
                "The hidden dims cannot be empty for a gaussian actor network."
            )

        self._model: nn.Module = mlp_block(
            input_dim=input_dim,
            hidden_dims=hidden_dims[:-1],
            output_dim=hidden_dims[-1],
            hidden_activation=hidden_activation,
            last_activation=hidden_activation,
            effective_input_dim=effective_input_dim,
        )
        self.fc_mu = torch.nn.Linear(hidden_dims[-1], output_dim)
        self.fc_std: nn.Module = torch.nn.Linear(hidden_dims[-1], output_dim)


        # preventing the actor network from learning a flat or a point mass distribution
        self._log_std_min = -5
        self._log_std_max = 2

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self._model(x)
        mean = self.fc_mu(x)
        log_std = self.fc_std(x)
        # log_std = torch.clamp(log_std, min=self._log_std_min, max=self._log_std_max)

        # alternate to standard clamping; not sure if it makes a difference but still
        # trying out
        log_std = torch.tanh(log_std)
        log_std = self._log_std_min + 0.5 * (
            self._log_std_max - self._log_std_min
        ) * (log_std + 1)
        return mean, log_std

    def sample_action(
        self, 
        state_batch: Tensor, 
        params, 
        buffers,
        action_space_low: torch.Tensor, 
        action_space_high: torch.Tensor, 
        action_space_mask: torch.Tensor,
        get_log_prob: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Sample an action from the actor network.

        Args:
            state_batch: A tensor of states.  # TODO: Enforce batch shape?
            get_log_prob: If True, also return the log probability of the sampled actions.

        Returns:
            action: Sampled action, scaled to the action space bounds.
            log_prob [Optional]: log probability of the sampled action.
        """
        mean, log_std = torch.func.functional_call(self, (params, buffers), (state_batch))
        std = log_std.exp()
        normal = Normal(mean, std)
        # normal.rsample() is not compatible with vmap, so we do reparameterization trick here manually
        noise = torch.normal(0, 1, mean.shape, device=mean.device)
        sample = mean + std * noise

        # ensure sampled action is within [-1, 1]^{action_dim}
        normalized_action = torch.tanh(sample)

        # clamp each action dimension to prevent numerical issues in tanh
        # normalized_action.clamp(-1 + epsilon, 1 - epsilon)
        action = action_scaling(action_space_low, action_space_high, normalized_action)
        action = action * action_space_mask
        if get_log_prob:
            action_bound = (action_space_high - action_space_low) / 2
            log_prob = self._get_log_prob_normal(normal, sample, normalized_action, action_bound, action_space_mask)
            return action, log_prob
        else:
            return action

    def _get_log_prob_normal(
        self,
        normal_dist: torch.distributions.Distribution,
        unnormalized_action_batch: Tensor,
        normalized_action_batch: Tensor,
        action_bound: torch.Tensor,
        action_space_mask: torch.Tensor,
    ) -> Tensor:
        log_prob = normal_dist.log_prob(unnormalized_action_batch)
        log_prob -= torch.log(
            action_bound * (1 - normalized_action_batch.pow(2)) + 1e-6
        )

        log_prob = log_prob * action_space_mask

        # for multi-dimensional action space, sum log probabilities over individual
        # action dimension
        if log_prob.dim() == 2:
            log_prob = log_prob.sum(dim=1)

        return log_prob


class ClipGaussianActorNetwork(nn.Module):
    """
    A multivariate gaussian actor network: parameterize the policy (action distirbution)
    as a multivariate gaussian. Given input state, the network outputs a pair of
    (mu, sigma), where mu is the mean of the Gaussian distribution, and sigma is its
    standard deviation along different dimensions.
       - Note: action distribution is assumed to be independent across different
         dimensions
       - Note: we implement both state-independent and state-dependent standard deviations
    Args:
        input_dim: input state dimension
        hidden_dims: list of hidden layer dimensions; cannot pass an empty list
        output_dim: action dimension
        state_conditioned_std: if True,
            the standard deviation shares the same network body with the mean,
            otherwise it is a set of k trainable parameters, where k is the number of actions
        hidden_activation: activation function for hidden layers
        log_std_init_offset: initial value of log of standard deviation,
            used only for state-independent std
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        hidden_activation: str = "tanh",
        log_std_init_offset: float = 0.0,
        effective_input_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if len(hidden_dims) < 1:
            raise ValueError(
                "The hidden dims cannot be empty for a gaussian actor network."
            )

        self._model: nn.Module = mlp_block(
            input_dim=input_dim,
            hidden_dims=hidden_dims[:-1],
            output_dim=hidden_dims[-1],
            hidden_activation=hidden_activation,
            last_activation=hidden_activation,
            effective_input_dim=effective_input_dim,
        )
        self.fc_mu = torch.nn.Linear(hidden_dims[-1], output_dim)
        self.log_std: torch.Tensor = nn.Parameter(
            torch.zeros(output_dim) + log_std_init_offset
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self._model(x)
        mean = self.fc_mu(x)
        log_std = self.log_std.expand_as(mean)
        return mean, log_std

    def sample_action(
        self, state_batch: Tensor, params, buffer, action_space_low: torch.Tensor = None, action_space_high: torch.Tensor = None, action_space_mask: torch.Tensor = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Sample an action from the actor network.

        Args:
            state_batch: A tensor of states.  # TODO: Enforce batch shape?
            get_log_prob: If True, also return the log probability of the sampled actions.

        Returns:
            action: Sampled action, scaled to the action space bounds.
            log_prob [Optional]: log probability of the sampled action.
        """
        mean, log_std = torch.func.functional_call(self, (params, buffer), (state_batch))
        std = log_std.exp()
        normal = Normal(mean, std)
        action = normal.sample()  # reparameterization trick
        return action

    def get_action_log_prob_and_entropy(
        self, 
        state_batch: torch.Tensor, 
        action_batch: torch.Tensor, 
        params, 
        buffers,
        action_space_mask: torch.Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute log probability of actions, pi(a|s) under the policy parameterized by
        the actor network.
        Args:
            state_batch: batch of states
            action_batch: batch of actions
        Returns:
            log_prob: log probability of each action in the batch
        """
        mean, log_std = torch.func.functional_call(self, (params, buffers), (state_batch))
        std = log_std.exp()
        normal = Normal(mean, std)
        log_prob = normal.log_prob(action_batch)
        log_prob = log_prob * action_space_mask
        entropy = normal.entropy()
        entropy = entropy * action_space_mask
        assert log_prob.dim() == 2
        assert entropy.dim() == 2
        if log_prob.dim() == 2:
            log_prob = log_prob.sum(dim=1, keepdim=True)
        if entropy.dim() == 2:
            entropy = entropy.sum(dim=1, keepdim=True)
        return log_prob, entropy  # shape (batch_size, 1), (batch_size, 1)
