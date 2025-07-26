from typing import List

import torch
from torch import Tensor

import gymnasium as gym
from gymnasium.spaces import Box


class VectorBoxSpace:
    """A continuous, box space. This class is used to represent a list of box gym spaces.
    """

    def __init__(
        self,
        low: Tensor,
        high: Tensor,
        mask: Tensor,
    ) -> None:
        super(VectorBoxSpace, self).__init__()
        """Contructs a `BoxSpace`.

        Args:
            low: The lower bound on each dimension of the space.
            high: The upper bound on each dimension of the space.
            seed: Random seed used to initialize the random number generator of the
                underlying Gymnasium `Box` space.
        """
        self.low = low
        self.high = high
        self.device = None
        self.mask = mask
        self.actual_sizes = mask.sum(dim=1, dtype=torch.int32)

    def num_elements(self) -> int:
        """Returns the number of elements in the space."""
        return self.low.shape[0]
    
    def element_dim(self) -> int:
        """Returns the dimension of each element in the space."""
        return self.low.shape[1]
    
    def to(self, device: torch.device) -> None:
        self.device = device
        self.low = self.low.to(device)
        self.high = self.high.to(device)
        self.mask = self.mask.to(device)
        
    def sample(self) -> Tensor:
        """Sample an element uniformly at random from the space.
        """
        assert self.device is not None, "Device is not set"
        sampled_tensor = torch.rand(self.low.shape).to(self.device) * (self.high - self.low) + self.low
        sampled_tensor = sampled_tensor * self.mask
        return sampled_tensor

    @staticmethod
    def from_gym(gym_spaces: List[gym.Space]):
        """Constructs a `BoxSpace` given a Gymnasium `Box` space.

        Args:
            gym_space: A Gymnasium `Box` space.

        Returns:
            A `BoxSpace` with the same bounds and seed as `gym_space`.
        """
        for gym_space in gym_spaces:
            assert isinstance(gym_space, Box)
        
        # if the spaces have different dimensions, we need to pad them to the same dimension to use vmap
        max_dim = 0
        for gym_space in gym_spaces:
            if gym_space.low.shape[0] > max_dim:
                max_dim = gym_space.low.shape[0]

        # default low and high are -1 and 1
        low = torch.ones((len(gym_spaces), max_dim)) * -1
        high = torch.ones((len(gym_spaces), max_dim))
        mask = torch.zeros((len(gym_spaces), max_dim))

        for i, gym_space in enumerate(gym_spaces):
            low[i, :gym_space.low.shape[0]] = torch.from_numpy(gym_space.low)
            high[i, :gym_space.high.shape[0]] = torch.from_numpy(gym_space.high)
            mask[i, :gym_space.high.shape[0]] = 1
        return VectorBoxSpace(
            low=low,
            high=high,
            mask=mask,
        )
