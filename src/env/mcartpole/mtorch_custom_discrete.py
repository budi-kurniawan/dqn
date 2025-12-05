import torch
from torch import device

class MTorchCustomDiscrete:
    """
    A minimal class that mimics the gymnasium.spaces.Discrete functionality 
    to support the .sample() method.
    """
    def __init__(self, n, device: device, n_envs: int):
        """
        Initializes the space.
        Args:
            n (int): The number of possible actions (0, 1, ..., n-1).
        """
        self._device = device
        self.n = n
        self._n_envs = n_envs
        self.space = torch.arange(n, device=device)

    def sample(self):
        return torch.randint(low=0, high=self.n, size=(self._n_envs,), device=self._device)