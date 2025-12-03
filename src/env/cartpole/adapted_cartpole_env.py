import torch
from torch import Tensor, device
from env.cartpole.cartpole import Cartpole, X_THRESHOLD, THETA_THRESHOLD
from env.cartpole.custom_discrete import CustomDiscrete
from env.cartpole.cartpole_env import CartpoleEnv

class AdaptedCartpoleEnv(CartpoleEnv):
    """
        Class for testing tensor-based CartpoleEnv with DQNAgent that takes Python types
    """
    def __init__(self, seed: int, device: torch.device):
        super().__init__(seed, device)

    def reset(self, seed=42) -> Tensor:
        response = super().reset(seed)
        #print("response on reset:", response)
        return response.cpu().tolist(), {}

    def step(self, action: torch) -> Tensor:
        action_tensor = torch.tensor(action, dtype=torch.int32)
        state, reward, terminated_float, truncated = super().step(action_tensor)
        return state.cpu().tolist(), reward.item(), (terminated_float != 0).item(), (truncated != 0).item(), {}

