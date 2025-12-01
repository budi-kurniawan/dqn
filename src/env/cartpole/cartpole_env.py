import torch
from torch import Tensor, device
from env.cartpole.cartpole import Cartpole, X_THRESHOLD, THETA_THRESHOLD
from env.cartpole.custom_discrete import CustomDiscrete

class CartpoleEnv:
    

    MAX_STEPS = 500

    def __init__(self, seed: int, device: torch.device):
        self.action_space = CustomDiscrete(2, device)
        self._cartpole = Cartpole(seed, device)
        self._one_tensor = torch.tensor(1.0, device=device)
        self._max_steps_tensor = torch.tensor(CartpoleEnv.MAX_STEPS, device=device, dtype=torch.int32)
        self._steps_done = torch.tensor(0, device=device, dtype=torch.int32)
        self._X_THRESHOLD_TENSOR = torch.tensor(X_THRESHOLD, device=device)
        self._THETA_THRESHOLD_TENSOR = torch.tensor(THETA_THRESHOLD, device=device)


    def reset(self, seed=42) -> Tensor:
        self._steps_done.zero_()
        return self._cartpole.reset()

    def step(self, action: torch) -> Tensor:
        self._steps_done.add_(1)
        state: Tensor = self._cartpole.apply_action(action)
        abs_state = state.abs()
        x_terminated = abs_state[0] > self._X_THRESHOLD_TENSOR
        theta_terminated = abs_state[2] > self._THETA_THRESHOLD_TENSOR
        
        # Logical OR is performed on the GPU
        terminated_bool_tensor = torch.logical_or(x_terminated, theta_terminated)
        
        # Final conversion to float (1.0 or 0.0) is performed on the GPU
        terminated_float = terminated_bool_tensor.float()

        # reward = 1 if not terminated, 0 if terminated
        reward = (self._one_tensor - terminated_float).view(1)
        terminated_float = terminated_float.view(1)

        truncated = (self._steps_done >= self._max_steps_tensor).float().view(1)
        return state, reward, terminated_float, truncated



