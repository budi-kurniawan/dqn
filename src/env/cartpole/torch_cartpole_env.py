import torch
from torch import Tensor, device
from env.cartpole.torch_cartpole import TorchCartpole, X_THRESHOLD, THETA_THRESHOLD
from env.cartpole.torch_custom_discrete import TorchCustomDiscrete

class TorchCartpoleEnv:
    

    MAX_STEPS = 500

    def __init__(self, seed: int, device: torch.device):
        self.action_space = TorchCustomDiscrete(2, device)
        self._cartpole = TorchCartpole(seed, device)
        self._one_tensor = torch.tensor(1.0, device=device)
        self._max_steps_tensor = torch.tensor(TorchCartpoleEnv.MAX_STEPS, device=device, dtype=torch.int32)
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
        terminated = torch.logical_or(x_terminated, theta_terminated) # tensor of bools
        
        # reward = 1 if not terminated, 0 if terminated
        reward = (self._one_tensor - terminated.float()).view(1)
        truncated = self._steps_done >= self._max_steps_tensor
        # state: shape([4])
        # reward: shape([1])
        # terminated.view(1): shape([1])
        # truncated.view(1): shape([1])
        return state, reward, terminated.view(1), truncated.view(1)



