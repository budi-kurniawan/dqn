import torch
from torch import Tensor, device
from env.cartpole.cartpole import Cartpole

class CartpoleEnv:

    MAX_STEPS = 500

    def __init__(self, seed: int, device: torch.device):
        self._cartpole = Cartpole(seed, device)
        self._one_tensor = torch.tensor(1.0, device=device)
        self._steps_done = torch.tensor(0, device=device, dtype=torch.int32)
        self._X_THRESHOLD_TENSOR = torch.tensor(Cartpole.X_THRESHOLD, device=device)
        self._THETA_THRESHOLD_TENSOR = torch.tensor(Cartpole.THETA_THRESHOLD, device=device)


    def reset(self) -> Tensor:
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
        return torch.cat((state, reward, terminated_float), dim=0)



