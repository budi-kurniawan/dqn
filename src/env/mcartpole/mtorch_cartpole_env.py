import torch
from torch import Tensor
from env.mcartpole.mtorch_cartpole import MTorchCartpole, X_THRESHOLD, THETA_THRESHOLD
from env.mcartpole.mtorch_custom_discrete import MTorchCustomDiscrete

class MTorchCartpoleEnv:    

    MAX_STEPS = 500

    def __init__(self, device: torch.device, n_envs: int):
        self._n_envs = n_envs
        self.action_space = MTorchCustomDiscrete(2, device)
        self._cartpole = MTorchCartpole(device, n_envs)
        self._max_steps_tensor = torch.tensor(MTorchCartpoleEnv.MAX_STEPS, device=device, dtype=torch.int32)
        self._steps_done = torch.tensor(0, device=device, dtype=torch.int32)
        self._X_THRESHOLD_TENSOR = torch.tensor(X_THRESHOLD, device=device)
        self._THETA_THRESHOLD_TENSOR = torch.tensor(THETA_THRESHOLD, device=device)
        self._rewards = torch.tensor([0], device=device)

    def reset(self) -> Tensor:
        self._steps_done.zero_()
        return self._cartpole.reset() #shape(n_envs, n_observations)

    def step(self, action: torch) -> Tensor:
        self._steps_done.add_(self._n_envs)
        state: Tensor = self._cartpole.apply_action(action) #state shape(n_envs, 4)
        abs_state = state.abs()
        x_terminated = abs_state[:, 0] > self._X_THRESHOLD_TENSOR
        theta_terminated = abs_state[:, 2] > self._THETA_THRESHOLD_TENSOR
        
        # Logical OR is performed on the GPU
        terminated = torch.logical_or(x_terminated, theta_terminated) # bool tensor of shape[n_envs]
        # reward = 1 if not terminated, 0 if terminated
        reward = (~terminated).float() #(self._one_tensor - terminated.float()).view(1)
        truncated = self._steps_done >= self._max_steps_tensor
        randoms = self._cartpole.generate_random_tensor()
        # TODO if done, push steps_done to rewards
        return state, reward, terminated, truncated, randoms #shape(n_envs,4), (n_envs), (n_envs), (n_envs)
