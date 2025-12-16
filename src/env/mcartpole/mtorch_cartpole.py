import math
import torch
from torch import Tensor

X_THRESHOLD: float = 2.4
# 12 degrees converted to radians
THETA_THRESHOLD: float = 12.0 * 2.0 * math.pi / 360.0

# Physical properties
MASS_CART: float = 1.0
MASS_POLE: float = 0.1
TOTAL_MASS: float = MASS_CART + MASS_POLE
LENGTH: float = 0.5  # Half the pole's length is 0.5 (full length 1.0)
POLEMASS_LENGTH: float = MASS_POLE * LENGTH

# Simulation parameters
FORCE_MAG: float = 10.0
GRAVITY: float = 9.8
TAU: float = 0.02  # Time step (e.g., 50 FPS)
FOUR_THIRDS: float = 4.0 / 3.0

class MTorchCartpole:
    """
    Represents the CartPole environment simulation developed using Torch and support multiple envs.
    """
    def __init__(self, device: torch.device, n_envs: int = 1):
        self._device = device
        self._n_envs = n_envs
        # state represents x, x_dot, theta, theta_dot
        self._state = torch.zeros(4, dtype=torch.float32, device=self._device)
        self._G = torch.tensor(GRAVITY, device=device, dtype=torch.float32)
        self._TOTAL_MASS = torch.tensor(TOTAL_MASS, device=device, dtype=torch.float32)
        self._POLEMASS_LENGTH = torch.tensor(POLEMASS_LENGTH, device=device, dtype=torch.float32)
        self._LENGTH = torch.tensor(LENGTH, device=device, dtype=torch.float32)
        self._FOUR_THIRDS = torch.tensor(FOUR_THIRDS, device=device, dtype=torch.float32)
        self._MASS_POLE = torch.tensor(MASS_POLE, device=device, dtype=torch.float32)
        self._TAU = torch.tensor(TAU, device=device, dtype=torch.float32)
        self._FORCE_MAG = torch.tensor([-FORCE_MAG, FORCE_MAG], device=device, dtype=torch.float32)

    def reset(self) -> Tensor:
        # Generates 4 float32 between -0.05 and 0.05 (rand() returns a number between 0 and 1)
        self._state = self.generate_random_tensor()
        return self._state #shape(n_envs, 4)
    
    def reset_done_elements(self, done: Tensor) -> None :
        randoms = self.generate_random_tensor()
        self._state = torch.where(done.unsqueeze(1), randoms, self._state) # reset here

    
    def apply_action(self, actions: Tensor) -> Tensor: #action shape(n_envs)
        x, x_dot, theta, theta_dot = self._state.T #transpose enables unpacking
        force = self._FORCE_MAG[actions]
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        temp = (force + self._POLEMASS_LENGTH * theta_dot.pow(2) * sin_theta) / self._TOTAL_MASS
        denom = self._LENGTH * (self._FOUR_THIRDS - self._MASS_POLE * cos_theta.pow(2) / self._TOTAL_MASS)
        theta_acc = (self._G * sin_theta - cos_theta * temp) / denom
        x_acc = temp - self._POLEMASS_LENGTH * theta_acc * cos_theta / self._TOTAL_MASS

        # self._state[0] += self._TAU * x_dot     # x += TAU * x_dot
        # self._state[1] += self._TAU * x_acc     # x_dot += TAU * x_acc
        # self._state[2] += self._TAU * theta_dot # theta += TAU * theta_dot
        # self._state[3] += self._TAU * theta_acc # theta_dot += TAU * theta_acc
        updates = torch.stack((x_dot, x_acc.squeeze().squeeze(), theta_dot, theta_acc.squeeze().squeeze()))
        updates.mul_(self._TAU)
        self._state.add_(updates.T)
        return self._state #shape[n_envs, 4]
    
    def generate_random_tensor(self):
        # Generates 4 float32 between -0.05 and 0.05 (rand() returns a number between 0 and 1)
        return torch.rand((self._n_envs, 4), device=self._device).sub_(.5).mul_(.1)

