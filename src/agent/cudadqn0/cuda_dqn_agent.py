import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from agent.dqn.dqn import DQN
from agent.cudadqn.cuda_replay_memory import CudaReplayMemory


BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 2500
TAU = 0.005
LR = 3e-4


class CudaDQNAgent:
    criterion = nn.SmoothL1Loss()

    def __init__(self, n_observations, n_actions, env, device):
        self._env = env
        self._device = device
        self._n_observations = n_observations
        self._n_actions = n_actions
        self._policy_net = DQN(n_observations, n_actions).to(device)
        self._target_net = DQN(n_observations, n_actions).to(device)
        self._target_net.load_state_dict(self._policy_net.state_dict())
        self._optimizer = optim.AdamW(self._policy_net.parameters(), lr=LR, amsgrad=True)
        self._memory = CudaReplayMemory(device, 10000)
        self._steps_done = torch.tensor(0, device=device, dtype=torch.int32)


    def select_action(self, state: Tensor) -> Tensor:
        state = state.unsqueeze(0) # convert shape(4) to (1,4)
        #sample = random.random()
        sample = torch.rand(1, device=self._device).squeeze() #shape([])
        # eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        #     math.exp(-1. * self._steps_done / EPS_DECAY)
        exponent_term = torch.exp(-1. * self._steps_done / EPS_DECAY)
        eps_threshold = EPS_END + (EPS_START - EPS_END) * exponent_term # shape([])
        self._steps_done.add_(1)
        with torch.no_grad():
            greedy_action = self._policy_net(state).max(1).indices #shape(1)
        random_action = self._env.action_space.sample() #shape(1)
        return torch.where(sample > eps_threshold, greedy_action, random_action).flatten()


    def update(self, state: Tensor, action: Tensor, next_state: Tensor, reward: Tensor, 
               terminated_float: Tensor, truncated_float: Tensor):
        memory = self._memory
        memory.push(state, action, next_state, reward, terminated_float)

        self.optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = self._target_net.state_dict()
        policy_net_state_dict = self._policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        self._target_net.load_state_dict(target_net_state_dict)
    
    def optimize_model(self):
        memory = self._memory
        if memory._size < BATCH_SIZE:
            return
        policy_net = self._policy_net
        target_net = self._target_net
        optimizer = self._optimizer

        transitions = memory.sample(BATCH_SIZE) #shape(BATCH_SIZE, 12)
        states_tensor = transitions[: , 0:4] #shape(BATCH_SIZE, 4)
        actions_tensor = transitions[: , 4].int().unsqueeze(1) #shape(BATCH_SIZE, 1) 
        next_states_tensor = transitions[: , 5:9] #shape(BATCH_SIZE, 4)
        rewards_tensor = transitions[: , 9] #shape(BATCH_SIZE)
        terminals_float_tensor = transitions[: , 10] #shape(BATCH_SIZE)

        non_final_mask = (1 - terminals_float_tensor).bool()       # non_final_mask.shape = Size(128) of bools
        non_final_next_states = next_states_tensor[non_final_mask] #shape([x, 4]), x <= BATCH_SIZE

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = policy_net(states_tensor).gather(1, actions_tensor) #shape(BATCH_SIZE, 1)


        # next_state_values = target_net(next_states_tensor).max(1).values #shape(BATCH_SIZE)
        # terminals_bool_tensor = terminals_float_tensor != 0
        # nz = terminals_bool_tensor.nonzero(as_tuple=False)
        # nz_size = nz.size()
        # if nz_size[0] == 1:
        #     row_indices = nz[0]
        # else:
        #     row_indices = nz.squeeze()

        # zeros_tensor = torch.zeros(nz_size[0], dtype=torch.float32, device=self._device)
        # next_state_values.index_put_((row_indices,), zeros_tensor)



        next_state_values = torch.zeros(BATCH_SIZE, device=self._device) #shape(BATCH_SIZE)
        with torch.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values

        expected_state_action_values = (next_state_values * GAMMA) + rewards_tensor #shape(BATCH_SIZE)

        # Compute Huber loss
        loss = CudaDQNAgent.criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()

    def get_policy_net(self):
        return self._policy_net
    
    def get_target_net(self):
        return self._target_net
