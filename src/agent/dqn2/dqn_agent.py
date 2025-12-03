import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import math
import numpy as np
from agent.dqn.dqn import DQN
from agent.dqn2.replay_memory import ReplayMemory, Transition


BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 2500
TAU = 0.005
LR = 3e-4


class DQNAgent:
    criterion = nn.SmoothL1Loss()

    def __init__(self, n_observations, n_actions, env, device):
        self._env = env
        self._device = device
        self._n_observations = n_observations
        self._n_actions = n_actions
        self._policy_net = DQN(n_observations, n_actions).to(device)
        self._target_net = DQN(n_observations, n_actions).to(device)
        self._target_net.load_state_dict(self._policy_net.state_dict())
        # with seed 42, setting amsgrad=True improves the results and make it reproducible
        self._optimizer = optim.AdamW(self._policy_net.parameters(), lr=LR, amsgrad=True)
        self._memory = ReplayMemory(10000)
        self._steps_done = 0


    def select_action(self, observation):
        state = torch.tensor(observation, dtype=torch.float, device=self._device).unsqueeze(0) #state tensor [1,4]
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self._steps_done / EPS_DECAY)
        self._steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self._policy_net(state).max(1).indices.view(1, 1).item()
        else:
            return torch.tensor([[self._env.action_space.sample()]], device=self._device, dtype=torch.long).item()


    def update(self, state: np.ndarray, action: int, next_state: np.ndarray, reward: float, 
               terminated: bool, truncated: bool):
        if terminated:
            self._memory.push(state, action, None, reward, terminated)
        else:
            self._memory.push(state, action, next_state, reward, terminated)

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
        if len(memory) < BATCH_SIZE:
            return
        policy_net = self._policy_net
        target_net = self._target_net
        optimizer = self._optimizer

        transitions = memory.sample(BATCH_SIZE)

        states = np.zeros((BATCH_SIZE, self._n_observations), dtype=np.float32)
        next_states = np.zeros((BATCH_SIZE, self._n_observations), dtype=np.float32)
        actions = np.zeros((BATCH_SIZE, 1), dtype=int)
        rewards = np.zeros((BATCH_SIZE), dtype=np.float32)
        terminals = np.zeros((BATCH_SIZE), dtype=bool)

        for i, transition in enumerate(transitions):
            states[i] = transition.state
            next_states[i] = transition.next_state
            actions[i] = transition.action
            rewards[i] = transition.reward
            terminals[i] = transition.terminal

        states_tensor = torch.from_numpy(states).to(self._device)
        next_states_tensor = torch.from_numpy(next_states).to(self._device)
        actions_tensor = torch.from_numpy(actions).to(self._device)
        rewards_tensor = torch.from_numpy(rewards).to(self._device)
        terminals_tensor = torch.from_numpy(terminals).to(self._device)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = policy_net(states_tensor).gather(1, actions_tensor) #shape(BATCH_SIZE, 1)
        next_state_values = target_net(next_states_tensor).max(1).values #shape(BATCH_SIZE)
        nz = terminals_tensor.nonzero(as_tuple=False)
        nz_size = nz.size()
        if nz_size[0] == 1:
            row_indices = nz[0]
        else:
            row_indices = nz.squeeze()

        zeros_tensor = torch.zeros(nz_size[0], dtype=torch.float32, device=self._device)
        next_state_values.index_put_((row_indices,), zeros_tensor)
        expected_state_action_values = (next_state_values * GAMMA) + rewards_tensor #shape(BATCH_SIZE)

        # Compute Huber loss
        loss = DQNAgent.criterion(state_action_values, expected_state_action_values.unsqueeze(1))

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
