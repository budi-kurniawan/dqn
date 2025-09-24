import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import math
import numpy as np

from dqn import DQN
from replay_memory import ReplayMemory, Transition


BATCH_SIZE = 2 #128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 2500
TAU = 0.005
LR = 3e-4


class DQNAgent:
    criterion = nn.SmoothL1Loss()

    def __init__(self, n_observations, n_actions):
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )
        self._n_actions = n_actions
        self._policy_net = DQN(n_observations, n_actions).to(self._device)
        self._target_net = DQN(n_observations, n_actions).to(self._device)
        self._target_net.load_state_dict(self._policy_net.state_dict())
        self._optimizer = optim.AdamW(self._policy_net.parameters(), lr=LR)
        self._memory = ReplayMemory(10000)
        self._steps_done = 0


    def select_action(self, state):
        self._steps_done += 1
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self._steps_done / EPS_DECAY)
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self._policy_net(state).max(1).indices.view(1, 1)
        else:
            # return torch.tensor([[self._env.action_space.sample()]], device=self._device, dtype=torch.long)
            random_action = np.random.randint(0, self._n_actions, dtype=np.int64)
            return torch.tensor([[random_action]], device=self._device, dtype=torch.long)


    def update(self, state, action, next_state, reward, terminated, truncated):
        if terminated:
            self._memory.push(state, action, None, reward)
        else:
            self._memory.push(state, action, next_state, reward)

        # Perform one step of the optimization (on the policy network)
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
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self._device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state) # convert a tuple (batch.state) to torch.Tensor
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        #print("state_batch.size:", state_batch.shape, ", nstate shape:", non_final_next_states.shape)


        # print("batch.state:", len(batch.state)) # a tuple, len() = 128
        # print("state_batch:", state_batch.shape) # a torch.Tensor Size([128, 4])
        

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = policy_net(state_batch).gather(1, action_batch)

        #print("batch.next_state:", batch.next_state)
        #print("non_final_mask:", non_final_mask.shape) # Size([128])
        # print("non_final_n_states:", non_final_next_states.shape) # shape([x, 4]), x <= BATCH_SIZE
        # print("\n\n")

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=self._device)
        with torch.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
            # non_final_next_states.size = next_state_values.size - non_final_mask(False)
            # so, next_state_values[x] = 0.00 where non_final_mask[x] = False
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # print("state_batch:", state_batch)
        # print("action_batch:", action_batch)
        # print('Q(state_batch):', policy_net(state_batch))
        # print('state_action_values:', state_action_values)
        # print("next_state_values:", next_state_values)
        # print("reward_batch:", reward_batch)
        # print("expected_state_action_values:", expected_state_action_values)
        # print('\n')

        # Compute Huber loss
        loss = DQNAgent.criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()
