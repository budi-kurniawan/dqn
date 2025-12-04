import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
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
        self._n_actions = n_actions
        self._policy_net = DQN(n_observations, n_actions).to(device)
        self._target_net = DQN(n_observations, n_actions).to(device)
        self._target_net.load_state_dict(self._policy_net.state_dict())
        # with seed 42, setting amsgrad=True improves the results
        self._optimizer = optim.AdamW(self._policy_net.parameters(), lr=LR, amsgrad=True)
        self._memory = CudaReplayMemory(device, 10000)
        self._steps_done = torch.tensor(0, device=device)


    def select_action(self, state: Tensor) -> Tensor:
        state = state.unsqueeze(0) # convert shape(4) to (1,4)
        exponent_term = torch.exp(-1. * self._steps_done / EPS_DECAY)
        eps_threshold = EPS_END + (EPS_START - EPS_END) * exponent_term # shape([])
        self._steps_done.add_(1)
        sample = torch.rand(1, device=self._device).squeeze() #shape([])
        with torch.no_grad():
            greedy_action = self._policy_net(state).max(1).indices #shape(1)
        random_action = self._env.action_space.sample() #shape(1)
        return torch.where(sample > eps_threshold, greedy_action, random_action).flatten()


    def update(self, state, action, next_state, reward, terminated: Tensor, truncated: Tensor):
        # next_state: shape([4])
        # reward: shape([1])
        # terminated.view(1): shape([1])
        # truncated.view(1): shape([1])
        state = torch.tensor(state, dtype=torch.float32, device=self._device).unsqueeze(0) #convert to shape[1,4]
        next_state = torch.tensor(next_state, dtype=torch.float32, device=self._device).unsqueeze(0) #to shape[1,4]

        self._memory.push(state, action, next_state, reward, terminated.float())

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

        samples = memory.sample(BATCH_SIZE)
        state_batch = samples[:, 0:4] #shape(BATCH_SIZE, 4)
        action_batch = samples[:, 4].int().unsqueeze(1) #Shape(BATCH_SIZE, 1)
        reward_batch = samples[:, 9] #shape(BATCH_SIZE)
        terminated_batch = samples[:, 10] 
        next_state_batch = samples[:, 5:9] #shape(BATCH_SIZE, 4)

        non_final_mask = terminated_batch == 0 #Shape(128) of bools
        non_final_next_states = next_state_batch[non_final_mask] # #shape([x, 4]), x <= BATCH_SIZE

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = policy_net(state_batch).gather(1, action_batch) #shape(BATCH_SIZE, 1)
        # policy_net(state_batch) shape(BATCH_SIZE, n_actions)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=self._device) #shape(BATCH_SIZE)
        with torch.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
            # non_final_next_states.size = next_state_values.size - non_final_mask(False)
            # so, next_state_values[x] = 0.00 where non_final_mask[x] = False
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch #shape(BATCH_SIZE)


        # Compute Huber loss
        loss = CudaDQNAgent.criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()

    def get_policy_net(self):
        return self._policy_net
    
    def get_target_net(self):
        return self._target_net
