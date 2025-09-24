import gymnasium as gym
import random
from itertools import count

import torch
from dqn_agent import DQNAgent
import os
if os.name == 'nt':
    os.system('cls')
else:
    os.system('clear')

env = gym.make("CartPole-v1")

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

seed = 42
random.seed(seed)
torch.manual_seed(seed)
env.reset(seed=seed)
env.action_space.seed(seed)
env.observation_space.seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    
n_actions = env.action_space.n
state, info = env.reset()
n_observations = len(state)
print("dqn learning v1.288")

dqn_agent = DQNAgent(n_observations, n_actions)


episode_durations = []
num_episodes = 2

# state_batch: tensor([[-0.0413, -0.0046, -0.0054,  0.0348],
#         [-0.0414,  0.1906, -0.0047, -0.2596]])
# state_action_values: tensor([[107.0216],
#         [103.0867]], grad_fn=<GatherBackward0>)

# my_state = torch.tensor([[-0.0413, -0.0046, -0.0054,  0.0348]])
# print('my_state:', my_state)
# print("num_episode:", num_episodes)
# with torch.no_grad():
#         policy_net.layer1.weight.fill_(0.1) # Set all weights to 0.1
#         policy_net.layer2.weight.fill_(0.2) # Set all weights to 0.1
#         policy_net.layer3.weight.fill_(0.3) # Set all weights to 0.1
# q = policy_net(my_state)
# print('q:', q) #tensor([[ 0.0032, -0.0477]], grad_fn=<AddmmBackward0>)
# for name, param in policy_net.named_parameters():
#     print(f"Parameter Name: {name}")
#     print(f"Parameter Data:\n{param.data}")
#     print("-" * 30)



for i_episode in range(num_episodes):
    observation, info = env.reset() # observation is tuple [4]
    state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0) #state tensor [1,4]
    for t in count():
        action = dqn_agent.select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        reward = torch.tensor([reward], device=device)
        dqn_agent.update(state, action, next_state, reward, terminated, truncated)
        state = next_state
        if terminated or truncated:
            episode_durations.append(t + 1)
            print("episode ", i_episode, ", reward: ", t)
            break
