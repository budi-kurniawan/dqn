import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from util.nn_util import save_params_as_torchscript, save_params_as_onnx
from dqn import DQN
from dqn_agent import DQNAgent
import numpy as np
import random

class TestDQNAgent(unittest.TestCase):
    def test_dqn_agent(self):
        n_observations = 4
        n_actions = 2
        device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )
        seed = 42
        random.seed(seed)
        torch.manual_seed(seed)


        dqn_agent = DQNAgent(n_observations, n_actions)
        policy_net = dqn_agent.get_policy_net()
        target_net = dqn_agent.get_target_net()
        # save_params_as_onnx(policy_net, Path("D:/Downloads/policy_net.onnx"))


        # with torch.no_grad():
        #     policy_net.layer1.weight.fill_(0.1)
        #     policy_net.layer1.bias.fill_(0.11)
        #     policy_net.layer2.weight.fill_(0.2)
        #     policy_net.layer2.bias.fill_(0.21)
        #     policy_net.layer3.weight.fill_(0.3)
        #     policy_net.layer3.bias.fill_(0.31)

        #     target_net.layer1.weight.fill_(0.1)
        #     target_net.layer1.bias.fill_(0.11)
        #     target_net.layer2.weight.fill_(0.2)
        #     target_net.layer2.bias.fill_(0.21)
        #     target_net.layer3.weight.fill_(0.3)
        #     target_net.layer3.bias.fill_(0.31)



        episode_durations = []
        num_episodes = 2

        # state_batch: tensor([[-0.0413, -0.0046, -0.0054,  0.0348],
        #         [-0.0414,  0.1906, -0.0047, -0.2596]])
        # state_action_values: tensor([[107.0216],
        #         [103.0867]], grad_fn=<GatherBackward0>)

        # my_state = torch.tensor([[-0.0413, -0.0046, -0.0054,  0.0348]])
        # print('my_state:', my_state)
        # print("num_episode:", num_episodes)
        # q = policy_net(my_state)
        # print('q:', q) #tensor([[ 0.0032, -0.0477]], grad_fn=<AddmmBackward0>)
        # for name, param in policy_net.named_parameters():
        #     print(f"Parameter Name: {name}")
        #     print(f"Parameter Data:\n{param.data}")
        #     print("-" * 30)

        state = torch.tensor([.5, .6, .7, 1], dtype=torch.float32, device=device).unsqueeze(0) 
        next_state = torch.tensor([1.5, 1.6, 1.7, 2], dtype=torch.float32, device=device).unsqueeze(0) 


        for i in range(100):
            print("i = ", i)
            action = np.int64(i % 2)
            action = np.int64(0)
            action = torch.tensor([[action]], device=device, dtype=torch.long)
            reward = 1 #1 if i % 10 == 0 else 0
            reward = torch.tensor([reward], device=device)
            terminated = False #i % 3 == 0
            truncated = False 
            dqn_agent.update(state, action, next_state, reward, terminated, truncated)


        print("\npolicy net after update")
        for name, param in policy_net.named_parameters():
            print(f"name:{name}:\nvalues:{param.data}\n")

        print("\ntarget net after update")
        for name, param in target_net.named_parameters():
            print(f"name:{name}:\nvalues:{param.data}\n")




if __name__ == '__main__':
    import os
    if os.name == 'nt':
        os.system('cls')
    # For macOS / Linux
    else:
        os.system('clear')
    unittest.main()