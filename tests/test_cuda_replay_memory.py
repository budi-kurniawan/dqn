import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import unittest

import torch
import torch.nn as nn
from agent.cudadqn.replay_memory import ReplayMemory
import random



class TestPrintDQNParams(unittest.TestCase):
    def test_cuda_replay_memory(self):
        n_observations = 4
        n_actions = 2
        device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )
        capacity = 100
        seed = 42
        #random.seed(seed)
        #torch.manual_seed(seed)

        replay_memory = ReplayMemory(device, capacity)
        # replay_memory.push(torch.ones((11)))
        # print("size:", replay_memory.size)

        # row = replay_memory.sample(20)
        # print(row)
        print("--------------")

        for i in range(200):
            t = torch.full((1, 11), i)
            replay_memory.push(t)

        row = replay_memory.sample(3)
        print(row)





if __name__ == '__main__':
    import os
    if os.name == 'nt':
        os.system('cls')
    # For macOS / Linux
    else:
        os.system('clear')
    unittest.main()