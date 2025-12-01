import unittest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import torch
from agent.dqn.dqn import DQN

class TestCudaDQNMethods(unittest.TestCase):
    def test_cuda_dqn(self):
        policy_net = DQN(4, 2)
        input = torch.tensor([0.1, -.2, .2, .2])
        with torch.no_grad():
            #greedy_action = self._policy_net(state).max(1).indices.view(1, 1)
            output = policy_net(input)
            print("output:", output)





if __name__ == '__main__':
    import os
    if os.name == 'nt':
        os.system('cls')
    # For macOS / Linux
    else:
        os.system('clear')
    unittest.main()