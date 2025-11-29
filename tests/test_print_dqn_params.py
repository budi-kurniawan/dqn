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

class TestPrintDQNParams(unittest.TestCase):
    def test_print_dqn_params(self):
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

        for name, param in policy_net.named_parameters():
            print(f"name:{name}:\nvalues:{param.data}\n")
        save_path = Path("D:/Downloads/model-params-seed42.pt")
        save_params_as_torchscript(policy_net, save_path)





if __name__ == '__main__':
    import os
    if os.name == 'nt':
        os.system('cls')
    # For macOS / Linux
    else:
        os.system('clear')
    unittest.main()