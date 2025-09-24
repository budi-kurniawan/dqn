import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from nn_util import save_params_as_torchscript
from dqn import DQN

class TestMainMethods(unittest.TestCase):
    def test_dqn(self):
        model = DQN(4, 2)
        with torch.no_grad():
            model.layer1.weight.fill_(0.1)
            model.layer1.bias.fill_(0.11)
            model.layer2.weight.fill_(0.2)
            model.layer2.bias.fill_(0.21)
            model.layer3.weight.fill_(0.3)
            model.layer3.bias.fill_(0.31)

        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        criterion = nn.SmoothL1Loss()

        # Dummy input & target
        input = torch.tensor([[0.1, 0.2, 0.3, 0.4], [-0.1, 0.2, 0.3, 0.4]])
        target = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

        # Forward
        output = model(input)
        print("result:", output)
        loss = criterion(output, target)
        print("loss:", loss)
        save_path = Path("D:/Downloads/model-params0.pt")
        save_params_as_torchscript(model, save_path)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 100)
        optimizer.step()        #
        #for name, param in model.named_parameters():
        #     print(f"name:{name}:\ngrad:{param.grad}\n")
        for name, param in model.named_parameters():
            print(f"name:{name}:\nvalues:{param.data}\n")

        # Represent a path
        save_path = Path("D:/Downloads/model-params.pt")
        save_params_as_torchscript(model, save_path)





if __name__ == '__main__':
    import os
    if os.name == 'nt':
        os.system('cls')
    # For macOS / Linux
    else:
        os.system('clear')
    unittest.main()