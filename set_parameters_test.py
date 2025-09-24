import unittest

import torch
import torch.nn as nn

class MyBlock(nn.Module):
    def __init__(self):
        super(MyBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features=4, out_features=4),  # need input size here
            nn.ReLU(),
            nn.Linear(4, 2)
        )

    def forward(self, x):
        return self.block(x)

class TestMainMethods(unittest.TestCase):

    def test_block(self):
        model = MyBlock()
        with torch.no_grad():
            model.block[0].weight.fill_(0.1)
            model.block[0].bias.fill_(0.01)
            model.block[2].weight.fill_(0.2)
            model.block[2].bias.fill_(0.02)
        for name, param in model.named_parameters():
            print(f"{name}:\n{param.data}\n")

        input = torch.tensor([[0.1, 0.2, 0.3, 0.4], [-0.1, 0.2, 0.3, 0.4]])
        result = model(input)
        print("result:", result)
        self.assertEqual('foo'.upper(), 'FOO')

if __name__ == '__main__':
    unittest.main()