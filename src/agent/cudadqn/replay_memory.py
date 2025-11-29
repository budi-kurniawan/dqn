from collections import namedtuple, deque
import torch

class ReplayMemory(object):

    def __init__(self, device: torch.device, capacity: int) -> None:
        self.device = device
        # self.capacity can probably be just an int
        self.capacity = torch.tensor(capacity, dtype=torch.int32, device=device)
        length = 4 + 1 + 4 + 1 + 1
        self.pointer = torch.tensor(0, dtype=torch.int32, device=device)
        self.size = torch.tensor(0, dtype=torch.int32, device=device)
        self.memory = torch.zeros((capacity, length), dtype=torch.float32, device=device) 

    def push(self, row) -> None:
        self.memory[self.pointer] = row
        # by using two lines, both operations are in-place and pointer stays on GPU
        self.pointer.add_(1)
        self.pointer %= self.capacity
        self.size.add_( (self.size < self.capacity).to(self.size.dtype) )


    def sample(self, batch_size):
        allowed_batch_size = torch.minimum(torch.tensor(batch_size, device=self.device), self.size)
        # inefficient because it generates self.size numbers and ony use batch_size numbers
        # indices = torch.randperm(self.size)[:self.batch_size]
        weights = torch.ones((self.size,), device=self.device, dtype=torch.float32)
        indices = torch.multinomial(weights, allowed_batch_size, replacement=False)
        return self.memory[indices]
