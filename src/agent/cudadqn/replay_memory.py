from collections import namedtuple, deque
import torch

class ReplayMemory(object):

    def __init__(self, device: torch.device, capacity: int) -> None:
        self._device = device
        # self.capacity can probably be just an int
        self._capacity = capacity #torch.tensor(capacity, dtype=torch.int32, device=device)
        length = 4 + 1 + 4 + 1 + 1 + 1 # state, action, next_state, reward, terminated, my FLAG
        self._pointer = torch.tensor(0, dtype=torch.int32, device=device)
        self._size = torch.tensor(0, dtype=torch.int32, device=device)
        self._memory = torch.zeros((capacity, length), dtype=torch.float32, device=device) 

    def push(self, row) -> None:
        self._memory[self._pointer] = row
        # by using two lines, both operations are in-place and pointer stays on GPU
        self._pointer.add_(1)
        self._pointer %= self._capacity
        self._size.add_( (self._size < self._capacity).to(self._size.dtype) )


    def sample(self, batch_size):
        gpu_indices = torch.randint(
            low=0, 
            high=self._capacity,
            size=(batch_size,), 
            device=self._device, 
            dtype=torch.long
        )
        sampled_data = self._memory[gpu_indices] # may include zeros when _memory not full yet
        mask = sampled_data[:, -1] != 0
        return sampled_data[mask]
