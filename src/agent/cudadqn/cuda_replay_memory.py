from collections import namedtuple, deque
import torch

class CudaReplayMemory(object):

    def __init__(self, device: torch.device, capacity: int) -> None:
        self._device = device
        # self.capacity can probably be just an int
        self._capacity = capacity # must be int to keep sample() GPU-safe
        length = 4 + 1 + 4 + 1 + 1 + 1 # state, action, next_state, reward, terminated, my FLAG
        self._pointer = torch.tensor(0, dtype=torch.int32, device=device)
        self._size = torch.tensor(0, dtype=torch.int32, device=device)
        self._memory = torch.zeros((capacity, length), dtype=torch.float32, device=device) 
        self._flag = torch.tensor(1.0, device=device, dtype=torch.float32)

    def push(self, state, action, next_state, reward, terminated_float) -> None:
        # flag is used in sample()
        row = torch.cat((state, action, next_state, reward, terminated_float, self._flag))
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
    
    def size(self):
        return self._size
