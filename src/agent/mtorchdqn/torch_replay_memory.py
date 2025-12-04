import torch

class TorchReplayMemory:
    def __init__(self, device: torch.device, capacity: int):
        self._device = device
        self._capacity = capacity # must be int to keep sample() GPU-safe
        length = 4 + 1 + 4 + 1 + 1  # state, action, next_state, reward, terminated
        self._pointer = torch.tensor(0, dtype=torch.int32, device=device)
        self._size = torch.tensor(0, dtype=torch.int32, device=device)
        self._memory = torch.zeros((capacity, length), dtype=torch.float32, device=device) 
        #self._flag = torch.tensor([1.0], device=device, dtype=torch.float32)

    def push(self, state, action, next_state, reward, terminated_float):
        row = torch.cat((state.squeeze(), action, next_state.squeeze(), reward, terminated_float))
        self._memory[self._pointer] = row
        # by using two lines, both operations are in-place and pointer stays on GPU
        self._pointer.add_(1)
        self._pointer %= self._capacity
        self._size.add_( (self._size < self._capacity).to(self._size.dtype) )

    def sample(self, batch_size):
        gpu_indices = torch.randperm(self._size)[:batch_size]
        return self._memory[gpu_indices]

    def __len__(self):
        return self._size