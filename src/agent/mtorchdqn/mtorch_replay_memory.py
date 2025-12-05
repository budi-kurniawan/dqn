import torch

class MTorchReplayMemory:


    def __init__(self, device: torch.device, capacity: int, row_length:int, n_envs: int):
        self._device = device
        self._capacity = capacity # must be int to keep sample() GPU-safe
        #length = 4 + 1 + 4 + 1 + 1  # state, action, next_state, reward, terminated
        self._length = row_length
        self._n_envs = n_envs
        self._pointer = torch.tensor(0, dtype=torch.int32, device=device)
        self._size = torch.tensor(0, dtype=torch.int32, device=device)
        self._memory = torch.zeros((capacity, row_length * n_envs), dtype=torch.float32, device=device) 

    def push(self, state, action, next_state, reward, terminated_float):
        # state, next_state: shape(n_envs,4)
        # reward, action, terminated: shape(n_envs)
        # 1-dim tensors need to be unsqueezed to make them 2-dim
        row = torch.cat((state, action.unsqueeze(1), next_state, reward.unsqueeze(1), 
                         terminated_float.unsqueeze(1)), dim=1)
        self._memory[self._pointer] = row.view(1, self._length * self._n_envs)
        # by using two lines, both operations are in-place and pointer stays on GPU
        self._pointer.add_(1)
        self._pointer %= self._capacity
        self._size.add_( (self._size < self._capacity).to(self._size.dtype) )

    def sample(self, batch_size):
        gpu_indices = torch.randperm(self._size)[:batch_size]
        return self._memory[gpu_indices]

    def __len__(self):
        return self._size