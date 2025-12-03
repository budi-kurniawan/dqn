from collections import namedtuple
import torch

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'terminated'))


class ReplayMemory(object):

    def __init__(self, device: torch.device, capacity: int):
        self._device = device
        self._capacity = capacity # must be int to keep sample() GPU-safe
        length = 4 + 1 + 4 + 1 + 1 + 1 # state, action, next_state, reward, terminated, my FLAG
        self._pointer = torch.tensor(0, dtype=torch.int32, device=device)
        self._size = torch.tensor(0, dtype=torch.int32, device=device)
        self._memory = torch.zeros((capacity, length), dtype=torch.float32, device=device) 
        self._flag = torch.tensor([1.0], device=device, dtype=torch.float32)

    def push(self, state, action, next_state, reward, terminated_float):
        # state = state.squeeze() # shape[1,4] to [4
        # next_state = next_state.squeeze() # shape[1,4] to [4]
        # reward = reward.float()
        # action = action.float()
        # print('state:', state.shape, state)
        # print('action:', action.shape, action)
        # print('next_s:', next_state.shape, next_state)
        # print('reward:', reward.shape, reward)
        # print('termianted:', terminated_float.shape, terminated_float)
        row = torch.cat((state.squeeze(), action, next_state.squeeze(), reward, terminated_float, self._flag))
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