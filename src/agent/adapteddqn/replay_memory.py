import random
from collections import namedtuple, deque
import torch

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'terminated'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self._memory = deque([], maxlen=capacity)

    def push(self, state, action, next_state, reward, terminated):
        self._memory.append(Transition(state, action, next_state, reward, terminated))

    def sample(self, batch_size):
        gpu_indices = torch.randperm(len(self._memory))[:batch_size]
        deque_list = list(self._memory)
        sampled_data = [deque_list[i] for i  in gpu_indices.tolist()]
        return sampled_data

    def __len__(self):
        return len(self._memory)