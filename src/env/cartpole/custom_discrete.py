import random
import numpy as np

class CustomDiscrete:
    """
    A minimal class that mimics the gymnasium.spaces.Discrete functionality 
    to support the .sample() method.
    """
    def __init__(self, n):
        """
        Initializes the space.
        Args:
            n (int): The number of possible actions (0, 1, ..., n-1).
        """
        self.n = n

    def sample(self):
        """
        Returns a randomly sampled action from the discrete space [0, n-1].
        """
        # random.randrange(n) returns a random integer k such that 0 <= k < n
        return random.randrange(self.n)