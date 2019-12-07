from collections import deque, namedtuple
import numpy as np
from copy import deepcopy
import random
from recordtype import *
import torch

class ReplayBuffer:
    """ Fixed size buffer to store experience tuple """
    
    def __init__(self, buffer_size, batch_size, seed = 0):
        """
        Params
        ====
            buffer_size(int) : maximum size of a buffer
            batch_size(int) : size of each training batch
            seed (int) : random seed
        """
        self.memory = deque() # deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = recordtype("Experience", field_names=["s", "a", "r", "s_", "done"])
        self.seed = random.seed(seed)
        
    def add(self, s, a, r, s_, done):
        """ Add a new experience to memory """
        e = self.experience(s, a, r, s_, done)
        self.memory.append(e)
    
    def sample(self, K):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, K)

        states = torch.cat([e.s for e in experiences])
        actions = torch.cat([e.a for e in experiences]).view(K,-1)
        rewards = torch.tensor([e.r for e in experiences]).view(K, -1)
        next_states = torch.cat([e.s_ for e in experiences])
        dones = torch.tensor([e.done for e in experiences]).float().view(K,-1)
  
        return (states, actions, rewards, next_states, dones)     

    def _len__(self):
        return len(self.memory)