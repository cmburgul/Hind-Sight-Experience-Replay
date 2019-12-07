from collections import deque, namedtuple
import numpy as np
from copy import deepcopy as dc
import random
from recordtype import *

class HER:
    """ To modify experiences in a single episode """
    def __init__(self, N):
        self.N = int(N)
        self.experience = recordtype("Experience", field_names=["s", "a", "r", "s_", "done"])
        self.buffer = deque()
        
    def reset(self):
        self.buffer = deque()
        
    def add(self, state, action, reward, next_state, done):
        """ Add a new experience to memory """
        e = self.experience(state, action, reward, next_state, done)
        self.buffer.append(e)
        
    def update(self):
        """
        Updating the virtual goals as real goals in all the experiences of an episode
        """
        her_buffer = dc(self.buffer)
        goal = her_buffer[-1].s_[0,0:self.N] # Taking s_ from last experience 
        for i in range(len(her_buffer)):
            her_buffer[i].s[0,self.N:] = goal  # Modify s
            her_buffer[i].s_[0,self.N:] = goal # Modify s_
            her_buffer[i].r = -1      # Modify r
            her_buffer[i].done = False    # Modify done
            if ((her_buffer[i].s_[0,0:5] - goal).abs().sum() == 0): # S_(state == goal)
                her_buffer[i].done = True
                her_buffer[i].r = 0.0
        return her_buffer