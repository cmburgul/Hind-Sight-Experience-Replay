import torch
from copy import deepcopy as dc

class BfEnv:
    def __init__(self, N):
        self.N = N    # Number of bits
        self.state_size = self.N*2
        self.action_size = self.N
        
    def reset(self):
        # Input : None
        # Output : state
        # Randomly generates a state tuple 
        state = torch.rand((1,self.N)).round()
        goal = torch.rand((1,self.N)).round()
        done = False
        return torch.cat((state, goal), dim=1), done
    
    def step(self, s, action):
        # Inputs: s, action
        # s : state, action : index number of action to be taken 
        # Output : (next_state, reward, done, dist)
        s[0, action] = 1.0 - s[0, action] #Taking action(a) and getting next_state(s_)
        r = -1.0 # Sparse reward penalty
        done = False
        
        # Calculate distance
        dist = (s[0,0:self.N] - s[0,self.N:]).abs().sum() 
        
        if dist == 0:
            done = True
            r = 0.0 # Sparse reward 
        return s, r, done, dist     