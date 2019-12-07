import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """ DQN (Policy) Network """
    def __init__(self, state_size, action_size, seed, fc1_units=128):
        """
        Initialize parameters and build model
        Parameters
        ====
            state_size (int): Dimension of state size
            action_size (int): Dimenstion of action size
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
            seed (int) : Random seed
        """ 
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, action_size)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.fc2(x)
        return x