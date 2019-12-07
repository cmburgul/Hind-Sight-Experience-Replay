import numpy as np
import random
from collections import namedtuple, deque
import torch
import torch.nn.functional as F
import torch.optim as optim
from model import QNetwork
from replay_buffer import ReplayBuffer
from her import HER

BUFFER_SIZE = int(1e-5) # replay buffer size
BATCH_SIZE = 64         # mini-batch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 0.0001             # learning rate
UPDATE_EVERY = 4        # how often update the network

device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 
class Agent:
    """ Interacts with and learns with the environment """
    # DQN + HER    
    def __init__(self, state_size, action_size, seed):
        """
        Params
        ====
            state_size (int) : dimensions of each state
            action_size (int) : dimensions of each action
            seed (int) : random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.N = self.state_size/2
        self.update_target_step = 1000
        self.step_counter = 0
        
        # Q-Network 
        self.qnetwork_local = QNetwork(self.state_size, self.action_size, seed).to(device)  # local model
        self.qnetwork_target = QNetwork(self.state_size, self.action_size, seed).to(device) # target model                          # target model
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)                # optimizer

        # Replay Buffer - to store experiences 
        self.buffer = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)
        
        # HER -Hindsight Experience Replay Buffer - to modify goal to a virtual goal
        self.her = HER(self.N)
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.update_every = 1000  # For hard update
        #self.update_every = 4     # For Soft update
        
        # Epsilon # Need to define in main rather than here
        self.epsilon = 0.1
        self.eps_max = 0.99
        self.eps_min = 0.05
        self.eps = self.eps_max # start eps from eps_max
        
    def act(self, state, eps):
        """ Returns action for given state as per current policy
        Params
        ======
            state (array_like): current state
            eps (float) : epsilon, for epsilon-greedy action selection
        """
        self.qnetwork_local.eval()  # evaluation mode
        with torch.no_grad():
            Q = self.qnetwork_local(state.to(device))
        self.qnetwork_local.train() # training mode
        
        rand_num = np.random.rand()
        if (rand_num < eps):  # Exploration
            a = torch.randint(0, Q.shape[1], (1,)).type(torch.LongTensor)
        else:                 # Exploitation
            a = torch.argmax(Q, dim=1)
        return a
    
    def step(self, s, a, r, s_, done):
        # Save experience in replay buffer 
        self.buffer.add(s, a, r, s_, done)
        
        # Save experience in her buffer
        self.her.add(s, a, r, s_, done)
        
        if len(self.buffer.memory) > BATCH_SIZE:
            # Learning every time step and update target model every update_every step  
            loss = self.learn()
            
            
    def learn(self):  
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        num = len(self.buffer.memory)    # Check the length of replay buffer
        
        states, actions, rewards, next_states, dones = self.buffer.sample(BATCH_SIZE)
        
        # Get max predicted Q-values (for next_states) from target model
        Q_targets_next = self.qnetwork_target(next_states.to(device)).detach().max(1)[0].unsqueeze(1)
        
        # Compute Qtargets for current states
        Q_targets = rewards.to(device) + (GAMMA*Q_targets_next * (1-dones.to(device)))
        
        # Get expected Q-values from local model
        Q_expected = self.qnetwork_local(states.to(device)).gather(1, actions.to(device))
        
        # learn
        loss = F.smooth_l1_loss(Q_expected.squeeze(),Q_targets.squeeze())
        #loss = F.mse_loss(Q_expected, Q_targets)
        
        # optimizer
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # ------------------- update target network ------------------- #
        #self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)  
                
        self.step_counter += 1
        if (self.step_counter > self.update_target_step):
            self.hard_update(self.qnetwork_local, self.qnetwork_target)
            self.step_counter = 0
        
        
        
    def hard_update():    
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
        
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
    def her_update(self):
        her_buffer = self.her.update()
        for e in her_buffer:
            self.buffer.memory.append(e)