from env import BfEnv
from agent import Agent
from utils import *
import numpy as np
from collections import deque

# Initialize env
env = BfEnv(5)

# Initialize agent
agent = Agent(10, 5, 0)

def dqn_her(num_episodes=5000, N=5, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """ Deep Q-Learning + HER
    
    Params
    ======
        num_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """

    log = logger()
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    log.add_log('scores')
    log.add_log('episodes_loss')
    log.add_log('final_dist')
    mean_loss = mean_val()
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, num_episodes+1):
        state, _ = env.reset()
        score = 0
        min_dist = N
        for t in range(N):
            action = agent.act(state, eps)
            next_state, reward, done, dist = env.step(state, action)
            
            if dist < min_dist:
                min_dist = dist
                
            if (t+1) == N:  # Breaking the episode after N number of time steps
                done = True
                 
            agent.step(state, action, reward, next_state, done)
            
            #if (loss != None): 
            #    mean_loss.append(loss) 
            state = next_state
            score += reward
            
            if done:
                break 
                
        agent.her_update()                # her update the experience trajectory
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        log.add_item('scores', score)     # save most recent score
        log.add_item('episodes_loss', mean_loss.get())
        log.add_item('final_dist', min_dist)
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}\tmin dist: {}'.format(i_episode, np.mean(scores_window), min_dist), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        
    return log

if __name__ == "__main__":

    log = dqn_her(num_episodes=5000, N=5, eps_start=1.0, eps_end=0.01, eps_decay=0.995)

    """
    # Plotting

    Y = np.asarray(log.get_log('scores'))
    Y2 = smooth(Y)
    x = np.linspace(0, len(Y), len(Y))
    fig1 = plt.figure()
    ax1 = plt.axes()
    ax1.plot(x, Y, Y2)
    plt.xlabel('episodes')
    plt.ylabel('episode return')

    Y = np.asarray(log.get_log('episodes_loss'))
    Y2 = smooth(Y)
    x = np.linspace(0, len(Y), len(Y))
    fig2 = plt.figure()
    ax2 = plt.axes()
    ax2.plot(x, Y, Y2)
    plt.xlabel('episodes')
    plt.ylabel('average loss')

    Y = np.asarray(log.get_log('final_dist'))
    Y2 = smooth(Y)
    x = np.linspace(0, len(Y), len(Y))
    fig3 = plt.figure()
    ax3 = plt.axes()
    ax3.plot(x, Y, Y2)
    plt.xlabel('episodes')
    plt.ylabel('minimum distance')

    Y = np.asarray(log.get_log('final_dist'))
    Y[Y > 1] = 1.0
    K = 100
    Z = Y.reshape(int(num_epochs/K),K)
    T = 1 - np.mean(Z,axis=1)
    x = np.linspace(0, len(T), len(T))*K
    fig4 = plt.figure()
    ax4 = plt.axes()
    ax4.plot(x, T)
    plt.xlabel('episodes')
    plt.ylabel('sucess rate')
    """