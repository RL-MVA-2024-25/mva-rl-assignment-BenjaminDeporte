# -----------------------------------------------------------------------------
# --- CODE TO LEARN A DQN -----------------------------------------------------
# --- TO BE USED IN A DQN AGENT CLASS -----------------------------------------
# -----------------------------------------------------------------------------


# --- imports -----------------------------------------------------------------

import os, sys
# print(os.getcwd())
sys.path.append('/home/benjamin.deporte/MVA/mva-rl-assignment-BenjaminDeporte/src/')
sys.path.append('/home/benjamin.deporte/MVA/mva-rl-assignment-BenjaminDeporte/lib/')

from replay import ReplayBuffer2, get_replay_buffer
from sklearn.ensemble import RandomForestRegressor
from networks import DQN

import numpy as np

from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient

# import random
from tqdm import tqdm
from tqdm import trange
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
# import gymnasium as gym

import matplotlib.pyplot as plt

# --- should we create a random first replay buffer ? ------------------------------



# --- PICK GREEDY ACTION WRT NETWORK AND STATE

def greedy_action(network, state):
    device = "cuda" if next(network.parameters()).is_cuda else "cpu"
    with torch.no_grad():
        Q = network(torch.Tensor(state).unsqueeze(0).to(device))
        return torch.argmax(Q).item()


# --- DQN AGENT -----------------------------------------------------------------

class dqn_agent:
    
    dqn_inter1_savepath = '/home/benjamin.deporte/MVA/mva-rl-assignment-BenjaminDeporte/modeles/dqn_inter1.pth'
    dqn_inter2_savepath = '/home/benjamin.deporte/MVA/mva-rl-assignment-BenjaminDeporte/modeles/dqn_inter2.pth'
    save_delays = 20 # saves models every N episodes, just in case
    
    # constructor - takes a config file and a model
    def __init__(self, config, model):
        device = "cuda" if next(model.parameters()).is_cuda else "cpu"
        self.gamma = config['gamma']
        self.batch_size = config['batch_size']
        self.nb_actions = config['nb_actions']
        # implement an empty replay buffer
        self.memory = ReplayBuffer2(config['buffer_size'], device)
        self.epsilon_max = config['epsilon_max']
        self.epsilon_min = config['epsilon_min']
        self.epsilon_stop = config['epsilon_decay_period'] # counted in steps, ie episodes * 200. Time when epsilon = epsilon_min
        self.epsilon_delay = config['epsilon_delay_decay'] # time à partir duquel epsilon commence à décroître, compté en steps = episodes * 200
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
        self.model = model 
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])
    
    # one gradient step - 
    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)  # sample batch
            QYmax = self.model(Y).max(1)[0].detach()  # get max(q(s',a'))
            #update = torch.addcmul(R, self.gamma, 1-D, QYmax)
            # ---
            # torch.addcmul(input, tensor1, tensor2, *, value=1, out=None) → Tensor
            # Performs the element-wise multiplication of tensor1 by tensor2, multiplies the result by the scalar value and adds it to input.
            # output = input + value * (tensor1 (*) tensor2) 
            # --- 
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma) # compute R + (1-D) * gamma * QYmax, which is the update
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1)) # get current estimate q(X,A). gather ~ vlookup
            loss = self.criterion(QXA, update.unsqueeze(1)) # unsqueeze fait passer la shape de B à B,1
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 
    
    # train the network for max_episodes
    def train(self, env, max_episode):
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0

        while episode < max_episode:
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)

            # select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = greedy_action(self.model, state)

            # step
            next_state, reward, done, trunc, _ = env.step(action)
            # fill the replay buffer wrt action taken (epsilon-greedy)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward

            # train
            self.gradient_step()

            # next transition
            step += 1
            print(f"Currently at episode {episode+1}/{max_episode}, total {step} steps done", end="\r")
            if done or trunc:
                # end of episode, report out
                episode += 1
                print(f"Episode {episode:5d}, epsilon {epsilon:6.2f}, buffer size {len(self.memory):6d}, episode return {episode_cum_reward:.1e}")
                state, _ = env.reset()
                episode_return.append(episode_cum_reward)
                episode_cum_reward = 0
                
                # save intermediate models at regular intervals, in case IRT's GPU kicks me out
                if episode % self.save_delays == 0:
                    print(f"saving intermediate DQN model as inter1")
                    torch.save(self.model.state_dict(), self.dqn_inter1_savepath)
                if episode % self.save_delays == 1:
                    print(f"saving intermediate DQN model as inter2")
                    torch.save(self.model.state_dict(), self.dqn_inter2_savepath)
            else:
                state = next_state
            
        return episode_return
    
    
# ------------------------------------------------------------------------------------------
# --- TRAIN NETWORK ------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------

# --- instantiate environment---------------------------------------------------
# --- ie a patient, limited to 200 steps as in evaluation ----------------------
# ------------------------------------------------------------------------------

patient = TimeLimit(HIVPatient(), max_episode_steps=200)

# --- DQN ----------------------------------------------------------------------

# going GPU if we have one
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hardcoding problem dimensions
state_dim = 6
n_action = 4

# network parameters
nb_neurons=128

# class DQN(nn.Module):
#     """ok, the engine"""
       
#     def __init__(self, nb_neurons=128, device=device, state_dim=6, action_dim=4):
#         super(DQN, self).__init__()
#         self.state_dim = state_dim
#         self.action_dim = action_dim
#         self.fc1 = nn.Linear(self.state_dim, nb_neurons).to(device)  # in : B x state_dim, out : B x nb_neurons
#         self.fc2 = nn.Linear(nb_neurons, nb_neurons).to(device)
#         self.fc3 = nn.Linear(nb_neurons, self.action_dim).to(device)
      
#     def forward(self, inputs):
#         x = F.relu(self.fc1(inputs.to(device)))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x).to(device)
#         return x
    
# DQN = torch.nn.Sequential(
#     nn.Linear(state_dim, nb_neurons),  # in : B x state_dim, out : B x nb_neurons
#     nn.ReLU(),
#     nn.Linear(nb_neurons, nb_neurons),
#     nn.ReLU(), 
#     nn.Linear(nb_neurons, n_action)).to(device) # out : B x nb_action

DQN_instance = DQN(device=device)

# -------------------------------------------------------------------------------------
# ---- TRAINING LOOP ------------------------------------------------------------------
# -------------------------------------------------------------------------------------

# DQN config
config = {'nb_actions': 4,
          'learning_rate': 0.001,
          'gamma': 0.95,
          'buffer_size': 1000000,
          'epsilon_min': 0.01,
          'epsilon_max': 1.,
          'epsilon_decay_period': 10*200, # durée prise par epsilon pour décroître jusqu'à epsilon min, compté en steps = episodes * 200
          'epsilon_delay_decay': 10*200, # time à partir duquel epsilon commence à décroître, compté en steps = episodes * 200
          'batch_size': 20}

# Train agent
MAX_EPISODES = 1000

print(f"training DQN for {MAX_EPISODES:6d} episodes")
agent = dqn_agent(config, DQN_instance)
scores = agent.train(patient, max_episode=MAX_EPISODES)

# display results
# print(f"scores = {scores}")
dqn_scores_savepath = '/home/benjamin.deporte/MVA/mva-rl-assignment-BenjaminDeporte/sandbox/dqn_scores.pkl'
with open(dqn_scores_savepath, "wb") as f:
    print(f"saving DQN training scores")
    pickle.dump(scores, f)

# save DQN
dqn_savepath = '/home/benjamin.deporte/MVA/mva-rl-assignment-BenjaminDeporte/modeles/dqn.pth'
print(f"saving DQN model")
torch.save(agent.model.state_dict(), dqn_savepath)