# -----------------------------------------------------------------------------
# --- CODE TO LEARN A TARGET NETWORK  -----------------------------------------
# --- TO BE USED IN A tn AGENT CLASS ------------------------------------------
# -----------------------------------------------------------------------------


# --- imports -----------------------------------------------------------------

import os, sys
# print(os.getcwd())
sys.path.append('/home/benjamin.deporte/MVA/mva-rl-assignment-BenjaminDeporte/src/')
sys.path.append('/home/benjamin.deporte/MVA/mva-rl-assignment-BenjaminDeporte/lib/')

from replay import ReplayBuffer2, get_replay_buffer
# from sklearn.ensemble import RandomForestRegressor
from networks import DQN

import numpy as np

from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
# from fast_env import FastHIVPatient # merci Clement

# import random
from tqdm import tqdm
from tqdm import trange
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
# import gymnasium as gym

import matplotlib.pyplot as plt

# --- PICK GREEDY ACTION WRT NETWORK AND STATE

def greedy_action(network, state):
    device = "cuda" if next(network.parameters()).is_cuda else "cpu"
    with torch.no_grad():
        Q = network(torch.Tensor(state).unsqueeze(0).to(device))
        return torch.argmax(Q).item()
    
# ---------------------------------------------------------------------------------
# --- DOUBLE DQN NETWORK CLASS ----------------------------------------------------
# ---------------------------------------------------------------------------------

class ddqn_agent:
    
    # evaluate action from the epsilon-greedy policy via online network
    # estimate the value function via the target network, not the online network
    
    # tn_dqn_inter1_savepath = '/home/benjamin.deporte/MVA/mva-rl-assignment-BenjaminDeporte/modeles/tn_dqn_inter1.pth'
    # tn_dqn_inter2_savepath = '/home/benjamin.deporte/MVA/mva-rl-assignment-BenjaminDeporte/modeles/tn_dqn_inter2.pth'
    ddqn_savepath = '/home/benjamin.deporte/MVA/mva-rl-assignment-BenjaminDeporte/modeles/ddqn.pth'
    
    # save_delays = 20 # saves models every N episodes, just in case
    
    def __init__(self, config, model):
        device = "cuda" if next(model.parameters()).is_cuda else "cpu"
        self.nb_actions = config['nb_actions']
        self.gamma = config['gamma'] if 'gamma' in config.keys() else 0.95
        self.batch_size = config['batch_size'] if 'batch_size' in config.keys() else 100
        self.buffer_size = config['buffer_size'] if 'buffer_size' in config.keys() else int(1e5)
        if config['load_buffer'] is True:
            rb_path = '/home/benjamin.deporte/MVA/mva-rl-assignment-BenjaminDeporte/data/buffer4dqn.pkl'
            with open(rb_path, "rb") as f:
                self.memory = pickle.load(f)
            print(f"Loaded pre-existing replay buffer of size {len(self.memory)}")
        else:
            self.memory = ReplayBuffer2(self.buffer_size,device)
            print(f"Empty replay buffer size {len(self.memory)}")
        self.epsilon_max = config['epsilon_max'] if 'epsilon_max' in config.keys() else 1.
        self.epsilon_min = config['epsilon_min'] if 'epsilon_min' in config.keys() else 0.01
        self.epsilon_stop = config['epsilon_decay_period'] if 'epsilon_decay_period' in config.keys() else 1000
        self.epsilon_delay = config['epsilon_delay_decay'] if 'epsilon_delay_decay' in config.keys() else 20
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
        self.model = model 
        self.target_model = copy.deepcopy(self.model).to(device)
        self.criterion = config['criterion'] if 'criterion' in config.keys() else torch.nn.MSELoss()
        lr = config['learning_rate'] if 'learning_rate' in config.keys() else 0.001
        self.optimizer = config['optimizer'] if 'optimizer' in config.keys() else torch.optim.Adam(self.model.parameters(), lr=lr)
        self.nb_gradient_steps = config['gradient_steps'] if 'gradient_steps' in config.keys() else 1
        self.update_target_strategy = config['update_target_strategy'] if 'update_target_strategy' in config.keys() else 'replace'
        self.update_target_freq = config['update_target_freq'] if 'update_target_freq' in config.keys() else 20
        self.update_target_tau = config['update_target_tau'] if 'update_target_tau' in config.keys() else 0.005
        # sandbox environment patient for buffer fill at the end of each episode
        self.patient_for_buffer = TimeLimit(HIVPatient(), max_episode_steps=200)
        self.average_episodes = config['average_episodes'] if 'average_episodes' in config.keys() else 5
    
    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            # ---- sample batch from replay buffer ---------------------
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            # ---- get Q(s,a) from Q network ---------------------------
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1)) # gather is the PyTorch version of VLookUp : we pick the q(x,a) for each (x,a) in (X,A)
            # --- select best action in Y based on Q network values ----
            # --- ie argmax Q(Y,a') ------------------------------------
            Q_values = self.model(Y)
            _, a_prime = Q_values.max(1)
            # --- update is Q_target(Y,a') -------------------------------------         
            QYmax = self.target_model(Y).gather(1,a_prime.unsqueeze(1)).squeeze()
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma) # compute R + self.gamma * (1-D) * QYmax, that is the update
            # --- optimization -----
            loss = self.criterion(QXA, update.unsqueeze(1)) # unsqueeze fait passer la shape de QXA de B à (B,1)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 
            
    # def update_buffer(self, epsilon, n=10):
    #     # runs n episodes with the current model and an epsilon greedy policy to update the buffer
    #     for i in range(n):
    #         s2, _ = self.patient_for_buffer.reset()
    #         # select epsilon-greedy action -----------------------------------------
    #         if np.random.rand() < epsilon:
    #             a2 = self.patient_for_buffer.action_space.sample()
    #         else:
    #             a2 = greedy_action(self.model, s2)
    #         # run a full episode to update the buffer
    #         done2 = False
    #         trunc2 = False
    #         while done2 is False and trunc2 is False:
    #             next_s2, r2, done2, trunc2, _ = self.patient_for_buffer.step(a2)
    #             self.memory.append(s2, a2, r2, next_s2, done2)
    #             s2 = next_s2
    #             # select epsilon-greedy action -----------------------------------------
    #             if np.random.rand() < epsilon:
    #                 a2 = self.patient_for_buffer.action_space.sample()
    #             else:
    #                 a2 = greedy_action(self.model, s2)
    
    def train(self, env, max_episode):
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0
        best_avg_reward = 0
        
        while episode < max_episode:
            # update epsilon -------------------------------------------------------
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)
            # select epsilon-greedy action -----------------------------------------
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = greedy_action(self.model, state)
            # one step : update replay buffer + get next state and reward ----------
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward
            # train : perform n gradient steps on the target network ---------------
            for gs in range(self.nb_gradient_steps): 
                # print(f"Performing gradient step on target network {gs+1} / {self.nb_gradient_steps}", end="\r")
                self.gradient_step()
            # update target network if needed --------------------------------------
            if self.update_target_strategy == 'replace':
                if step % self.update_target_freq == 0: 
                    # print(f"updating target model with replacement strategy")
                    self.target_model.load_state_dict(self.model.state_dict())
            if self.update_target_strategy == 'ema':
                target_state_dict = self.target_model.state_dict()
                model_state_dict = self.model.state_dict()
                tau = self.update_target_tau
                for key in model_state_dict:
                    target_state_dict[key] = tau*model_state_dict[key] + (1-tau)*target_state_dict[key]
                # print(f"updating target model with moving average strategy")
                target_model.load_state_dict(target_state_dict)
            # next transition ------------------------------------------------------
            step += 1
            if done or trunc:
                # episode is done, report out my dear ------------------------------
                episode += 1
                state, _ = env.reset()
                episode_cum_reward = episode_cum_reward / step * 200 # normalize reward at 200 steps
                print(f"Episode {episode:5d}, epsilon {epsilon:6.2f}, buffer size {len(self.memory):6d}, episode return {episode_cum_reward:.1e}")
                episode_return.append(episode_cum_reward)
                episode_cum_reward = 0
                
                # update buffer with a full episode --------------------------------
                # if len(self.memory) < self.buffer_size:
                #     self.update_buffer(epsilon, n=20)
                
                # look at model performance, save if best
                if episode >= self.average_episodes:
                    last_avg_reward = np.mean(episode_return[-self.average_episodes:])
                    if last_avg_reward > best_avg_reward:
                        best_avg_reward = last_avg_reward
                        print(f"=> saving model with average perf {best_avg_reward:.2e} over last {self.average_episodes} episodes")
                        torch.save(self.model.state_dict(), self.ddqn_savepath)
                
            else:
                state = next_state
                
        return episode_return
    
# ------------------------------------------------------------------------------------------
# --- TRAIN TARGET NETWORK -----------------------------------------------------------------
# ------------------------------------------------------------------------------------------

# --- instantiate environment---------------------------------------------------
# --- ie a patient, limited to 200 steps as in evaluation ----------------------
# ------------------------------------------------------------------------------

MAX_STEPS_PER_PATIENT=200

patient = TimeLimit(HIVPatient(), max_episode_steps=MAX_STEPS_PER_PATIENT)

# --- DQN ----------------------------------------------------------------------

# going GPU if we have one
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hardcoding problem dimensions
state_dim = 6
n_action = 4

# network parameters
nb_neurons=32

# instantiate
DQN_instance = DQN(device=device, nb_neurons=nb_neurons)

# -------------------------------------------------------------------------------------
# ---- TRAINING LOOP ------------------------------------------------------------------
# -------------------------------------------------------------------------------------

# DQN config
config = {'nb_actions': 4,
          'learning_rate': 0.01,
          'gamma': 0.90,
          'buffer_size': 100000,
          'load_buffer': True,
          'epsilon_min': 0.01,
          'epsilon_max': 1.,
          'epsilon_decay_period': 10*200, # durée prise par epsilon pour décroître jusqu'à epsilon min, compté en steps = episodes * 200
          'epsilon_delay_decay': 10*200, # time à partir duquel epsilon commence à décroître, compté en steps = episodes * 200
          'batch_size': 1000,
          'gradient_steps': 100, # gradient steps for the target network
          'update_target_strategy': 'replace', # or 'ema'
          'update_target_freq': 100,  # fréquence d'update du target network, en steps (rappel : 200 steps / episode)
          'update_target_tau': 0.005,
          'criterion': torch.nn.SmoothL1Loss(),
          'average_episodes' : 5, # average reward is calculated over this number of episodes, and model is saved when performance is best ever
          }

# Train agent
MAX_EPISODES = 1000

print(f"Training DDQN for {MAX_EPISODES:6d} episodes")
agent = ddqn_agent(config, DQN_instance)
scores = agent.train(patient, max_episode=MAX_EPISODES)

# display results
# print(f"scores = {scores}")
tn_dqn_scores_savepath = '/home/benjamin.deporte/MVA/mva-rl-assignment-BenjaminDeporte/sandbox/training_scores.pkl'
with open(tn_dqn_scores_savepath, "wb") as f:
    print(f"saving DDQN training scores")
    pickle.dump(scores, f)