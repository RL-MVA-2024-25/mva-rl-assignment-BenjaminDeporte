from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient

import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# import os, sys
# sys.path.append('networks.py')
# from networks import DQN

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.

# --- Manage device ------------------------------------------------------

# --- FOR TRAINING ------------------------------------------------------------
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# --- FOR UPLOAD TO GITHUB ----------------------------------------------------
device = torch.device('cpu')

#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
#------------ TEMPLATE ---------------------------------------------------------------
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------

# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
# class ProjectAgent:
#     def act(self, observation, use_random=False):
#         return 0

#     def save(self, path):
#         pass

#     def load(self):
#         pass


#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
#---- FITTED Q-ITERATION -------------------------------------------------------------
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------

# class ProjectAgent:
#     """Fitted Q Iteration Agent - to demonstrate a Minimum Viable Product
#     """
    
#     rfr_path = '/home/benjamin.deporte/MVA/mva-rl-assignment-BenjaminDeporte/modeles/rfr_regressor.hdf5' # or .pkl'
#     nb_actions = 4 # number of possible actions, hard coded, we know it
    
#     def _greedy_action(self, s):
#         """determines greedy action wrt fitted regressor

#         Args:
#             s (np.array): one single state (no batch)
#         """
#         Qsa = []
#         for a in range(self.nb_actions):
#             sa = np.append(s,a).reshape(1,-1)
#             Qsa.append(self.rfr.predict(sa))
#         return np.argmax(Qsa)
        
#     def act(self, observation, use_random=False):
#         action = self._greedy_action(observation)
#         return action

#     def save(self, path):
#         pass

#     def load(self):
        #   with h5py.File(self.rfr_path, 'r') as f:
        #      self.rfr = f['default']
#         with open(self.rfr_path, "rb") as f:
#             self.rfr = pickle.load(f)
            
# ----------------------------------------------------------------------------------------------
# ----------------------- DQN Agent ------------------------------------------------------------
# ----------------------------------------------------------------------------------------------

class DQN(nn.Module):
    """MLP un peu péchû"""
       
    def __init__(self, device, nb_neurons=256, state_dim=6, action_dim=4):
        super(DQN, self).__init__()
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fc1 = nn.Linear(self.state_dim, nb_neurons).to(device)  # in : B x state_dim, out : B x nb_neurons
        self.fc2 = nn.Linear(nb_neurons, nb_neurons).to(device)
        self.fc3 = nn.Linear(nb_neurons, nb_neurons).to(device)
        self.fc4 = nn.Linear(nb_neurons, self.action_dim).to(device)
        print(f"Instantiating MLP with {nb_neurons} neurons per layer")
      
    def forward(self, inputs):
        x = inputs.to(self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x).to(self.device)
        return x


# ----------------------------------------------------------------------------------------------------------------
# ------------- TARGET NETWORK -----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------
        
class ProjectAgent:
    """ Target Network Agent"""
    
    # for GitHub
    model_path = "model.pth"
    # for autograding
    # model_path = '/home/benjamin.deporte/MVA/mva-rl-assignment-BenjaminDeporte/modeles/tn_dqn_last.pth'
    
    # nb_actions = 4 # number of possible actions, hard coded, we know it
    
    def _greedy_action(self, s):
        """determines greedy action wrt dqn agent

        Args:
            s (np.array): one single state (no batch)
        """
        device = "cuda" if next(self.dqn.parameters()).is_cuda else "cpu"
        with torch.no_grad():
            Q = self.dqn(torch.Tensor(s).unsqueeze(0).to(device))
            return torch.argmax(Q).item()
        
    def act(self, observation, use_random=False):
        action = self._greedy_action(observation)
        return action

    def save(self, path):
        pass

    def load(self):
        # for upload to GitHub : cpu
        device = torch.device('cpu')
        
        # need to instantiate a DQN() for the torch.load to work        
        self.dqn = DQN(device=device, nb_neurons=256)
        
        # load
        self.dqn.load_state_dict(torch.load(self.model_path, map_location=device, weights_only=True))