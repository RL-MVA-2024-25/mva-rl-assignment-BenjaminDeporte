# --- where to store replay buffers -------------------------------

save_filepath = '/home/benjamin.deporte/MVA/mva-rl-assignment-BenjaminDeporte/data/replaybuffer.pkl'

# --- imports -----------------------------------------------------

import os, sys
# print(os.getcwd())
sys.path.append('/home/benjamin.deporte/MVA/mva-rl-assignment-BenjaminDeporte/src/')
sys.path.append('/home/benjamin.deporte/MVA/mva-rl-assignment-BenjaminDeporte/lib/')

import pickle
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from tqdm import trange
import random
import numpy as np

import torch
import torch.nn as nn

# -- Replay Buffer ----------
# -- from class 4 -----------

class ReplayBuffer:
    """ReplayBuffer class, from MVA class RL4
    - capacity = max number 
    of samples to collect
    """
    def __init__(self, capacity):
        self.capacity = int(capacity) # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
        
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
        
    def sample(self, batch_size):
        return random.sample(self.data, batch_size)
    
    def __len__(self):
        return len(self.data)
    
    def __repr__(self):
        msg = f"Replay buffer de taille {self.capacity}"
        return msg

    
# --- function to create or load a replay buffer ---------------------------------------------------
  
def get_replay_buffer(fromfile=True, save_filepath=save_filepath, replay_buffer_size=int(1e3)):
    """Load an existing ReplayBuffer, or create one and save it

    Args:
        fromfile (bool, optional): whether we load a replay buffer (True) or create one (False) Defaults to True.
        save_filepath (_type_, optional): file path to save the RB. Defaults to save_filepath.
        replay_buffer_size (_type_, optional): number of samples in the RB if we choose to create one. Defaults to int(1e3).
    """
    
    if fromfile is True:
        with open(save_filepath, "rb") as f:
            memory = pickle.load(f)
            print (f"Loading existing Replay Buffer size {memory.capacity}")
    else:
        patient = TimeLimit(HIVPatient(), max_episode_steps=200)

        nb_samples = int(replay_buffer_size)

        memory = ReplayBuffer(replay_buffer_size)
        state, _ = patient.reset()

        print(f"Creating Replay Buffer with size {replay_buffer_size}")

        for _ in trange(nb_samples):
            action = patient.action_space.sample()  # random action to get samples
            next_state, reward, done, trunc, _ = patient.step(action)
            memory.append(state, action, reward, next_state, done)
            if done:
                state, _ = patient.reset()
            else:
                state = next_state

        print("Done - Replay buffer size:", len(memory))
        
        print(f"saving Replay Buffer in {save_filepath}")
        with open(save_filepath, "wb") as f:
            pickle.dump(memory, f)
            
    return memory


# ---------------------------------------------------------------
# ------ REPLAY BUFFER FOR DQN ----------------------------------
# ---------------------------------------------------------------

class ReplayBuffer2:
    def __init__(self, capacity, device):
        self.capacity = int(capacity) # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
        self.device = device
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def __len__(self):
        return len(self.data)
    
    
# ------------------------------------------------
# --- CREATE REPLAY BUFFER FOR DQN ---------------
# ------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rb_path = '/home/benjamin.deporte/MVA/mva-rl-assignment-BenjaminDeporte/data/buffer4dqn.pkl'

def create_rb_for_dqn(save_filepath=rb_path,replay_buffer_size=int(1e5)):
    
    patient = TimeLimit(HIVPatient(), max_episode_steps=200)
    buffer = ReplayBuffer2(replay_buffer_size, device=device)
    
    state, _ = patient.reset()

    print(f"Creating Replay Buffer for DQN with size {replay_buffer_size}")

    for _ in trange(replay_buffer_size):
        action = patient.action_space.sample()  # random action to get samples
        next_state, reward, done, trunc, _ = patient.step(action)
        buffer.append(state, action, reward, next_state, done)
        if done or trunc:
            state, _ = patient.reset()
        else:
            state = next_state

    print("Done - Replay buffer size:", len(buffer))
        
    print(f"saving Replay Buffer in {rb_path}")
    with open(rb_path, "wb") as f:
        pickle.dump(buffer, f)
        
# ----------------------------------------------------------------

if __name__ == "__main__":
    create_rb_for_dqn()