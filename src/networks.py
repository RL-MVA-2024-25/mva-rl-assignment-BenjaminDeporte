import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    """ok, the engine"""
       
    def __init__(self, device, nb_neurons=128, state_dim=6, action_dim=4):
        super(DQN, self).__init__()
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        # self.bn1 = nn.BatchNorm1d(self.state_dim).to(self.device)
        self.fc1 = nn.Linear(self.state_dim, nb_neurons).to(device)  # in : B x state_dim, out : B x nb_neurons
        # self.bn2 = nn.BatchNorm1d(nb_neurons).to(device)
        self.fc2 = nn.Linear(nb_neurons, nb_neurons).to(device)
        # self.bn3 = nn.BatchNorm1d(nb_neurons).to(device)
        self.fc3 = nn.Linear(nb_neurons, nb_neurons).to(device)
        self.fc4 = nn.Linear(nb_neurons, nb_neurons).to(device)
        self.fc5 = nn.Linear(nb_neurons, self.action_dim).to(device)
        print(f"Instantiating MLP with {nb_neurons} neurons per layer")
      
    def forward(self, inputs):
        # x = self.bn1(inputs.to(self.device))
        x = F.relu(self.fc1(inputs.to(self.device)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x).to(self.device)
        return x