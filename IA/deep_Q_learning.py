import random
import numpy as np
import pandas as pd
from operator import add
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
DEVICE = 'cpu' # 'cuda' if torch.cuda.is_available() else 'cpu'
from main import RIGHT, LEFT, DOWN, UP
class DQNAgent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.learning_rate = 0.001 #0.00013629

        self.main_network = self.init_network()
        self.target_network = self.init_network()

    def choose_next_move(self, state):
        grid, score, alive, head = state
        return RIGHT #self.moves[self.i % len(self.moves)]

    @staticmethod
    def init_network(input_layer=12, first_layer=200, second_layer=20, third_layer=50, output_layer=4, weights_path ="None"):
        # Layers
        f1 = nn.Linear(input_layer, first_layer)
        f2 = nn.Linear(first_layer, second_layer)
        f3 = nn.Linear(second_layer, third_layer)
        f4 = nn.Linear(third_layer, output_layer)
        network = [f1, f2, f3, f4]
        # load saved weights
        if weights_path != "None":
            #self.model = self.load_state_dict(torch.load(weights_path))
            print("weights loaded")
        return network

    def forward(self, x):
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        x = F.relu(self.f3(x))
        x = F.softmax(self.f4(x), dim=-1)
        return x

    def get_state(self):
        pass

    def train(self, state, action, reward, next_state, done):
        """
            Train the DQN agent on the <state, action, reward, next_state, is_done>
            tuple at the current timestep.
        """
        self.train()
        torch.set_grad_enabled(True)
        target = reward
        next_state_tensor = torch.tensor(next_state.reshape((1, 11)), dtype=torch.float32).to(DEVICE)
        state_tensor = torch.tensor(state.reshape((1, 11)), dtype=torch.float32, requires_grad=True).to(DEVICE)
        if not done:
            target = reward + self.gamma * torch.max(self.forward(next_state_tensor[0]))
        output = self.forward(state_tensor)
        target_f = output.clone()
        target_f[0][np.argmax(action)] = target
        target_f.detach()
        self.optimizer.zero_grad()
        loss = F.mse_loss(output, target_f)
        loss.backward()
        self.optimizer.step()

    def play(self):
        pass