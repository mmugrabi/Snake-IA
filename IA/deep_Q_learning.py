import random
import numpy as np
import math
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from main import RIGHT, LEFT, DOWN, UP, EMPTY_CHAR
MOVE = [RIGHT, LEFT, DOWN, UP]

class DQNAgent(torch.nn.Module):
    def __init__(self):
        super(DQNAgent, self).__init__()
        self.training = True
        self.learning_rate = 0.001 #0.00013629

        self.gamma = 0.9
        self.optimizer = None

        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 200
        self.target_update_rate = 100
        self.iteration = 0

        self.memory = deque(maxlen=2500)
        self.batch_size = 128
        self.observation = dict.fromkeys(["state", "action", "reward", "next_state", "done"])

        self.weights_path = "None"
        self.main_network = nn.ModuleList(self.init_network())

    def choose_next_move(self, param):
        state = self.get_state(param)
        if self.training:
            move = self.observe(state)
        else:
            move = self.best_move(state)
        return move

    def observe(self, state):
        action = self.epsilon_greedy_exploration_strategy(state)
        self.observation["state"] = state
        self.observation["action"] = action
        return action

    def update(self, param):
        self.iteration += 1
        if self.training:
            grid, score, alive, head, food, eaten = param
            if not alive:
                reward = -10
            elif eaten:
                reward = 10
            else:
                reward = 0
            self.observation["reward"] = reward
            self.observation["done"] = not alive
            self.observation["next_state"] = self.get_state((grid, head, food))
            self.remember()
            self.train_short_memory()

    def epsilon_greedy_exploration_strategy(self, state):
        if random.random() <= self.get_epsilon():
            action = random.choice(MOVE)
        else:
            action = self.best_move(state)
        return action

    def best_move(self, state):
        Q_values = self.forward(state)
        return MOVE[torch.argmax(Q_values)]

    @staticmethod
    def get_state(state):
        def is_empty(_x, _y):
            return 0 <= _y < len(grid) and 0 <= _x < len(grid[0]) and grid[_y][_x] == EMPTY_CHAR
        grid, head, food = state
        y, x = head
        fy, fx = food
        bool_list = [fy > y,
                     fy < y,
                     fx > x,
                     fx < x,
                     is_empty(y+1, x  ),
                     is_empty(y+1, x+1),
                     is_empty(y  , x+1),
                     is_empty(y-1, x+1),
                     is_empty(y-1, x  ),
                     is_empty(y-1, x-1),
                     is_empty(y  , x-1),
                     is_empty(y+1, x-1)]
        state_tensor = torch.tensor(bool_list, dtype=torch.float32).to(DEVICE)
        return state_tensor

    def get_epsilon(self):
        epsilon = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.iteration / self.eps_decay)
        return epsilon

    def init_network(self, input_layer=12, first_layer=200, second_layer=20, third_layer=50, output_layer=4):
        f1 = nn.Linear(input_layer, first_layer)
        f2 = nn.Linear(first_layer, second_layer)
        f3 = nn.Linear(second_layer, third_layer)
        f4 = nn.Linear(third_layer, output_layer)
        network = [f1, f2, f3, f4]
        if self.weights_path != "None":
            self.load_state_dict(torch.load(self.weights_path))
        return network

    def save_network(self):
        if self.weights_path != "None":
            torch.save(self.model.state_dict(), self.weights_path)

    def forward(self, x):
        f1, f2 , f3, f4 = self.main_network
        x = F.relu(f1(x))
        x = F.relu(f2(x))
        x = F.relu(f3(x))
        x = F.softmax(f4(x), dim=-1)
        return x

    def remember(self):
        """
        Store the <state, action, reward, next_state, is_done> tuple in a memory buffer for replay memory.
        """
        self.memory.append((self.observation.values()))

    def replay_new(self):
        """
        Replay memory.
        """
        if len(self.memory) > self.batch_size:
            minibatch = random.sample(self.memory, self.batch_size)
        else:
            minibatch = self.memory
        for state, action, reward, next_state, done in minibatch:
            self.train_agent(state, action, reward, next_state, done)

    def train_short_memory(self):
        """
        Train at the current timestep.
        """
        state, action, reward, next_state, done = self.observation.values()
        self.train_agent(state, action, reward, next_state, done)

    def train_agent(self, state, action, reward, next_state, done):
        """
        Train the DQN agent on the <state, action, reward, next_state, is_done>
        """
        self.train()
        torch.set_grad_enabled(True)
        target = reward
        if not done:
            target = reward + self.gamma * torch.max(self.forward(next_state)) # bellman equation
        output = self.forward(state)
        target_f = output.clone()
        target_f[np.argmax(action)] = target
        target_f.detach()
        self.optimizer.zero_grad()
        loss = F.mse_loss(output, target_f)
        loss.backward()
        self.optimizer.step()

