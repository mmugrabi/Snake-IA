import torch
import random
import numpy as np
from collections import deque
from IA.model import CNNModel, QTrainer
from math import exp
from IA.draw import Draw
from main import RIGHT, LEFT, DOWN, UP, SNAKE_CHAR, EMPTY_CHAR, WALL_CHAR, FOOD_CHAR

MAX_MEMORY = 100_000
BATCH_SIZE = 2500
LR = 0.00025

<<<<<<< HEAD
FILE_NAME = "3_layers_6x6"
=======
FILE_NAME = "full_grid"
>>>>>>> 2cbb235c4a00989db8e9ea55617128d75971948d

class Agent:
    def __init__(self):
        self.gamma = 0.95  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)                                  # 11, 256, 3
        self.model = CNNModel([100, 128, 128, 128, 3], "./saves/"+FILE_NAME)  # 11, 200, 20, 50, 3
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)            # 11, 128, 128, 128, 3

        self.iteration = 0
        self.eps_start = 1
<<<<<<< HEAD
        self.eps_end = 0
        self.eps_decay = 0.995
=======
        self.eps_end = 0.01
        self.eps_decay = 40#0.995
>>>>>>> 2cbb235c4a00989db8e9ea55617128d75971948d

        self.observation = dict.fromkeys(["state", "action", "reward", "next_state", "done"])
        self.draw = Draw()

    def choose_next_move(self, param):
        grid, head, food, direction, rows, columns = param
        state = self.get_state((grid, head, food, direction, rows, columns))
        action = self.epsilon_greedy_exploration_strategy(state, direction)
        self.observation["state"] = state
        return action

    def update(self, param):
        grid, score, alive, head, food, eaten, direction, rows, columns = param
        if not alive:
            self.iteration += 1
            self.draw.plot(score)
        self.observation["reward"] = self.get_reward(alive, eaten, food, head, direction)
        self.observation["done"] = not alive
        self.observation["next_state"] = self.get_state((grid, head, food, direction, rows, columns))
        self.remember()
        self.train_short_memory()

    @staticmethod
    def get_reward(alive, eaten, food, head, direction):
        reward = 0
        y, x = head
        fy, fx = food
        if not alive:
            reward += -100
        elif eaten:
            reward += 10
        if fx < x: # food left
            if direction == LEFT:
                reward += 1
            elif direction == RIGHT:
                reward += -1
        if fx > x: # food right
            if direction == RIGHT:
                reward += 1
            elif direction == LEFT:
                reward += -1
        if fy < y: # food up
            if direction == UP:
                reward += 1
            elif direction == DOWN:
                reward += -1
        if fy > y: # food down
            if direction == DOWN:
                reward += 1
            elif direction == UP:
                reward += -1
        return reward

    def epsilon_greedy_exploration_strategy(self, state, direction):
        if random.random() <= self.get_epsilon():
            random_action = random.randint(0,2)
            action = self.direction_converter(random_action, direction)
            tmp = [0, 0, 0]
            tmp[random_action] = 1
            self.observation["action"] = tmp
        else:
            action = self.best_move(state, direction)
        return action

    def best_move(self, state, direction):
        state0 = torch.tensor(state, dtype=torch.float)
        prediction = self.model(state0)
        action = torch.argmax(prediction).item()
        tmp = [0, 0, 0]
        tmp[action] = 1
        self.observation["action"] = tmp
        move = self.direction_converter(action, direction)
        return move

    @staticmethod
    def direction_converter(action, direction):
        if direction == UP:
            move = [UP, LEFT, RIGHT][action]
        elif direction == RIGHT:
            move = [RIGHT, UP, DOWN][action]
        elif direction == DOWN:
            move = [DOWN, RIGHT, LEFT][action]
        else:
            move = [LEFT, DOWN, UP][action]
        return move

    @staticmethod
    def get_state(state):
        grid, head, food, direction, rows, columns = state
        state_list = []
        for i in range(rows):
            state_list.append([])
            for j in range(columns):
                char = ord(grid[i][j])
                if (i, j) == head:
                    state_list[-1].append(2)
                elif char in [SNAKE_CHAR, WALL_CHAR]:
                    state_list[-1].append(3)
                elif char is FOOD_CHAR:
                    state_list[-1].append(1)
                else:
                    state_list[-1].append(0)
        state_list = np.asarray(state_list)
        state_list = state_list / np.linalg.norm(state_list)
        return state_list

    def get_epsilon(self):
        epsilon = self.eps_end + (self.eps_start - self.eps_end) * exp(-1. * self.iteration / self.eps_decay)
        return epsilon

    def remember(self):
        self.memory.append((self.observation.values()))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self):
        self.trainer.train_step(*self.observation.values())

    def save(self, file_name=FILE_NAME):
        self.model.save(file_name)