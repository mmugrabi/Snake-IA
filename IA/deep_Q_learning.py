import torch
import random
import numpy as np
from collections import deque
from IA.model import Linear_QNet, QTrainer
from math import exp
from IA.draw import Draw
from main import RIGHT, LEFT, DOWN, UP, EMPTY_CHAR, FOOD_CHAR

MAX_MEMORY = 100_000
BATCH_SIZE = 2500
LR = 0.00025

FILE_NAME = "3_layers_20x20"

class Agent:
    def __init__(self):
        self.gamma = 0.95  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)                                  # 11, 256, 3
        self.model = Linear_QNet([11, 128, 128, 128, 3], "./saves/"+FILE_NAME)  # 11, 200, 20, 50, 3
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)            # 11, 128, 128, 128, 3

        self.iteration = 0
        self.eps_start = 1
        self.eps_end = 0.0
        self.eps_decay = 0.995

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

    def get_state(self, state):
        def danger(_y, _x):
            return not(0 <= _y < rows and 0 <= _x < columns and (grid[_y][_x] in (EMPTY_CHAR,FOOD_CHAR)))
        grid, head, food, direction, rows, columns = state
        y, x = head
        fy, fx = food
        bool_list = [
            # Danger straight
            danger(*np.add((y, x), direction)),
            # Danger right
            danger(*np.add((y, x), self.direction_converter(2, direction))),
            # Danger left
            danger(*np.add((y, x), self.direction_converter(1, direction))),
            # Move direction
            direction == LEFT,
            direction == RIGHT,
            direction == UP,
            direction == DOWN,
            # Food location
            fx < x, # food left
            fx > x, # food right
            fy < y, # food up
            fy > y] # food down
        return np.asarray(bool_list, dtype=bool)

    def get_epsilon(self):
        epsilon = self.eps_end + (self.eps_start - self.eps_end) * exp(-1. * self.iteration / self.eps_decay)
        return epsilon

    def remember(self):
        self.memory.append((self.observation.values()))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        # for state, action, reward, next_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self):
        self.trainer.train_step(*self.observation.values())

    def save(self, file_name=FILE_NAME):
        self.model.save(file_name)