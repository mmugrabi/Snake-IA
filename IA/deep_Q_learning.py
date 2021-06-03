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
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet([11, 128, 128, 128, 3], "./saves/"+FILE_NAME)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

        self.iteration = 0
        self.eps_start = 1
        self.eps_end = 0.0
        self.eps_decay = 0.995

        self.observation = dict.fromkeys(["state", "action", "reward", "next_state", "done"])
        self.draw = Draw()

    def choose_next_move(self, param):
        """"
        methode call automatically by the game to select the next action of the agent
        """
        grid, head, food, direction, rows, columns = param
        state = self.get_state((grid, head, food, direction, rows, columns))
        action = self.epsilon_greedy_exploration_strategy(state, direction)
        self.observation["state"] = state
        return action

    def update(self, param):
        """
        methode call by the game after a movement
        """
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
        """
        compute the agent reward for its last action
        """
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
        """
        return a random action with a probability epsilon otherwise choose best know action
        """
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
        """
        return the best action from the state of the agent
        """
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
        """
        convert an action from the snake to a cardinal point
        """
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
        """
        obtain the state for the agent from the game
        """
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
        """
        compute the epsilon value
        """
        epsilon = self.eps_end + (self.eps_start - self.eps_end) * exp(-1. * self.iteration / self.eps_decay)
        return epsilon

    def remember(self):
        """
        save in a memory states: before the action, the action", the reward of the action, the state after the action and if the sneak died
        """
        self.memory.append((self.observation.values()))

    def train_long_memory(self):
        """
        train the model on a sample of the memory
        """
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self):
        """
        train the model on the last action
        """
        self.trainer.train_step(*self.observation.values())

    def save(self, file_name=FILE_NAME):
        """
        save the model in a file
        """
        self.model.save(file_name)