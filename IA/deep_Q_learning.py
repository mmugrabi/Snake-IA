import random
import numpy as np
import pandas as pd
from operator import add
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
DEVICE = 'cpu' # 'cuda' if torch.cuda.is_available() else 'cpu'
from main import RIGHT, LEFT, DOWN, UP, EMPTY_CHAR
MOVE = [RIGHT, LEFT, DOWN, UP]

class DQNAgent(torch.nn.Module):
    def __init__(self):
        super(DQNAgent, self).__init__()
        self.training = True
        self.learning_rate = 0.001 #0.00013629
        self.epsilon = 0.1
        self.gamma = 0.9
        self.optimizer = None

        self.memory = deque(maxlen=2500)
        self.observation = dict.fromkeys(["state", "action", "reward", "next_state", "done"])

        self.main_network = self.init_network()
        #self.target_network = self.init_network()

        # self.dataframe = pd.DataFrame()
        # self.short_memory = np.array([])
        # self.agent_target = 1
        # self.agent_predict = 0
        # self.actual = []

    def set_parameter(self, training=None, epsilon=None, learning_rate=None):
        if training is not None:
            self.training = training
        if epsilon is not None:
            self.epsilon = epsilon
        if learning_rate is not None:
            self.learning_rate = learning_rate

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
        if random.random() <= self.epsilon:
            action = random.choice(MOVE)
        else:
            action = self.best_move(state)
        return action

    def best_move(self, state):
        Q_values = self.forward(state)
        return MOVE[torch.argmax(Q_values)]

    # def train(self):
    #     mb_size = 50  # Learning minibatch size
    #     minibatch = random.sample(D, mb_size)  # Sample some moves
    #
    #     inputs_shape = (mb_size,) + state.shape[1:]
    #     inputs = np.zeros(inputs_shape)
    #     targets = np.zeros((mb_size, env.action_space.n))
    #
    #     for i in range(0, mb_size):
    #         state = minibatch[i][0]
    #         action = minibatch[i][1]
    #         reward = minibatch[i][2]
    #         state_new = minibatch[i][3]
    #         done = minibatch[i][4]
    #
    #         # Build Bellman equation for the Q function
    #         inputs[i:i + 1] = np.expand_dims(state, axis=0)
    #         targets[i] = model.predict(state)
    #         Q_sa = model.predict(state_new)
    #
    #         if done:
    #             targets[i, action] = reward
    #         else:
    #             targets[i, action] = reward + gamma * np.max(Q_sa)
    #
    #         # Train network to output the Q function
    #         model.train_on_batch(inputs, targets)
    #     print('Learning Finished')

    @staticmethod
    def get_state(state):
        grid, head, food = state
        y, x = head
        fy, fx = food
        bool_list = [fy > y,
                     fy < y,
                     fx > x,
                     fx < x,
                     y + 1 < len(grid) and grid[y + 1][x] == EMPTY_CHAR,
                     y + 1 < len(grid) and x + 1 < len(grid[0]) and grid[y + 1][x + 1] == EMPTY_CHAR,
                     x + 1 < len(grid[0]) and grid[y][x + 1] == EMPTY_CHAR,
                     y - 1 >= 0 and x + 1 < len(grid[0]) and grid[y - 1][x + 1] == EMPTY_CHAR,
                     y - 1 >= 0 and grid[y - 1][x] == EMPTY_CHAR,
                     y - 1 >= 0 and x - 1 >= 0 and grid[y - 1][x - 1] == EMPTY_CHAR,
                     x - 1 >= 0 and grid[y][x - 1] == EMPTY_CHAR,
                     y + 1 < len(grid) and x - 1 >= 0 and grid[y + 1][x - 1] == EMPTY_CHAR]
        state_tensor = torch.tensor(bool_list, dtype=torch.float32).to(DEVICE)
        return state_tensor

    def bellman_equation(self):
        pass

    def init_network(self, input_layer=12, first_layer=200, second_layer=20, third_layer=50, output_layer=4, weights_path ="None"):
        f1 = nn.Linear(input_layer, first_layer)
        f2 = nn.Linear(first_layer, second_layer)
        f3 = nn.Linear(second_layer, third_layer)
        f4 = nn.Linear(third_layer, output_layer)
        network = [f1, f2, f3, f4]
        if weights_path != "None":
            self.model = self.load_state_dict(torch.load(weights_path))
            print("weights loaded")
        return network

    def save_network(self, path):
        torch.save(self.model.state_dict(), path)

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
        self.memory.append((self.observation["state"],
                            self.observation["action"],
                            self.observation["reward"],
                            self.observation["next_state"],
                            self.observation["done"]))

    def replay_new(self, batch_size=50):
        """
        Replay memory.
        """
        if len(self.memory) > batch_size:
            minibatch = random.sample(self.memory, batch_size)
        else:
            minibatch = self.memory
        for state, action, reward, next_state, done in minibatch:
            self.train()
            torch.set_grad_enabled(True)
            target = reward
            # next_state_tensor = torch.tensor(np.expand_dims(next_state, 0), dtype=torch.float32).to(DEVICE)
            # state_tensor = torch.tensor(np.expand_dims(state, 0), dtype=torch.float32, requires_grad=True).to(DEVICE)
            if not done:
                target = reward + self.gamma * torch.max(self.forward(next_state)[0])
            output = self.forward(state)
            target_f = output.clone()
            target_f[np.argmax(action)] = target
            target_f.detach()
            #self.optimizer.zero_grad()
            loss = F.mse_loss(output, target_f)
            loss.backward()
            #self.optimizer.step()

    def train_short_memory(self):
        """
        Train the DQN agent on the <state, action, reward, next_state, is_done>
        tuple at the current timestep.
        """
        state, action, reward, next_state, done = self.observation.values()
        self.train()
        torch.set_grad_enabled(True)
        target = reward
        # next_state_tensor = torch.tensor(next_state.reshape((1, 11)), dtype=torch.float32).to(DEVICE)
        # state_tensor = torch.tensor(state.reshape((1, 11)), dtype=torch.float32, requires_grad=True).to(DEVICE)
        if not done:
            target = reward + self.gamma * torch.max(self.forward(next_state))
        output = self.forward(state)
        target_f = output.clone()
        target_f[np.argmax(action)] = target
        target_f.detach()
        #self.optimizer.zero_grad()
        loss = F.mse_loss(output, target_f)
        loss.backward()
        #self.optimizer.step()


# def run():
#     """
#     Run the DQN algorithm, based on the parameters previously set.
#     """
#     pygame.init()
#     agent = DQNAgent()
#     agent = agent.to(DEVICE)
#     agent.optimizer = optim.Adam(agent.parameters(), weight_decay=0, lr=params['learning_rate'])
#     counter_games = 0
#     score_plot = []
#     counter_plot = []
#     record = 0
#     total_score = 0
#     while counter_games < params['episodes']:
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 pygame.quit()
#                 quit()
#         # Initialize classes
#         game = Game(440, 440)
#         player1 = game.player
#         food1 = game.food
#
#         # Perform first move
#         initialize_game(player1, game, food1, agent, params['batch_size'])
#         if params['display']:
#             display(player1, food1, game, record)
#
#         steps = 0  # steps since the last positive reward
#         while (not game.crash) and (steps < 100):
#             if not params['train']:
#                 agent.epsilon = 0.01
#             else:
#                 # agent.epsilon is set to give randomness to actions
#                 agent.epsilon = 1 - (counter_games * params['epsilon_decay_linear'])
#
#             # get old state
#             state_old = agent.get_state(game, player1, food1)
#
#             # perform random actions based on agent.epsilon, or choose the action
#             if random.uniform(0, 1) < agent.epsilon:
#                 final_move = np.eye(3)[randint(0, 2)]
#             else:
#                 # predict action based on the old state
#                 with torch.no_grad():
#                     state_old_tensor = torch.tensor(state_old.reshape((1, 11)), dtype=torch.float32).to(DEVICE)
#                     prediction = agent(state_old_tensor)
#                     final_move = np.eye(3)[np.argmax(prediction.detach().cpu().numpy()[0])]
#
#             # perform new move and get new state
#             player1.do_move(final_move, player1.x, player1.y, game, food1, agent)
#             state_new = agent.get_state(game, player1, food1)
#
#             # set reward for the new state
#             reward = agent.set_reward(player1, game.crash)
#
#             # if food is eaten, steps is set to 0
#             if reward > 0:
#                 steps = 0
#
#             if params['train']:
#                 # train short memory base on the new action and state
#                 agent.train_short_memory(state_old, final_move, reward, state_new, game.crash)
#                 # store the new data into a long term memory
#                 agent.remember(state_old, final_move, reward, state_new, game.crash)
#
#             record = get_record(game.score, record)
#             if params['display']:
#                 display(player1, food1, game, record)
#                 pygame.time.wait(params['speed'])
#             steps += 1
#         if params['train']:
#             agent.replay_new(agent.memory, params['batch_size'])
#         counter_games += 1
#         total_score += game.score
#         print(f'Game {counter_games}      Score: {game.score}')
#         score_plot.append(game.score)
#         counter_plot.append(counter_games)
#     mean, stdev = get_mean_stdev(score_plot)
#     if params['train']:
#         model_weights = agent.state_dict()
#         torch.save(model_weights, params["weights_path"])
#     if params['plot_score']:
#         plot_seaborn(counter_plot, score_plot, params['train'])
#     return total_score, mean, stdev