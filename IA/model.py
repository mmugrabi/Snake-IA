import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Linear_QNet(nn.Module):
    def __init__(self, layers = [11, 200, 3], weights_path="None"):
        super().__init__()
        self.path = weights_path
        self.network = nn.ModuleList()
        for i in range(len(layers)-1):
            self.network.append(nn.Linear(layers[i], layers[i+1]))
        if weights_path != "None" and os.path.exists(weights_path):
            self.load_state_dict(torch.load(weights_path))
            self.eval()

    def forward(self, x):
        """
        obtain an output from the network
        """
        for layer in self.network[:-1]:
            x = F.relu(layer(x))
        x = F.softmax(self.network[-1](x), dim=-1)
        return x

    def save(self, file_name='model.pth'):
        """
        save the model in a file
        """
        model_folder_path = './saves'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        """
        train the model
        """
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)
        pred = self.model(state)
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action[idx]).item()] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()