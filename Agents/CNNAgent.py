##########
# Import #
##########

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .BaselineAgent import BaselineAgent
from .Memory import Memory


#############
# CNN Class #
#############

class DQNet(nn.Module):
    def __init__(self, n_actions):
        super(DQNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),              # 150 x 150 x 16
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),                             # 75 x 75 x 16
            nn.Conv2d(16, 32, 3, padding=1),             # 75 x 75 x 32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),                             # 37 x 37 x 32
            nn.Conv2d(32, 64, 3, padding=1),             # 37 x 37 x 64
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),                             # 18 x 18 x 64
            nn.Conv2d(64, 128, 3, padding=1),            # 18 x 18 x 128
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),                             # 9 x 9 x 128
            nn.Conv2d(128, 256, 3, padding=1),           # 9 x 9 x 256
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),                             # 4 x 4 x 256
            nn.Conv2d(256, 512, 3, padding=1),           # 4 x 4 x 512
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),                             # 2 x 2 x 512
            nn.Conv2d(512, 512, 3, padding=1),           # 2 x 2 x 512
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),                             # 1 x 1 x 512
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


class CNNAgent(BaselineAgent):
    def __init__(self, env, model_path, epsilon=0.1, memory_size=100, discount=0.9, train=False):
        super().__init__(env, epsilon=epsilon)
        self.__env = env
        self.__epsilon = epsilon
        self.__memory = Memory(memory_size)
        self.__discount = discount
        self.__device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.__train = train
        self.__model_path = model_path
        self.create_model()

    def create_model(self):
        if self.__train:
            nA = self.__env.get_number_of_actions()
            self.__model = DQNet(nA)
            self.__optimizer = optim.Adam(self.__model.parameters(), lr=0.0005)
            self.__criterion = nn.MSELoss()
        else:
            self.load()
        self.__model.to(self.__device)

    def learned_act(self, state):
        state = torch.Tensor([state]).permute(0, 3, 1, 2).to(self.__device)
        with torch.no_grad():
            q_values = self.__model(state)[0]
        return torch.argmax(q_values)

    def reinforce(self, state, next_state, action, reward, done, batch_size=32):
        self.__memory.remember([state, next_state, action, reward, done])
        if self.__memory.get_memory_size() < batch_size:
            return 0.0
        
        minibatch = [self.__memory.random_access() for i in range(batch_size)]
        input_states = torch.Tensor([values[0] / 255. for values in minibatch]).permute(0, 3, 1, 2).to(self.__device)
        new_states = torch.Tensor([values[1] / 255. for values in minibatch]).permute(0, 3, 1, 2).to(self.__device)
        input_qs_list = self.__model(input_states)
        future_qs_list = self.__model(new_states)
        target_q = torch.zeros((batch_size, self.__env.get_number_of_actions())).to(self.__device)

        for i, mem_item in enumerate(minibatch):
            _, _, action, reward, done = mem_item
            if done:
                target_q_value = reward
            else:
                future_reward = torch.max(future_qs_list[i])
                target_q_value = reward + self.__discount * future_reward

            target_q[i] = input_qs_list[i]
            target_q[i, action] = target_q_value
        
        self.__optimizer.zero_grad()
        target_q = torch.clamp(target_q, -3, 3)
        output_q = self.__model(input_states)
        loss = self.__criterion(output_q, target_q)
        loss.backward()
        self.__optimizer.step()
        return loss.item()

    def train(self, n_epochs=20, batch_size=32, output_path="./tmp"):
        loss = 0
        for e in range(n_epochs):
            state = self.__env.reset()
            done = False
            score = 0
            while not done:
                action = self.act(state, is_training=True)
                next_state, reward, done, info = self.__env.step(action)
                score += reward
                loss = self.reinforce(state, next_state, action, reward, done)
                state = next_state
                print("Epoch {:03d}/{:03d} | Loss {:.4f} | Score {}"
                      .format(e, n_epochs, loss, score))
            self.__env.draw_video(output_path + "/" + str(e))
            self.save()

    def save(self):
        torch.save(self.__model, self.__model_path)

    def load(self):
        self.__model = torch.load(self.__model_path)
