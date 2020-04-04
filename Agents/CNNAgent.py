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
            nn.Conv2d(3, 16, 3, padding=1),              # 8 x 8 x 16
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),                             # 4 x 4 x 16
            nn.Conv2d(16, 32, 3, padding=1),             # 4 x 4 x 32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                             # 2 x 2 x 32
            nn.Conv2d(32, 64, 3, padding=1),             # 2 x 2 x 64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                             # 1 x 1 x 64
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


class CNNAgent(BaselineAgent):
    def __init__(self, env, model_path,  memory_size=100000, discount=0.9, train=False, eps_start=0.5, eps_decay=0.99):
        super().__init__(env)
        self.__env = env
        self.__memory = Memory(memory_size)
        self.__discount = discount
        self.__device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.__train = train
        self.__model_path = model_path
        self.__eps_decay = eps_decay
        self.set_epsilon(eps_start)
        self.create_model()

    def create_model(self):
        if self.__train:
            nA = self.__env.get_number_of_actions()
            self.__model = DQNet(nA)
            self.__optimizer = optim.Adam(self.__model.parameters(), lr=0.001)
            self.__criterion = nn.MSELoss()
        else:
            self.load()
        self.__model.to(self.__device)

    def learned_act(self, state):
        state = torch.Tensor([state]).permute(0, 3, 1, 2).to(self.__device)
        with torch.no_grad():
            logits = self.__model(state)
            probabilities = F.softmax(logits, dim=-1)[0].detach().cpu().numpy()
        return np.random.choice(list(range(self.__env.get_number_of_actions())), p=probabilities)

    def reinforce(self, state, next_state, action, reward, done, batch_size):
        self.__memory.remember([state, next_state, action, reward, done])
        if self.__memory.get_memory_size() < batch_size:
            return 0.0
        
        minibatch = self.__memory.random_access(batch_size)
        input_states = torch.Tensor([values[0] for values in minibatch]).permute(0, 3, 1, 2).to(self.__device)
        next_states = torch.Tensor([values[1] for values in minibatch]).permute(0, 3, 1, 2).to(self.__device)
        input_qs_list = self.__model(input_states)
        future_qs_list = self.__model(next_states)
        target_q = torch.zeros((batch_size, self.__env.get_number_of_actions())).to(self.__device)

        for i, mem_item in enumerate(minibatch):
            _, _, action, reward, done = mem_item
            if done:
                target_q_value = reward
            else:
                target_q_value = reward + self.__discount * torch.max(future_qs_list[i])
            target_q[i] = input_qs_list[i]
            target_q[i, action] = target_q_value
        
        self.__optimizer.zero_grad()
        output_q = self.__model(input_states)
        loss = self.__criterion(output_q, target_q)
        loss.backward()
        self.__optimizer.step()
        return loss.item()

    def train(self, n_epochs, batch_size, output_path="./tmp"):
        for e in range(n_epochs):
            state = self.__env.reset()
            done = False
            score = 0
            loss = 0
            number_steps = 0
            while not done:
                action = self.act(state, is_training=True)
                next_state, reward, done, info = self.__env.step(action)
                score += reward
                loss += self.reinforce(state, next_state, action, reward, done, batch_size)
                number_steps += 1
                state = next_state
            loss /= number_steps
            print("Epoch {:03d}/{:03d} | Loss {:3.4f} | Score = {:03d} | Epsilon {:.4f}"
                    .format(e, n_epochs, loss, score, self.epsilon))
            self.__env.draw_video(output_path + "/" + str(e))
            self.save()
            self.set_epsilon(self.epsilon * self.__eps_decay)

    def save(self):
        torch.save(self.__model, self.__model_path)

    def load(self):
        self.__model = torch.load(self.__model_path)
