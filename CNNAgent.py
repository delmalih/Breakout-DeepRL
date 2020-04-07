##########
# Import #
##########

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import constants
from BaselineAgent import BaselineAgent
from Memory import Memory
from CNNModel import CNNModel


#############
# CNN Class #
#############


class CNNAgent(BaselineAgent):
    def __init__(self, env, model_path, is_training=False):
        super().__init__(env)
        self.env = env
        self.model_path = model_path
        self.is_training = is_training
        self.memory = Memory()
        self.discount = constants.DISCOUNT
        self.device = constants.DEVICE
        self.set_epsilon(constants.EPS_START)
        self.eps_min = constants.EPS_MIN
        self.eps_decay = constants.EPS_DECAY
        self.lr = constants.LEARNING_RATE
        self.create_model()

    def create_model(self):
        self.load()
        self.model.to(self.device)
        if self.is_training:
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0, weight_decay=1e-4)

    def learned_act(self, state):
        with torch.no_grad():
            q_values = self.model(state.unsqueeze(0))
            action = torch.argmax(q_values, dim=-1)[0]
        return action

    def train(self, n_epochs, batch_size, output_path="./tmp"):
        for e in range(n_epochs):
            score = loss = 0
            state = self.env.reset()
            done = False
            while not done:
                action = self.act(state, is_training=True)
                next_state, reward, done, _ = self.env.step(action)
                score += reward
                self.memory.remember(state, next_state, action, reward, done)
                loss += self.reinforce(*self.memory.random_access(batch_size))
                state = next_state
            self.set_epsilon(self.epsilon * self.eps_decay if self.epsilon > self.eps_min else self.eps_min)
            if (e + 1) % constants.SAVE_FREQ == 0:
                print("Epoch {:03d}/{:03d} | Epsilon {:.4f} | Loss {:3.4f} | Score {:04.2f}"
                        .format(e + 1, n_epochs, self.epsilon, loss / constants.MAX_TIME, score))
                self.env.draw_video(output_path + "/" + str(e + 1))
                self.save()

    def reinforce(self, state, next_state, action, reward, done):
        self.optimizer.zero_grad()
        input_q_values = self.model(state)
        next_q_values = self.model(next_state)
        target_q = input_q_values.clone()
        for k in range(len(done)):
            target_q[k, action[k]] = reward[k]
            if not done[k]:
                target_q[k, action[k]] += constants.DISCOUNT * next_q_values[k].max()
        loss = torch.mean(torch.pow(input_q_values - target_q, 2))
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def save(self):
        torch.save(self.model, self.model_path)

    def load(self):
        if os.path.exists(self.model_path):
            self.model = torch.load(self.model_path, map_location=self.device)
        else:
            nA = self.env.get_number_of_actions()
            self.model = CNNModel(nA)
