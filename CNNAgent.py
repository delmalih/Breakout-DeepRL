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
        if self.is_training:
            nA = self.env.get_number_of_actions()
            self.model = CNNModel(nA)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        else:
            self.load()
        self.model.to(self.device)

    def learned_act(self, state):
        state = torch.Tensor(state).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
            action = torch.argmax(q_values, dim=-1)
        return action

    def train(self, n_epochs, batch_size, output_path="./tmp"):
        state = self.env.reset()
        score = loss = 0
        print(n_epochs)
        for e in range(n_epochs):
            action = self.act(state, is_training=True)
            next_state, reward, done, _ = self.env.step(action)
            score += reward.sum()
            loss += self.reinforce(state, next_state, action, reward, done, batch_size)
            state = next_state
            self.set_epsilon(self.epsilon * self.eps_decay if self.epsilon > self.eps_min else self.eps_min)
            if (e + 1) % constants.SAVE_FREQ == 0:
                print("Epoch {:03d}/{:03d} | Epsilon {:.4f} | Loss {:3.4f} | Score {:04.2f}"
                        .format(e + 1, n_epochs, self.epsilon, loss, score))
                self.env.draw_video(output_path + "/" + str(e + 1))
                self.env._reset_video()
                self.save()
                score = loss = 0

    def reinforce(self, state, next_state, action, reward, done, batch_size):
        self.memory.remember(state, next_state, action, reward, done)
        input_states, next_states, actions, rewards, dones = self.memory.random_access(batch_size)
        self.optimizer.zero_grad()
        input_q_values = self.model(input_states)
        next_q_values = self.model(next_states)
        target_q = torch.zeros(input_q_values.shape).to(self.device)
        target_q[dones == 1, actions[dones == 1]] = rewards[dones == 1]
        target_q[dones == 0, actions[dones == 0]] = (rewards + torch.max(next_q_values, dim=-1).values)[dones == 0]
        loss = torch.mean(torch.pow(input_q_values - target_q, 2))
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def save(self):
        torch.save(self.model, self.model_path)

    def load(self):
        if os.path.exists(self.model_path):
            self.model = torch.load(self.model_path)
        else:
            nA = self.env.get_number_of_actions()
            self.model = CNNModel(nA)
