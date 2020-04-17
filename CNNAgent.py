##########
# Import #
##########

import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import namedtuple

import torch
import torch.optim as optim
import torch.nn.functional as F

import constants
from Memory import Memory
from CNNModel import CNNModel
from BaselineAgent import BaselineAgent


#############
# CNN Class #
#############


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


class CNNAgent(BaselineAgent):
    def __init__(self, env, model_path, is_training=False):
        super().__init__(env)
        self.env = env
        self.model_path = model_path
        self.is_training = is_training
        self.score_history = []
        self.score_history_ma = []
        self.memory = Memory()
        self.discount = constants.DISCOUNT
        self.device = constants.DEVICE
        self.set_epsilon(constants.EPS_START)
        self.eps_min = constants.EPS_MIN
        self.eps_decay = constants.EPS_DECAY
        self.create_model()

    def create_model(self):
        self.load()
        self.model.to(self.device)
        self.target_model.to(self.device)
        self.target_model.eval()
        if self.is_training:
            self.optimizer = optim.RMSprop(self.model.parameters())

    def learned_act(self, state):
        with torch.no_grad():
            q_values = self.model(state)[0]
            action = torch.argmax(q_values).item()
        return action

    def train(self, n_epochs=constants.N_TRAIN_EPOCHS, batch_size=constants.BATCH_SIZE):
        for e in range(n_epochs):
            
            self.env.reset()
            last_screen = self.state2tensor(self.env._get_rgb_screen())
            curr_screen = self.state2tensor(self.env._get_rgb_screen())
            state = torch.cat((last_screen, curr_screen), dim=1)
            done = False
            score = loss = 0
            
            while not done:
                # Render env
                self.env.render()

                # Select and perform action
                action = self.act(state, is_training=True)
                _, reward, done, _ = self.env.step(action)
                score += reward
                
                # Observe new state
                last_screen = curr_screen
                curr_screen = self.state2tensor(self.env._get_rgb_screen())
                if not done:
                    next_state = torch.cat((last_screen, curr_screen), dim=1)
                else:
                    next_state = None
                
                # Store the transition in memory
                action = torch.tensor([action], device=self.device)
                reward = torch.tensor([float(reward)], device=self.device)
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state
                
                # Perform one step of the optimization (on the target network)
                loss += self.reinforce()

                if done:
                    self.score_history.append(score)
                    self.score_history_ma.append(np.mean(self.score_history[-100:]))
                    self.plot_score()

            # Update epsilon
            self.set_epsilon(self.epsilon * self.eps_decay if self.epsilon > self.eps_min else self.eps_min)
            
            # Prints
            if (e + 1) % constants.PRINT_FREQ == 0:
                print("Epoch {:03d}/{:03d} | Epsilon {:.4f} | Loss {:3.4f} | Score {:04.2f}"
                        .format(e + 1, n_epochs, self.epsilon, loss, score))
            
            # Save
            if (e + 1) % constants.SAVE_FREQ == 0:
                self.target_model.load_state_dict(self.model.state_dict())
                self.save()

    def play(self, epochs):
        for e in tqdm(range(epochs)):
            
            self.env.reset()
            last_screen = self.state2tensor(self.env._get_rgb_screen())
            curr_screen = self.state2tensor(self.env._get_rgb_screen())
            state = torch.cat((last_screen, curr_screen), dim=1)
            done = False
            
            while not done:
                # Render env
                self.env.render()

                # Select and perform action
                action = self.act(state)
                _, _, done, _ = self.env.step(action)
                
                # Update state
                last_screen = curr_screen
                curr_screen = self.state2tensor(self.env._get_rgb_screen())
                state = torch.cat((last_screen, curr_screen), dim=1)

    def reinforce(self, batch_size=constants.BATCH_SIZE):
        if len(self.memory) < batch_size:
            return 0.
        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action).unsqueeze(-1)
        reward_batch = torch.cat(batch.reward)
        state_action_values = self.model(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_model(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.discount) + reward_batch
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss.item()

    def state2tensor(self, state):
        return torch.tensor(state).permute(2, 0, 1).to(self.device).float().unsqueeze(0)

    def plot_score(self):
        plt.figure(1)
        plt.clf()
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.plot(self.score_history)
        plt.plot(self.score_history_ma)

    def save(self):
        torch.save(self.model, self.model_path)

    def load(self):
        if os.path.exists(self.model_path):
            self.model = torch.load(self.model_path, map_location=self.device)
            self.target_model = torch.load(self.model_path, map_location=self.device)
        else:
            nA = self.env.get_number_of_actions()
            self.model = CNNModel(nA)
            self.target_model = CNNModel(nA)
        self.target_model.load_state_dict(self.model.state_dict())
