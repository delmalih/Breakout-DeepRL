##########
# Import #
##########


import torch

import constants


################
# Memory Class #
################

class Memory(object):
    def __init__(self, max_memory=constants.MAX_MEMORY):
        self.max_memory = max_memory
        self.device = constants.DEVICE
        self.states = torch.empty((0, constants.N_CHANNELS, constants.SIZE, constants.SIZE)).to(self.device)
        self.next_states = torch.empty((0, constants.N_CHANNELS, constants.SIZE, constants.SIZE)).to(self.device)
        self.actions = torch.empty((0,), dtype=torch.long).to(self.device)
        self.rewards = torch.Tensor((0,)).to(self.device)
        self.dones = torch.Tensor((0,)).to(self.device)

    def remember(self, state, next_state, action, reward, done):
        self.states = torch.cat((self.states, state), dim=0)
        self.next_states = torch.cat((self.next_states, next_state), dim=0)
        self.actions = torch.cat((self.actions, action), dim=0)
        self.rewards = torch.cat((self.rewards, reward), dim=0)
        self.dones = torch.cat((self.dones, done), dim=0)
        if self.states.size(0) > self.max_memory:
            excedent = self.states.size(0) - self.max_memory
            self.states = self.states[excedent:]
            self.next_states = self.next_states[excedent:]
            self.actions = self.actions[excedent:]
            self.rewards = self.rewards[excedent:]
            self.dones = self.dones[excedent:]

    def random_access(self, n):
        indexes = torch.randint(self.states.size(0), size=(n,))
        states = self.states[indexes]
        next_states = self.next_states[indexes]
        actions = self.actions[indexes]
        rewards = self.rewards[indexes]
        dones = self.dones[indexes]
        return states, next_states, actions, rewards, dones
