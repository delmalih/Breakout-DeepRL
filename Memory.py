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
        self.cpu_device = constants.CPU_DEVICE
        self.gpu_device = constants.DEVICE
        self.states = torch.empty((0, constants.N_CHANNELS, constants.SIZE, constants.SIZE)).to(self.cpu_device)
        self.next_states = torch.empty((0, constants.N_CHANNELS, constants.SIZE, constants.SIZE)).to(self.cpu_device)
        self.actions = torch.empty((0,), dtype=torch.long).to(self.cpu_device)
        self.rewards = torch.Tensor((0,)).to(self.cpu_device)
        self.dones = torch.Tensor((0,)).to(self.cpu_device)

    def remember(self, state, next_state, action, reward, done):
        self.states = torch.cat((self.states, state.to(self.cpu_device)), dim=0)
        self.next_states = torch.cat((self.next_states, next_state.to(self.cpu_device)), dim=0)
        self.actions = torch.cat((self.actions, action.to(self.cpu_device)), dim=0)
        self.rewards = torch.cat((self.rewards, reward.to(self.cpu_device)), dim=0)
        self.dones = torch.cat((self.dones, done.to(self.cpu_device)), dim=0)
        if self.states.size(0) > self.max_memory:
            excedent = self.states.size(0) - self.max_memory
            self.states = self.states[excedent:]
            self.next_states = self.next_states[excedent:]
            self.actions = self.actions[excedent:]
            self.rewards = self.rewards[excedent:]
            self.dones = self.dones[excedent:]

    def random_access(self, n):
        indexes = torch.randint(self.states.size(0), size=(n,))
        states = self.states[indexes].to(self.gpu_device)
        next_states = self.next_states[indexes].to(self.gpu_device)
        actions = self.actions[indexes].to(self.gpu_device)
        rewards = self.rewards[indexes].to(self.gpu_device)
        dones = self.dones[indexes].to(self.gpu_device)
        return states, next_states, actions, rewards, dones
