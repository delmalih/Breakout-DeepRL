###########
# Imports #
###########


import numpy as np
from tqdm import tqdm


###############
# Agent Class #
###############

class BaselineAgent(object):
    def __init__(self, env):
        self.env = env
    
    def set_epsilon(self, eps):
        self.epsilon = eps

    def act(self, state, is_training=False):
        if is_training:
            if np.random.rand() <= self.epsilon:
                a = self.env.sample_action()
            else:
                a = self.learned_act(state)
        else:
            a = self.learned_act(state)
        return a

    def play(self, epochs):
        for e in tqdm(range(epochs)):
            state = self.env.reset()
            done = False
            while not done:
                self.env.render()
                action = self.act(state)
                state, _, done, _ = self.env.step(action)

    def learned_act(self, state):
        pass

    def reinforce(self, *args):
        pass

    def train(self, *args):
        pass
