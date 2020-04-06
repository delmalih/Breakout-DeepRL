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

    def play(self, epochs, output_path=''):
        state = self.env.reset()
        for e in tqdm(range(epochs)):
            action = self.act(state)
            state, _, _, _ = self.env.step(action)
        self.env.draw_video(output_path + "/" + str(e))

    def learned_act(self, state):
        pass

    def reinforce(self, *args):
        pass

    def train(self, *args):
        pass
