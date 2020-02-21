###########
# Imports #
###########

import numpy as np


###############
# Agent Class #
###############

class BaselineAgent(object):
    def __init__(self, env, epsilon=0.1):
        self.__env = env
        self.set_epsilon(epsilon)

    def set_epsilon(self, epsilon):
        self.__epsilon = epsilon

    def act(self, state, is_training=False):
        if is_training:
            if np.random.rand() <= self.__epsilon:
                a = np.random.randint(0, self.__env.nA, size=1)[0]
            else:
                a = self.learned_act(state)
        else:
            a = self.learned_act(state)

        return a

    def learned_act(self, state):
        pass

    def reinforce(self):
        pass
