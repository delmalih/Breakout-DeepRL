###########
# Imports #
###########

from .BaselineAgent import BaselineAgent


###############
# Agent Class #
###############

class RandomAgent(BaselineAgent):
    def __init__(self, env, epsilon=0.1):
        super().__init__(env, epsilon=epsilon)
        self.__env = env
        self.set_epsilon(epsilon)

    def learned_act(self, state):
        return self.__env.sample_action()
