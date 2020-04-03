###########
# Imports #
###########

from .BaselineAgent import BaselineAgent


###############
# Agent Class #
###############

class RandomAgent(BaselineAgent):
    def __init__(self, env):
        super().__init__(env)
        self.__env = env

    def learned_act(self, state):
        return self.__env.sample_action()
