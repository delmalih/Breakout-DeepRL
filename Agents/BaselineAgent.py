###########
# Imports #
###########


import numpy as np


###############
# Agent Class #
###############

class BaselineAgent(object):
    def __init__(self, env):
        self.__env = env
    
    def set_epsilon(self, eps):
        self.epsilon = eps

    def act(self, state, is_training=False):
        if is_training:
            if np.random.rand() <= self.epsilon:
                a = self.__env.sample_action()
            else:
                a = self.learned_act(state)
        else:
            a = self.learned_act(state)

        return a

    def play(self, epochs, output_path='', verbose=True):
        total_score = 0
        for e in range(epochs):
            score = 0
            done = False
            state = self.__env.reset()
            while not done:
                action = self.act(state)
                state, reward, done, _ = self.__env.step(action)
                score += reward
            self.__env.draw_video(output_path + "/" + str(e))
            total_score += score
            print("Epoch = {:06d} | Current score = {:06.2f} | N Steps = {:06d}".format(e, score, len(self.__env.get_video())))
        print("Average score: {}".format(1. * total_score / epochs))

    def learned_act(self, state):
        pass

    def reinforce(self, *args):
        pass

    def train(self, *args):
        pass
