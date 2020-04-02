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
            print("Epoch = {:4d} | Current score = {:.2f}".format(e, score))
        print("Average score: {}".format(1. * total_score / epochs))

    def learned_act(self, state):
        pass

    def reinforce(self, *args):
        pass

    def train(self, *args):
        pass
