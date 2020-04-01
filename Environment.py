###########
# Imports #
###########

import gym
import gym_snake
import skvideo.io
import numpy as np


###############
# Environment #
###############

class Environment(object):
    def __init__(self):
        self.__env = gym.make("snake-v0")
        self.reset()

    def reset(self):
        self.__env.unit_gap = 0
        state = self.__env.reset()
        self.__video = [state]
        return state

    def step(self, action):
        action = int(action)
        state, reward, done, info = self.__env.step(action)
        self.__video.append(state)
        return state, reward, done, info

    def render(self):
        self.__env.render()

    def close(self):
        self.__env.close()

    def sample_action(self):
        return self.__env.action_space.sample()

    def draw_video(self, output_path):
        self.__video = np.array(self.__video)
        skvideo.io.vwrite(
            "{}.mp4".format(output_path),
            self.__video,
        )

    def get_number_of_actions(self):
        return self.__env.action_space.shape[0]

    def get_state_shape(self):
        return self.__video[0].shape
