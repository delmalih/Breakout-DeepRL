###########
# Imports #
###########

import gym
import skvideo.io
import numpy as np


###############
# Environment #
###############

class Environment(object):
    def __init__(self):
        self.__env = gym.make('Breakout-v0')
        self.reset()

    def reset(self):
        self.__video = []
        self.__env.reset()

    def step(self, action):
        self.__video.append(self.get_current_state())
        return self.__env.step(action)

    def render(self):
        self.__env.render()

    def close(self):
        self.__env.close()

    def sample_action(self):
        return self.__env.action_space.sample()

    def get_current_state(self):
        return self.__env.env.ale.getScreenRGB()

    def draw_video(self, output_path):
        self.__video = np.array(self.__video)
        skvideo.io.vwrite(
            "{}.mp4".format(output_path),
            self.__video,
        )

    def get_number_of_actions(self):
        return len(self.__env.env.ale.getMinimalActionSet())

    def get_state_shape(self):
        return self.get_current_state().shape
