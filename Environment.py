###########
# Imports #
###########


import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt

import constants


###############
# Environment #
###############

class Environment(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self, grid_size=constants.SIZE, init_length=constants.INIT_LENGTH):
        self.grid_size = grid_size
        self.init_length = init_length
        self.action_space = spaces.Discrete(4)
        self.snake_coords = None
        self.apple_coords = None
        self.heading = None
        plt.ion()

    def reset(self):
        self.heading = np.array([1, 0])
        self.snake_coords = [np.array([self.grid_size // 2, self.grid_size // 2])]
        for _ in range(self.init_length - 1):
            self._add_snake_cell()
        self._generate_apple()
        return self._get_rgb_screen()

    def step(self, action):
        assert self.action_space.contains(action)
        new_heading = constants.ACTIONS_TO_HEADINGS[action]
        if (self.heading + new_heading != 0).any():
            self.heading = new_heading
        done = self._update_snake()
        eaten = self._update_apple()
        reward = self._compute_reward(done, eaten)
        if eaten:
            done = done or self._add_snake_cell()
        return self._get_rgb_screen(), reward, done, {}
    
    def render(self):
        plt.figure(0)
        plt.clf()
        plt.axis('off')
        plt.imshow(self._get_rgb_screen())
        plt.pause(0.01)

    def sample_action(self):
        return self.action_space.sample()

    def get_number_of_actions(self):
        return self.action_space.n

    def _generate_apple(self):
        possible_coords = np.array([(i, j)
            for i in range(self.grid_size)
            for j in range(self.grid_size)
            if not self._is_in_snake((i, j))
        ])
        self.apple_coords = possible_coords[np.random.randint(0, len(possible_coords))]

    def _is_in_snake(self, coords, include_head=True):
        snake_coords = self.snake_coords if include_head else self.snake_coords[1:]
        for snake_coord in snake_coords:
            if coords[0] == snake_coord[0] and coords[1] == snake_coord[1]:
                return True
        return False

    def _get_rgb_screen(self):
        screen = np.zeros((self.grid_size + 2, self.grid_size + 2, 3))
        screen[self.apple_coords[0] + 1, self.apple_coords[1] + 1] = (0., 1., 0.)
        for i, j in self.snake_coords:
            if self._is_valid_coords((i, j)):
                screen[i + 1, j + 1] = (0., 0., 1.)
        head_coords = self.snake_coords[0]
        if self._is_valid_coords(head_coords):
            screen[head_coords[0] + 1, head_coords[1] + 1] = (1., 0., 0.)
        screen[0, :, :] = screen[-1, :, :] = .5
        screen[:, 0, :] = screen[:, -1, :] = .5
        return screen

    def _is_valid_coords(self, coords):
        x, y = coords
        return x >= 0 and y >= 0 and x < self.grid_size and y < self.grid_size

    def _update_snake(self):
        for i in range(len(self.snake_coords)-1, 0, -1):
            self.snake_coords[i][0] = self.snake_coords[i-1][0]
            self.snake_coords[i][1] = self.snake_coords[i-1][1]
        self.snake_coords[0] += self.heading
        done = not self._is_valid_coords(self.snake_coords[0]) or \
               self._is_in_snake(self.snake_coords[0], include_head=False)
        return done
    
    def _update_apple(self):
        eaten = False
        if (self.snake_coords[0] == self.apple_coords).all():
            eaten = True
            self._generate_apple()
        return eaten

    def _add_snake_cell(self):
        new_cell_coords = self.snake_coords[-1] - self.heading
        if not self._is_valid_coords(new_cell_coords):
            i, j = self.snake_coords[-1]
            possible_new_cell_coords = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
            possible_new_cell_coords = list(filter(lambda coords: self._is_valid_coords(coords) and not self._is_in_snake(coords), possible_new_cell_coords))
            if len(possible_new_cell_coords) == 0:
                return True
            new_cell_coords = np.array(possible_new_cell_coords[np.random.randint(0, len(possible_new_cell_coords))])
        self.snake_coords.append(new_cell_coords)
        return False

    def _compute_reward(self, done, eaten):
        if done:
            return -1.
        if eaten:
            return +1.
        return -.01
