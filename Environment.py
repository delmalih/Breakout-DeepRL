###########
# Imports #
###########


import cv2
import skvideo.io
import numpy as np


###############
# Environment #
###############


class Environment(object):
    def __init__(self, game_width=30, game_height=30, initial_length=2):
        self.__game_width = game_width
        self.__game_height = game_height
        self.__initial_length = initial_length
        self.__action_matrix = [
            np.array([[+1, 0], [0, +1]]),
            np.array([[0, -1], [+1, 0]]),
            np.array([[0, +1], [-1, 0]]),
        ]
        self.reset()

    def reset(self):
        self.generate_snake()
        self.generate_food()
        state = self.draw()
        self.__video = [state]
        return state

    def generate_food(self):
        food_x = np.random.randint(0, self.__game_width - 1)
        food_y = np.random.randint(0, self.__game_height - 1)
        while self.is_in_snake((food_x, food_y)):
            food_x = np.random.randint(0, self.__game_width - 1)
            food_y = np.random.randint(0, self.__game_height - 1)
        self.__food_coordinates = (food_x, food_y)

    def generate_snake(self):
        self.__snake_coordinates = []
        head_x = np.random.randint(0, self.__game_width - 1)
        head_y = np.random.randint(0, self.__game_height - 1)
        self.__snake_coordinates.append(np.array((head_x, head_y)))
        for _ in range(self.__initial_length - 1):
            possible_coords = []
            if head_x > 0 and not self.is_in_snake((head_x - 1, head_y)):
                possible_coords.append(np.array((head_x - 1, head_y)))
            if head_x < self.__game_width - 1 and not self.is_in_snake((head_x + 1, head_y)):
                possible_coords.append(np.array((head_x + 1, head_y)))
            if head_y > 0 and not self.is_in_snake((head_x, head_y - 1)):
                possible_coords.append(np.array((head_x, head_y + 1)))
            if head_y < self.__game_height - 1 and not self.is_in_snake((head_x, head_y + 1)):
                possible_coords.append(np.array((head_x, head_y + 1)))
            self.__snake_coordinates.append(possible_coords[np.random.randint(0, len(possible_coords))])
        self.__snake_heading = [np.array((1, 0)), np.array((0, 1)), np.array((-1, 0)), np.array((0, -1))][np.random.randint(0, 4)]

    def draw(self):
        state = np.full((self.__game_height + 2, self.__game_width + 2, 3), 1.0)
        state[self.__food_coordinates[0] + 1, self.__food_coordinates[1] + 1] = (0., 1., 0.)
        for snake_coords in self.__snake_coordinates:
            if snake_coords[0] >= 0 and snake_coords[0] < state.shape[0] and snake_coords[1] >= 0 and snake_coords[1] < state.shape[1]:
                state[snake_coords[0] + 1, snake_coords[1] + 1] = 0.
        snake_head_coords = self.__snake_coordinates[0]
        if snake_head_coords[0] >= 0 and snake_head_coords[0] < state.shape[0] and snake_head_coords[1] >= 0 and snake_head_coords[1] < state.shape[1]:
            state[snake_head_coords[0] + 1, snake_head_coords[1] + 1] = (1., 0., 0.)
        state[0,  :] = 0.
        state[-1, :] = 0.
        state[:,  0] = 0.
        state[:, -1] = 0.
        return state

    def step(self, action):
        # 0 --> TURN 0° || 1 --> TURN 90° || 2 --> TURN -90°
        action = int(action)
        reward, done = self.apply_action(action)
        state = self.draw()
        self.__video.append(state)
        return state, reward, done, {}
    
    def apply_action(self, action):
        self.__snake_heading = self.__action_matrix[action].dot(self.__snake_heading)
        snake_head_coords = self.__snake_coordinates[0] + self.__snake_heading
        for i in range(len(self.__snake_coordinates) - 1, 0, -1):
            self.__snake_coordinates[i] = self.__snake_coordinates[i-1]
        self.__snake_coordinates[0] = snake_head_coords
        if (self.__snake_coordinates[0] == self.__food_coordinates).all():
            reward = 1
            self.generate_food()
            done = self.add_cell_to_snake()
        elif self.is_game_done():
            reward = -1
            done = True
        else:
            reward = 0.1 * np.abs(self.__snake_coordinates[0] - self.__food_coordinates).sum() / (self.__game_width + self.__game_height)
            done = False
        return reward, done

    def add_cell_to_snake(self):
        new_coords = None
        done = False
        last1_x, last1_y = self.__snake_coordinates[-1]
        last2_x, last2_y = self.__snake_coordinates[-2]
        if last1_x == last2_x and (2 * last1_y - last2_y) >= 0 and (2 * last1_y - last2_y) < self.__game_height:
            new_coords = (last1_x, 2 * last1_y - last2_y)
        elif last1_y == last2_y and (2 * last1_x - last2_x) >= 0 and (2 * last1_x - last2_x) < self.__game_height:
            new_coords = (2 * last1_x - last2_x, last1_y)
        else:
            possible_coordinates = []
            if not self.is_in_snake((last1_x - 1, last1_y)) and last1_x > 0:
                possible_coordinates.append((last1_x - 1, last1_y))
            if not self.is_in_snake((last1_x + 1, last1_y)) and last1_x < self.__game_width - 1:
                possible_coordinates.append((last1_x + 1, last1_y))
            if not self.is_in_snake((last1_x, last1_y - 1)) and last1_y > 0:
                possible_coordinates.append((last1_x, last1_y - 1))
            if not self.is_in_snake((last1_x, last1_y + 1)) and last1_y < self.__game_height - 1:
                possible_coordinates.append((last1_x, last1_y + 1))
            if len(possible_coordinates) == 0:
                done = True
            else:
                new_coords = possible_coordinates[np.random.randint(0, len(possible_coordinates))]
        if new_coords is not None:
            self.__snake_coordinates.append(np.array(new_coords))
        return done

    def is_game_done(self):
        snake_head_coords = self.__snake_coordinates[0]
        self.__snake_coordinates[0] = (None, None)
        is_head_in_tail = self.is_in_snake(snake_head_coords)
        self.__snake_coordinates[0] = snake_head_coords
        return (
            is_head_in_tail or
            snake_head_coords[0] < 0 or
            snake_head_coords[1] < 0 or
            snake_head_coords[0] > self.__game_width - 1 or
            snake_head_coords[1] > self.__game_height -1
        )
    
    def is_in_snake(self, coords):
        x, y = coords
        for (x_s, y_s) in self.__snake_coordinates:
            if x == x_s and y == y_s:
                return True
        return False

    def sample_action(self):
        return np.random.randint(0, self.get_number_of_actions())

    def draw_video(self, output_path):
        self.__video = (np.array(self.__video) * 255).astype(np.uint8)
        writer = skvideo.io.FFmpegWriter("{}.mp4".format(output_path))
        for frame in self.__video:
            frame = cv2.resize(frame, (500, 500), interpolation=cv2.INTER_NEAREST)
            for _ in range(5):
                writer.writeFrame(frame)
        writer.close()

    def get_number_of_actions(self):
        return 3

    def get_state_shape(self):
        return (self.__game_width, self.__game_height)

    def get_video(self):
        return self.__video
