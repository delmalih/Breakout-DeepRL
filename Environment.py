###########
# Imports #
###########


import cv2
import skvideo.io
import numpy as np
import math

import torch
import torch.nn.functional as F

import constants


###############
# Environment #
###############


class Environment(object):
    def __init__(self, size=constants.SIZE, max_time=constants.MAX_TIME):
        self.size = size + 2 * constants.SIGHT
        self.max_time = max_time
        self.device = constants.DEVICE
        self.n_channels = constants.N_CHANNELS
        self.reset()

    def reset(self):
        self._reset_video()
        self._reset_board()
        return self.get_state()

    def _reset_video(self):
        self.video = []
        self.scores = []

    def _reset_board(self):
        self.time = 0
        self.board = torch.zeros((self.n_channels, self.size, self.size)).to(self.device)
        self._reset_head()
        self._reset_bonus()
        self._reset_malus()

    def _reset_head(self):
        self.head_coords = torch.randint(3, self.size - 3, size=(2, 1)).to(self.device)[:, 0]
        self.board[constants.HEAD_CHANNEL] = 0.
        self.board[constants.HEAD_CHANNEL, 0:constants.SIGHT, :] = -1.
        self.board[constants.HEAD_CHANNEL, -constants.SIGHT:, :] = -1.
        self.board[constants.HEAD_CHANNEL, :, 0:constants.SIGHT] = -1.
        self.board[constants.HEAD_CHANNEL, :, -constants.SIGHT:] = -1.
        self.board[constants.HEAD_CHANNEL, self.head_coords[0], self.head_coords[1]] = 1.

    def _reset_bonus(self):
        bonus = torch.distributions.binomial.Binomial(1, torch.full((self.size, self.size), constants.FILL_PERC)).sample().to(self.device)
        bonus[self.board[constants.HEAD_CHANNEL] != 0] = 0.
        bonus[self.board[constants.MALUS_CHANNEL] != 0] = 0.
        self.board[constants.BONUS_CHANNEL] = bonus
    
    def _reset_malus(self):
        malus = torch.distributions.binomial.Binomial(1, torch.full((self.size, self.size), constants.FILL_PERC)).sample().to(self.device)
        malus[self.board[constants.HEAD_CHANNEL] != 0] = 0.
        malus[self.board[constants.BONUS_CHANNEL] != 0] = 0.
        self.board[constants.MALUS_CHANNEL] = malus

    def get_state(self):
        x, y = self.head_coords
        s = constants.SIGHT
        return self.board[:, x - s:x + s + 1, y - s:y + s + 1]

    def step(self, action):
        self.time += 1
        self._update_head(action)
        reward = self._compute_reward_and_update()
        done = self.time >= self.max_time
        self.video.append(self.render())
        self.scores.append(reward if len(self.scores) == 0 else self.scores[-1] + reward)
        return self.get_state(), reward, done, {}
    
    def _update_head(self, action):
        self.board[constants.HEAD_CHANNEL, self.head_coords[0], self.head_coords[1]] = 0.
        displacement = constants.ACTIONS_TO_VECT[action]
        new_coords = self.head_coords + displacement
        if self.board[constants.HEAD_CHANNEL, new_coords[0], new_coords[1]] == -1:
            new_coords = self.head_coords - displacement
        self.head_coords = new_coords
        self.board[constants.HEAD_CHANNEL, self.head_coords[0], self.head_coords[1]] = 1.
    
    def _compute_reward_and_update(self):
        # Get envs
        head_board  = self.board[constants.HEAD_CHANNEL,  :, :]
        bonus_board = self.board[constants.BONUS_CHANNEL, :, :]
        malus_board = self.board[constants.MALUS_CHANNEL, :, :]

        # Get eaten
        eaten_bonus = (head_board * bonus_board).nonzero()
        eaten_malus = (head_board * malus_board).nonzero()

        # Update eaten
        self.board[constants.BONUS_CHANNEL, eaten_bonus[:, 0], eaten_bonus[:, 1]] = 0.
        self.board[constants.MALUS_CHANNEL, eaten_malus[:, 0], eaten_malus[:, 1]] = 0.

        # Compute reward
        reward = eaten_bonus.size(0) * constants.BONUS_REWARD + eaten_malus.size(0) * constants.MALUS_REWARD

        return reward

    def sample_action(self):
        return torch.randint(self.get_number_of_actions(), size=(1,)).to(self.device)[0]

    def render(self):
        sight = constants.SIGHT
        frame = self.board.permute(1, 2, 0).clone().cpu().detach().numpy()
        frame = frame[:, :, [constants.MALUS_CHANNEL, constants.BONUS_CHANNEL, constants.HEAD_CHANNEL]]
        frame[0:sight, :, :] = 1.
        frame[-sight:, :, :] = 1.
        frame[:, 0:sight, :] = 1.
        frame[:, -sight:, :] = 1.
        return frame

    def draw_video(self, output_path):
        writer = skvideo.io.FFmpegWriter("{}.mp4".format(output_path))
        font = cv2.FONT_HERSHEY_SIMPLEX
        white_color = (255, 255, 255)
        frame_size = constants.FRAME_SIZE
        for k, frame in enumerate(self.video):
            frame = (frame * 255).astype(np.uint8)
            frame = cv2.resize(frame, (frame_size, frame_size), interpolation=cv2.INTER_NEAREST)
            frame_with_text = np.zeros((frame.shape[0] + 100, frame.shape[1], 3), dtype=np.uint8)
            frame_with_text[100:, :, :] = frame
            cv2.putText(frame_with_text, "Score = {:.1f}".format(self.scores[k]),
                        (int(0.5 * frame.shape[1] - 100), 75), font, 1, white_color, 2)
            for _ in range(constants.FRAME_REPEAT):
                writer.writeFrame(frame_with_text)
        writer.close()

    def get_number_of_actions(self):
        return 4
