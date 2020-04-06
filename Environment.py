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
    def __init__(self, num_envs=constants.NUM_ENVS, size=constants.SIZE):
        self.num_envs = num_envs
        self.size = size
        self.device = constants.DEVICE
        self.n_channels = constants.N_CHANNELS
        self.reset()

    def reset(self):
        self._reset_video()
        self._create_envs()
        return self.envs

    def _create_envs(self):
        # Init. envs
        self.envs = torch.zeros((self.num_envs, self.n_channels, self.size, self.size)).to(self.device)
        
        # Init. head
        head_coords = torch.cat((
            torch.arange(0, self.num_envs).unsqueeze(1).to(self.device),
            torch.randint(1, self.size - 2, size=(self.num_envs, 2)).to(self.device),
        ), dim=-1).to(self.device)
        self.envs[head_coords[:, 0], constants.HEAD_CHANNEL, head_coords[:, 1], head_coords[:, 2]] = 1

        # Init. bonus malus
        nb_bonus = nb_malus = int(constants.SIZE * constants.SIZE * constants.FILL_PERC / 2.)
        for env_id in range(self.num_envs):
            self._generate_bonus(env_id, nb_bonus)
            self._generate_malus(env_id, nb_malus)

    def _generate_bonus(self, env_id, nb):
        chosen_coords = self._sample_possible_coords(env_id, nb)
        self.envs[env_id, constants.BONUS_CHANNEL, chosen_coords[:, 0], chosen_coords[:, 1]] = 1
    
    def _generate_malus(self, env_id, nb):
        chosen_coords = self._sample_possible_coords(env_id, nb)
        self.envs[env_id, constants.MALUS_CHANNEL, chosen_coords[:, 0], chosen_coords[:, 1]] = 1
    
    def _sample_possible_coords(self, env_id, nb):
        already_taken = self.envs[env_id, :, :, :].sum(0)
        possible_coords = (already_taken == 0).nonzero().to(self.device)
        chosen_idx = torch.randint(possible_coords.size(0), size=(nb, 1)).to(self.device)[:, 0]
        chosen_coords = possible_coords[chosen_idx]
        return chosen_coords

    def _reset_video(self):
        self.video = []
        self.scores = []

    def step(self, actions):
        self._update_head(actions)
        reward = self._compute_reward_and_update()
        done = torch.zeros((self.num_envs,)).to(self.device)
        self.video.append(self.render())
        self.scores.append(reward.sum() if len(self.scores) == 0 else self.scores[-1] + reward.sum())
        return self.envs, reward, done, {}
    
    def _update_head(self, action):
        conv_kernels = constants.CONV_FILTERS.to(self.device)
        action_onehot = torch.zeros((self.num_envs, self.get_number_of_actions())).to(self.device)
        action_onehot.scatter_(1, action.to(self.device).unsqueeze(-1), 1)
        head_envs = self.envs[:, constants.HEAD_CHANNEL:constants.HEAD_CHANNEL+1, :, :].clone()
        head_envs = F.conv2d(head_envs, conv_kernels, padding=1)
        head_envs = torch.einsum('bchw,bc->bhw', [head_envs, action_onehot]).long().float()
        to_update = head_envs.sum(-1).sum(-1) > 0
        self.envs[to_update, constants.HEAD_CHANNEL, :, :] = head_envs[to_update]
    
    def _compute_reward_and_update(self):
        head_envs = self.envs[:, constants.HEAD_CHANNEL, :, :]
        bonus_envs = self.envs[:, constants.BONUS_CHANNEL, :, :]
        malus_envs = self.envs[:, constants.MALUS_CHANNEL, :, :]
        eaten_bonus = (head_envs * bonus_envs).nonzero()
        eaten_malus = (head_envs * malus_envs).nonzero()
        self.envs[eaten_bonus[:, 0], constants.BONUS_CHANNEL, eaten_bonus[:, 1], eaten_bonus[:, 2]] = 0.
        self.envs[eaten_malus[:, 0], constants.MALUS_CHANNEL, eaten_malus[:, 1], eaten_malus[:, 2]] = 0.
        for env_id in eaten_bonus[:, 0]: self._generate_bonus(env_id, 1)
        for env_id in eaten_malus[:, 0]: self._generate_malus(env_id, 1)
        reward = torch.zeros((self.num_envs,)).to(self.device)
        reward[eaten_bonus[:, 0]] = constants.BONUS_REWARD
        reward[eaten_malus[:, 0]] = constants.MALUS_REWARD
        return reward

    def sample_action(self):
        return torch.randint(self.get_number_of_actions(), size=(self.num_envs,))

    def render(self):
        frames = np.ones((self.num_envs, self.size + 2, self.size + 2, 3))
        frames[:, 1:-1, 1:-1, 0] = self.envs[:, constants.MALUS_CHANNEL, :, :].cpu().detach().numpy()
        frames[:, 1:-1, 1:-1, 1] = self.envs[:, constants.BONUS_CHANNEL, :, :].cpu().detach().numpy()
        frames[:, 1:-1, 1:-1, 2] = self.envs[:, constants.HEAD_CHANNEL, :, :].cpu().detach().numpy()
        final_frame_w = int(np.ceil(np.sqrt(self.num_envs)))
        final_frame_h = int(np.ceil(self.num_envs / final_frame_w))
        final_frame = np.zeros((final_frame_h * (self.size + 2), final_frame_w * (self.size + 2), 3))
        counter = 0
        for i in range(final_frame_h):
            for j in range(final_frame_w):
                if counter < self.num_envs:
                    final_frame[i * (self.size + 2) : (i+1) * (self.size + 2), j * (self.size + 2) : (j+1) * (self.size + 2)] = frames[counter]
                counter += 1
        return final_frame

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
