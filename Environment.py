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
    def __init__(self,
                 num_envs: int = constants.NUM_ENVS,
                 size: int = constants.SIZE):
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
        self.directions = torch.zeros((self.num_envs, 2, 1)).to(self.device)

        # Update envs
        for k in range(self.num_envs):
            self._generate_snake(k)
            self._generate_food(k)

    def _generate_snake(self, env_id):
        # Init. variable
        self.envs[env_id, constants.HEAD_CHANNEL] = torch.zeros((self.size, self.size)).to(self.device)
        self.directions[env_id] = torch.zeros((2, 1)).to(self.device)

        # Generate head of snake
        head_coords = torch.randint(1, self.size - 2, size=(2,)).to(self.device)
        self.envs[env_id, constants.HEAD_CHANNEL, head_coords[0], head_coords[1]] = 1

        # Generate direction
        self.directions[env_id, 0, 0] = -1

    def _generate_food(self, env_id):
        self.envs[env_id, constants.FOOD_CHANNEL] = torch.zeros((self.size, self.size)).to(self.device)
        head = self.envs[env_id, constants.HEAD_CHANNEL, :, :]
        possible_coords = (head == 0).nonzero()
        chosen_id = torch.randint(possible_coords.size(0), size=(1,))[0]
        chosen_coord = possible_coords[chosen_id]
        self.envs[env_id, constants.FOOD_CHANNEL, chosen_coord[0], chosen_coord[1]] = 1

    def _reset_video(self):
        self.video = []
        self.scores = []

    def step(self, actions: torch.Tensor):
        old_distances = self._compute_distance_to_food()
        self._update_direction(actions)
        self._update_head()
        done = self._check_collision().byte().float()
        eaten = self._check_food().byte().float()
        new_distances = self._compute_distance_to_food()
        delta_distances = new_distances - old_distances
        reward = self._compute_reward(delta_distances, done, eaten)
        self.video.append(self.render())
        self.scores.append(reward.sum())
        return self.envs, reward, done, {}

    def _update_direction(self, action: torch.Tensor):
        rotation_angle = (action.to(self.device).float() - 1.) * math.pi / 2.
        rotation_matrix = torch.zeros((self.num_envs, 2, 2)).to(self.device)
        rotation_matrix[:, 0, 0] = torch.cos(rotation_angle).to(self.device)
        rotation_matrix[:, 0, 1] = -torch.sin(rotation_angle).to(self.device)
        rotation_matrix[:, 1, 0] = torch.sin(rotation_angle).to(self.device)
        rotation_matrix[:, 1, 1] = torch.cos(rotation_angle).to(self.device)
        self.directions = torch.matmul(rotation_matrix, self.directions).long().float()

    def _update_head(self):
        conv_kernels = constants.CONV_FILTERS.to(self.device)
        directions_idx = torch.zeros((self.num_envs,), dtype=torch.long).to(self.device)
        directions_idx[(self.directions[:, 0, 0] == 0) & (self.directions[:, 1, 0] == -1)] = 0
        directions_idx[(self.directions[:, 0, 0] == +1) & (self.directions[:, 1, 0] == 0)] = 1
        directions_idx[(self.directions[:, 0, 0] == 0) & (self.directions[:, 1, 0] == +1)] = 2
        directions_idx[(self.directions[:, 0, 0] == -1) & (self.directions[:, 1, 0] == 0)] = 3
        directions_onehot = torch.zeros((self.num_envs, 4)).to(self.device)
        directions_onehot.scatter_(1, directions_idx.unsqueeze(-1), 1)
        head_envs = self.envs[:, constants.HEAD_CHANNEL:constants.HEAD_CHANNEL+1, :, :]
        head_envs = F.conv2d(head_envs, conv_kernels, padding=1)
        head_envs = torch.einsum('bchw,bc->bhw', [head_envs, directions_onehot])
        self.envs[:, constants.HEAD_CHANNEL, :, :] = head_envs

    def _check_collision(self):
        head_envs = self.envs[:, constants.HEAD_CHANNEL, :, :]
        collision = head_envs.view(self.num_envs, -1).sum(dim=-1) == 0
        for env_id in collision.nonzero().view(-1):
            self._generate_snake(env_id)
            self._generate_food(env_id)
        return collision
    
    def _check_food(self):
        head_envs = self.envs[:, constants.HEAD_CHANNEL, :, :]
        food_envs = self.envs[:, constants.FOOD_CHANNEL, :, :]
        eaten = (head_envs * food_envs).view(self.num_envs, -1).sum(-1) > 0
        for env_id in eaten.nonzero().view(-1):
            self._generate_food(env_id)
        return eaten
    
    def _compute_distance_to_food(self):
        head_coords = self.envs[:, constants.HEAD_CHANNEL, :, :].round().nonzero()
        food_coords = self.envs[:, constants.FOOD_CHANNEL, :, :].round().nonzero()
        print(head_coords.shape, food_coords.shape)
        distance = (head_coords - food_coords).abs().sum(-1)
        return distance
    
    def _compute_reward(self, delta_distances, done, eaten):
        reward = torch.zeros((self.num_envs,)).to(self.device)
        reward[delta_distances < 0] = +0.1
        reward[delta_distances > 0] = -0.1
        reward[done == 1] = -1
        reward[eaten == 1] = +1
        return reward

    def sample_action(self):
        return torch.randint(3, size=(self.num_envs,))

    def render(self):
        frames = np.zeros((self.num_envs, self.size + 2, self.size + 2, 3))
        frames[:, 1:-1, 1:-1, :self.n_channels] = self.envs.permute(0, 2, 3, 1).cpu().detach().numpy()
        frames[:, 0, :, :] = 1.
        frames[:, :, 0, :] = 1.
        frames[:, -1, :, :] = 1.
        frames[:, :, -1, :] = 1.
        frames[frames > 0] = 1.
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
        return 3
