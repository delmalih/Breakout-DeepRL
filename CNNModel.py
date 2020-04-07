##########
# Import #
##########


import torch
import torch.nn as nn
import torch.nn.functional as F

import constants


#############
# DQN Model #
#############


class CNNModel(nn.Module):
    def __init__(self, n_actions):
        super(CNNModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(constants.N_CHANNELS, 16, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
        )
        self.q_values_head = nn.Linear(32, n_actions)

    def forward(self, x: torch.Tensor):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        q_values = self.q_values_head(x)
        return q_values
