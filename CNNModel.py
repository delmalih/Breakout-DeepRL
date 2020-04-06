##########
# Import #
##########


import torch
import torch.nn as nn
import torch.nn.functional as F


#############
# DQN Model #
#############


class CNNModel(nn.Module):
    def __init__(self, n_actions: int):
        super(CNNModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1),     # 16 x 16 x 16
            nn.ReLU(),
            nn.MaxPool2d(2),                    # 8 x 8 x 16
            nn.Conv2d(16, 32, 3, padding=1),    # 8 x 8 x 32
            nn.ReLU(),
            nn.MaxPool2d(2),                    # 4 x 4 x 32
            nn.Conv2d(32, 64, 3, padding=1),    # 4 x 4 x 64
            nn.ReLU(),
            nn.MaxPool2d(2),                    # 2 x 2 x 64
            nn.Conv2d(64, 128, 3, padding=1),   # 2 x 2 x 128
            nn.ReLU(),
            nn.MaxPool2d(2),                    # 1 x 1 x 128
        )
        self.q_values_head = nn.Linear(128, n_actions)

    def forward(self, x: torch.Tensor):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        q_values = self.q_values_head(x)
        return q_values
