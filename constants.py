# Imports
import numpy as np
import torch


# GLOBAL CONSTANTS
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
CPU_DEVICE = torch.device('cpu')
OUTPUT_FOLDER = './tmp'


# GAME CONSTANTS
SIZE = 8
INIT_LENGTH = 3
ACTIONS_TO_HEADINGS = np.array([
    [+1, 0],
    [-1, 0],
    [0, +1],
    [0, -1],
])


# RANDOM AGENT
N_RANDOM_PLAYS = 100


# CNN AGENT
N_TRAIN_EPOCHS = 10000
MODEL_PATH = "./tmp/model.pth"
MEMORY_CAPACITY = 10000
DISCOUNT = .9
EPS_START = .99
EPS_MIN = .05
EPS_DECAY = .99
BATCH_SIZE = 256
TARGET_UPDATE = 10
SAVE_FREQ = 10
PRINT_FREQ = 1
