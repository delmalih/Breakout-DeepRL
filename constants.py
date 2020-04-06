# Imports
import torch


# GLOBAL CONSTANTS
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
CPU_DEVICE = torch.device('cpu')
OUTPUT_FOLDER = './tmp'


# GAME CONSTANTS
SIZE = 16
NUM_ENVS = 400
N_CHANNELS = 2
FOOD_CHANNEL = 0
HEAD_CHANNEL = 1


# RANDOM AGENT
N_RANDOM_PLAYS = 100


# CNN AGENT
N_EPOCHS = 100000
SAVE_FREQ = 100
BATCH_SIZE = 512
MODEL_PATH = "./tmp/model.pth"
MAX_MEMORY = 100000
DISCOUNT = .9
EPS_START = .99
EPS_MIN = .1
EPS_DECAY = .9995
LEARNING_RATE = 1e-3
CONV_FILTERS = torch.tensor([
    [
        [0, 1, 0],
        [0, 0, 0],
        [0, 0, 0],
    ],
    [
        [0, 0, 0],
        [0, 0, 1],
        [0, 0, 0],
    ],
    [
        [0, 0, 0],
        [0, 0, 0],
        [0, 1, 0],
    ],
    [
        [0, 0, 0],
        [1, 0, 0],
        [0, 0, 0],
    ],
]).unsqueeze(1).float()


# DISPLAY
FRAME_SIZE = 500
FRAME_REPEAT = 3