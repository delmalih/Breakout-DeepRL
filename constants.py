# Imports
import torch


# GLOBAL CONSTANTS
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
CPU_DEVICE = torch.device('cpu')
OUTPUT_FOLDER = './tmp'


# GAME CONSTANTS
SIZE = 16
N_CHANNELS = 3
HEAD_CHANNEL = 0
BONUS_CHANNEL = 1
MALUS_CHANNEL = 2
BONUS_REWARD = +0.5
MALUS_REWARD = -1.0
FILL_PERC = .5
SIGHT = 2
MAX_TIME = 100
ACTIONS_TO_VECT = torch.tensor([
    [-1, 0],
    [+1, 0],
    [0, -1],
    [0, +1],
]).to(DEVICE)


# RANDOM AGENT
N_RANDOM_PLAYS = 10


# CNN AGENT
N_EPOCHS = 10000
SAVE_FREQ = 1
BATCH_SIZE = 512
MODEL_PATH = "./tmp/model.pth"
MAX_MEMORY = 100000
DISCOUNT = .99
EPS_START = .99
EPS_MIN = .05
EPS_DECAY = .99
LEARNING_RATE = 1e-1


# DISPLAY
FRAME_SIZE = 500
FRAME_REPEAT = 3