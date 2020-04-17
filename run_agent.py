###########
# Imports #
###########


import os
import argparse

from Environment import Environment
from RandomAgent import RandomAgent
from CNNAgent import CNNAgent
import constants


#############
# Functions #
#############


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for running agents")
    parser.add_argument("-a", "--agent", dest="agent", help="Agent: Random (default)", default="Random")
    parser.add_argument("--train", action="store_true", help="Whether to train the agent or not")
    return parser.parse_args()


def get_agent(args, env):
    if args.agent == "Random":
        return RandomAgent(env)
    if args.agent == "CNN":
        return CNNAgent(env, constants.MODEL_PATH, is_training=args.train)
    return RandomAgent(env)


########
# Main #
########

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    args = parse_args()
    env = Environment()
    agent = get_agent(args, env)
    if args.train:
        agent.train()
    else:
        agent.play(constants.N_RANDOM_PLAYS)
