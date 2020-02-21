###########
# Imports #
###########

import argparse
from Agents.RandomAgent import RandomAgent
from Agents.CNNAgent import CNNAgent
from Environment import Environment


#############
# Functions #
#############

def parse_args():
    parser = argparse.ArgumentParser(
        description="Arguments for running agents")
    parser.add_argument(
        "-a", "--agent", dest="agent",
        help="Agent: Random (default)", default="Random")
    parser.add_argument(
        "-e", "--epochs", dest="epochs",
        help="Number of epochs (default: 10)", default=10, type=int)
    parser.add_argument(
        "-o", "--output_path", dest="output_path",
        help="Path to save videos", required=True)
    parser.add_argument(
        "--train", action="store_true",
        help="Whether to train the agent or not")
    parser.add_argument(
        "--n_epochs", dest="n_epochs", default=20, type=int,
        help="Number of training epochs (default: 20)")
    parser.add_argument(
        "--batch_size", dest="batch_size", default=32, type=int,
        help="Training batch size (default: 32)")
    return parser.parse_args()


def get_agent(args, env):
    if args.agent == "Random":
        return RandomAgent(env)
    if args.agent == "CNN":
        return CNNAgent(env)
    return RandomAgent(env)


########
# Main #
########

if __name__ == "__main__":
    args = parse_args()
    env = Environment()
    agent = get_agent(args, env)
    if args.train:
        agent.train(n_epochs=args.n_epochs, batch_size=args.batch_size)
    else:
        agent.play(args.epochs, output_path=args.output_path)
