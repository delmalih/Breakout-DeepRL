###########
# Imports #
###########

import argparse
from Agents.RandomAgent import RandomAgent
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
    return parser.parse_args()


def get_agent(args, env):
    if args.agent == "Random":
        return RandomAgent(env)
    return RandomAgent(env)


def test_agent(agent, env, epochs, output_path=''):
    total_score = 0
    for e in range(epochs):
        score = 0
        state = env.reset()
        done = False
        while not done:
            action = agent.act(state)
            state, reward, done, _ = env.step(action)
            score += reward
        env.draw_video(output_path + str(e))
        total_score += score
        print("Epoch = {:4d} | Current score = {:.2f}".format(e, score))
    print("Average score: {}".format(1. * total_score / epochs))


########
# Main #
########

if __name__ == "__main__":
    args = parse_args()
    env = Environment()
    agent = get_agent(args, env)
    test_agent(agent, env, args.epochs, output_path=args.output_path)
