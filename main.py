import argparse

parser = argparse.ArgumentParser(description='DQN for CartPole')

parser.add_argument('--mode', default=0, type=int, metavar='N',
        help='the mode of the run. 0: dqn_low, 1: dqn_high')

args = parser.parse_args()

