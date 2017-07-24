from algo.WIP_DQN import agent as agent
import argparse, gym
from utility.env_wrapper import wrap_dqn, SimpleMonitor


def make_env(game_name):
    env = gym.make(game_name + "NoFrameskip-v4")
    monitored_env = SimpleMonitor(env)
    env = wrap_dqn(monitored_env)  # applies a bunch of modification
    return env, monitored_env


if __name__ == '__main__':
    args = agent.parse_args()
    env, monitored_env = make_env(args.env)
