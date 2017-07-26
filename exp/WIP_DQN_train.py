from algo.WIP_DQN import agent as agent
import gym, numpy as np
from utility.env_wrapper import wrap_dqn, SimpleMonitor
from utility.replay_buffer import ReplayBuffer
from utility.epsilon import LinearAnnealEpsilon
from utility.utility import main_logger


def make_env(game_name):
    env = gym.make(game_name + "NoFrameskip-v4")
    monitored_env = SimpleMonitor(env)
    env = wrap_dqn(monitored_env)  # applies a bunch of modification
    return env, monitored_env


def main():
    args = agent.parse_args()
    env, monitored_env = make_env(args.env)
    with agent.make_session():
        agent.setup(env.action_space.n)
        replay_buffer = ReplayBuffer(args.replay_buffer_size)
        eps = LinearAnnealEpsilon(1.0, 0.1, int(1e6))
        obs = env.reset()
        num_iters = 0
        while num_iters < args.num_steps:
            num_iters += 1
            action = agent.take_action(np.array(obs)[None], eps.get(num_iters))
            obs_, reward, done, info = env.step(action)
            replay_buffer.add(obs, action, reward, obs_, float(done))
            obs = obs_
            if done:
                obs = env.reset()
            if num_iters % args.target_update_freq == 0:
                agent.update_target()
            if done:
                main_logger.info("{}".format(reward))


if __name__ == '__main__':
    main()
