from algo.WIP_DQN import agent as agent
import gym, numpy as np, time
from util.env_wrapper import wrap_dqn, SimpleMonitor
from util.replay_buffer import ReplayBuffer
from util.epsilon import LinearAnnealEpsilon
from util.util import *


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
        start_time, start_steps = None, None
        fps_estimate = RecentAvg()
        record = Record()
        num_iters = 0
        while num_iters < args.num_steps:
            num_iters += 1
            action = agent.take_action(np.array(obs)[None], eps.get(num_iters))[0]
            obs_, reward, done, info = env.step(action)
            replay_buffer.add(obs, action, reward, obs_, float(done))
            obs = obs_
            if done:
                obs = env.reset()
            if num_iters % args.target_update_freq == 0:
                agent.update_target()

            if (num_iters > max(5 * args.batch_size, args.replay_buffer_size // 20) and
                            num_iters % args.learning_freq == 0):
                # Sample a bunch of transitions from replay buffer
                s, a, r, s_, t = replay_buffer.sample(args.batch_size)
                # Minimize the error in Bellman's equation and compute TD-error
                agent.train(s, a, r, t, s_)

            if info["steps"] > args.num_steps:
                break

            if done:
                if start_time is not None:
                    steps_per_iter = info['steps'] - start_steps
                    iteration_time = time.time() - start_time
                    fps_estimate.update(steps_per_iter / iteration_time)
                start_time, start_steps = time.time(), info["steps"]
                steps_left = args.num_steps - info["steps"]
                completion = np.round(info["steps"] / args.num_steps, 2)

                record.add_key_value("% Completion", completion)
                record.add_key_value("Steps", pretty_num(info["steps"]))
                record.add_key_value("Iters", pretty_num(num_iters))
                record.add_key_value("Episodes", pretty_num(len(info["rewards"])))
                record.add_key_value("Reward (100 epi mean)", np.round(np.mean(info["rewards"][-100:]), 2))
                record.add_key_value("% Exploration", np.round(eps.get(num_iters) * 100, 2))
                record.add_line("ETA: " + (pretty_eta(int(steps_left / fps_estimate.value))
                                           if fps_estimate.value is not None else "calculating..."))

                main_logger.info("\n" + record.dumps("\t\t\t"))


if __name__ == '__main__':
    main()
