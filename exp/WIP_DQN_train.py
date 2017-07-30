from algo.WIP_DQN import agent as agent
import gym, numpy as np, time
from util.env_wrapper import wrap_dqn, SimpleMonitor
from util.replay_buffer import ReplayBuffer
from util.epsilon import LinearAnnealEpsilon
from util.util import *
from util.WIP_env_pool import EnvPool


def make_env(game_name):
    env = gym.make(game_name + "NoFrameskip-v4")
    monitored_env = SimpleMonitor(env)
    env = wrap_dqn(monitored_env)  # applies a bunch of modification
    return env, monitored_env


def main():
    args = agent.parse_args()
    ep = EnvPool(args.env, 2)
    with agent.make_session():
        replay_buffer = ReplayBuffer(args.replay_buffer_size)
        agent.setup(ep.action_num, replay_buffer)
        eps = LinearAnnealEpsilon(args.eps[0], args.eps[1], int(args.eps[2]))
        obs = ep.reset()
        start_time, start_steps = None, None
        fps_estimate = RecentAvg()
        record = Record()
        num_iters = 0
        while num_iters < args.num_steps:
            num_iters += ep.size
            action = agent.take_action(obs, eps.get(num_iters))
            obs_, reward, done, info = ep.step(action)
            [replay_buffer.add(obs[_], action[_], reward[_], float(done[_]), obs_[_]) for _ in range(ep.size)]
            obs = ep.auto_reset()
            if num_iters % args.target_update_freq == 0:
                agent.update_target()

            if (num_iters > max(5 * args.batch_size, args.replay_buffer_size // 20) and
                            num_iters % args.learning_freq == 0):
                # Minimize the error in Bellman's equation and compute TD-error
                agent.train()

            if num_iters > args.num_steps:
                break

            if done[0] or done[1]:
                total_step = sum(info[_]['steps'] for _ in range(ep.size))
                total_epi = sum(len(info[_]['rewards']) for _ in range(ep.size))
                mean_reward = np.mean([np.mean(info[_]["rewards"][-100:]) for _ in range(ep.size)])
                if start_time is not None:
                    steps_per_iter = total_step - start_steps
                    iteration_time = time.time() - start_time
                    fps_estimate.update(steps_per_iter / iteration_time)
                start_time, start_steps = time.time(), total_step
                steps_left = args.num_steps - total_step
                completion = np.round(total_step / args.num_steps, 2)

                record.add_key_value("% Completion", completion)
                record.add_key_value("Steps", pretty_num(total_step))
                record.add_key_value("Iters", pretty_num(num_iters))
                record.add_key_value("Episodes", pretty_num(total_epi))
                record.add_key_value("Reward (100 epi mean)", np.round(mean_reward, 2))
                record.add_key_value("% Exploration", np.round(eps.get(num_iters) * 100, 2))
                record.add_line("ETA: " + (pretty_eta(int(steps_left / fps_estimate.value))
                                           if fps_estimate.value is not None else "calculating..."))

                main_logger.info("\n" + record.dumps("\t\t\t"))


if __name__ == '__main__':
    main()
