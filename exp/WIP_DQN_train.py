from algo.WIP_DQN import agent as agent
import numpy as np, time
from util.replay_buffer import ReplayBuffer
from util.epsilon import LinearAnnealEpsilon, MultiStageEpsilon
from util.util import *
from util.WIP_env_pool import EnvPool


def main():
    args = agent.parse_args()
    ep = EnvPool(args.env, args.env_size)
    with agent.make_session():
        replay_buffer = ReplayBuffer(args.replay_buffer_size)
        agent.setup(ep.action_num, replay_buffer)
        main_logger.info("Replay Buffer Max Size: {}B".format(pretty_num(args.replay_buffer_size * 56456, True)))
        eps = MultiStageEpsilon([LinearAnnealEpsilon(1.0, 0.1, int(1e6)),
                                 LinearAnnealEpsilon(0.1, 0.01, int(1e7 - 1e6))])
        obs = ep.reset()
        start_time, start_steps, total_step = None, None, None
        fps_estimate = RecentAvg(10)
        record = Record()
        num_iters = 0
        while num_iters < args.num_steps:
            num_iters += 1
            action = agent.take_action(obs, eps.get(total_step if total_step else 0))
            obs_, reward, done, info = ep.step(action)
            [replay_buffer.add(obs[i], action[i], reward[i], float(done[i]), obs_[i]) for i in range(ep.size)]
            obs, info = ep.auto_reset()
            total_step = sum(info[i]['steps'] for i in range(ep.size))
            if num_iters % args.target_update_freq == 0:
                agent.update_target()

            if (num_iters > max(5 * args.batch_size, args.replay_buffer_size // 20) and
                            num_iters % args.learning_freq == 0):
                # Minimize the error in Bellman's equation and compute TD-error
                agent.train(ep.size)

            if done[0]:
                total_epi = sum(len(info[i]['rewards']) for i in range(ep.size))
                mean_reward = np.mean(
                    [np.mean(info[i]["rewards"][-100:]) for i in range(ep.size) if info[i]["rewards"]])
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
                record.add_key_value("% Exploration", np.round(eps.get(total_step if total_step else 0) * 100, 2))
                record.add_line("ETA: " + (pretty_eta(int(steps_left / fps_estimate.value))
                                           if fps_estimate.value is not None else "calculating..."))

                main_logger.info("\n" + record.dumps("\t\t\t"))

            if total_step > args.num_steps:
                break


if __name__ == '__main__':
    main()
