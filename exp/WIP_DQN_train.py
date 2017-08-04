from algo.WIP_DQN import agent as agent
import numpy as np, time
from util.replay_buffer import ReplayBuffer
from util.epsilon import LinearAnnealEpsilon, MultiStageEpsilon
from util.util import *
from util.env_pool import EnvPool
from tqdm import tqdm


class Game(object):
    def __init__(self):
        self.args = args = agent.parse_args()
        self.ep = EnvPool(args.env, self.args.env_size)
        self.eps = [MultiStageEpsilon([LinearAnnealEpsilon(1.0, 0.1, int(1e6)),
                                       LinearAnnealEpsilon(0.1, 0.05, int(1e7 - 1e6))]),
                    0]
        self.replay = ReplayBuffer(args.replay_buffer_size)
        main_logger.info("Replay Buffer Max Size: {}B".format(pretty_num(args.replay_buffer_size *
                                                                         (84 * 84 * 4 * 2 + 8), True)))
        self.sess = agent.make_session()
        self.sess.__enter__()
        agent.setup(self.ep.action_num, self.replay)
        self.train_epi = 0
        self.max_reward = agent.score

    def random(self):
        random_step = self.args.replay_buffer_size // 10
        obs = self.ep.reset()
        with tqdm(total=random_step, desc="random", ascii=True) as t:
            while t.n < random_step:
                action, (obs_, reward, done, info) = self.ep.random()
                [self.replay.add(obs[i], action[i], reward[i], float(done[i]), obs_[i]) for i in range(self.ep.size)]
                obs, info = self.ep.auto_reset()
                t.update(self.ep.size)
        mean_reward = np.mean(
            [np.mean(info[i]['rewards']) for i in range(self.ep.size) if info[i]['rewards']])
        record = Record()
        record.add_key_value('Phase', 'Random')
        record.add_key_value('Mean Reward', np.round(mean_reward, 2))
        main_logger.info("\n" + record.dumps())
        if not self.max_reward:
            self.max_reward = mean_reward

    def train(self):
        train_step = 250000
        self.ep.reset_state()
        obs = self.ep.reset()
        with tqdm(total=train_step, desc="Train", ascii=True) as t:
            while t.n < train_step:
                action = agent.take_action(obs, self.eps[0].get(self.train_epi * train_step + t.n))
                obs_, reward, done, info = self.ep.step(action)
                [self.replay.add(obs[i], action[i], reward[i], float(done[i]), obs_[i]) for i in range(self.ep.size)]
                obs, info = self.ep.auto_reset()
                if t.n % self.args.target_update_freq == 0:
                    agent.update_target()
                if t.n % self.args.learning_freq == 0:
                    agent.train(self.ep.size)
                t.update(self.ep.size)
        self.train_epi += 1
        completion = np.round(self.train_epi / self.args.num_iters, 2)
        total_epi = sum(len(info[i]['rewards']) for i in range(self.ep.size))
        mean_reward = np.mean(
            [np.mean(info[i]['rewards'][-100:]) for i in range(self.ep.size) if info[i]['rewards']])
        record = Record()
        record.add_key_value('Phase', 'Train')
        record.add_key_value('% Completion', completion)
        record.add_key_value('Episodes', pretty_num(total_epi))
        record.add_key_value('% Exploration', np.round(self.eps[0].get(self.train_epi * train_step) * 100, 2))
        record.add_key_value('Reward (100 epi mean)', np.round(mean_reward, 2))
        main_logger.info("\n" + record.dumps())

    def test(self):
        test_step = 200000
        self.ep.reset_state()
        obs = self.ep.reset()
        with tqdm(total=test_step, desc="Evaluation", ascii=True) as t:
            while t.n < test_step:
                action = agent.take_action(obs, self.eps[1])
                self.ep.step(action)
                obs, info = self.ep.auto_reset()
                t.update(self.ep.size)
        total_epi = sum(len(info[i]['rewards']) for i in range(self.ep.size))
        mean_reward = np.mean(
            [np.mean(info[i]['rewards']) for i in range(self.ep.size) if info[i]['rewards']])
        record = Record()
        record.add_key_value('Phase', 'Evaluation')
        record.add_key_value('Episodes', pretty_num(total_epi))
        record.add_key_value('Mean Reward', np.round(mean_reward, 2))
        main_logger.info("\n" + record.dumps())
        if self.max_reward < mean_reward:
            self.max_reward = mean_reward
            agent.score = mean_reward
            agent.save_model()

    def run(self):
        self.random()
        for i in range(self.args.num_iters):
            self.train()
            self.test()
        self.exit()

    def exit(self):
        self.ep.close()


if __name__ == '__main__':
    Game().run()
