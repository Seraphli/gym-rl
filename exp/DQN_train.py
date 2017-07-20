from tqdm import tqdm
from agent.DQN import DQN
from utility.utility import init_logger, load_config
from utility.env_pool import *
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


class Game(object):
    def __init__(self):
        self.name = 'DQN'
        self.usage = """
        Usage: DQN_train.py [DQN.yml]
        """
        self._agent_algo = DQN

    def setup(self, cfg='DQN.yml'):
        self._cfg = cfg
        self._config = load_config(self._cfg)
        self.logger = init_logger(self._config["Train"]["LoggerName"])
        self._env_name = self._config["Train"]["Game"]
        self.sess = tf.Session()
        env = Env(self._env_name)
        self._agent = self._agent_algo(self.sess, {"action_n": env.action_num, "name": env.name},
                                       self._config["Agent"], self.logger)
        del env
        init = tf.global_variables_initializer()
        self.sess.graph.finalize()

        if not self._agent.load_session():
            self.sess.run(init)
        self._use_lives = True
        self._env_size = 16
        self._ep = EnvPool(self._env_name, self._env_size, self._agent.preprocessor)

    def clear_stat(self):
        self.n_games = 0
        self.min_score = .0
        self.max_score = .0
        self.total_score = .0
        self.total_step = 0
        self.complete = False

    def update(self):
        for r in self._result:
            done, transition, frame, score = r
            self._agent.store_transition(transition)
            self.total_step += 1
            self._tqdm.update()
            if self.total_step == self._step:
                break
            if done:
                self.min_score = min(self.min_score, score)
                self.max_score = max(self.max_score, score)
                self.total_score += score
                self.n_games += 1

    def game(self, phase, step):
        self.clear_stat()
        self._step = step
        with tqdm(total=step, desc=phase) as self._tqdm:
            if phase == "Random":
                self._ep.reset()
                while self.total_step < step:
                    self._result = self._ep.random()
                    self.update()
            if phase == "Train":
                self._ep.reset()
                while self.total_step < step:
                    result = self._ep.before_step()
                    actions = [self._agent.train(r) for r in result]
                    self._result = self._ep.step(actions)
                    self.update()
            if phase == "Test":
                self._ep.reset()
                while self.total_step < step:
                    result = self._ep.before_step()
                    actions = [self._agent.eval(r) for r in result]
                    self._result = self._ep.step(actions)
                    self.update()
        self.average_score = self.total_score / self.n_games

        return self.n_games, self.total_step, self.average_score, self.min_score, self.max_score

    def run(self):
        self.logger.info('Session start')
        n_games, total_step, average_score, min_score, max_score \
            = self.game("Random", self._config["Agent"]["Algorithm"]["ReplayStartSize"])
        self.logger.info('PHASE: %s, N: %d, STEP: %d, AVG: %f, MIN: %d, MAX: %d' %
                         ("Random", n_games, total_step, average_score, min_score, max_score))
        self._agent.prepare_train()
        for _ in range(self._config["Train"]["Episode"]):
            n_games, total_step, average_score, min_score, max_score \
                = self.game("Train", self._config["Train"]["TrainStep"])
            self.logger.info('PHASE: %s, N: %d, STEP: %d, AVG: %f, MIN: %d, MAX: %d' %
                             ("Train", n_games, total_step, average_score, min_score, max_score))
            self._agent.add_summary("Train", (_, average_score, min_score, max_score))
            n_games, total_step, average_score, min_score, max_score \
                = self.game("Test", self._config["Train"]["TestStep"])
            self.logger.info('PHASE: %s, N: %d, STEP: %d, AVG: %f, MIN: %d, MAX: %d' %
                             ("Test", n_games, total_step, average_score, min_score, max_score))
            self._agent.add_summary("Test", (_, average_score, min_score, max_score))

    def run_safe(self):
        try:
            self.run()
        except KeyboardInterrupt:
            self.logger.info('Session stop')


if __name__ == '__main__':
    import sys

    g = Game()
    if len(sys.argv) > 2:
        print(g.usage)
    elif len(sys.argv) == 2:
        g.setup(sys.argv[1])
        g.run_safe()
    else:
        g.setup()
        g.run_safe()
