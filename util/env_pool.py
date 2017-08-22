import gym
from util.env_wrapper import wrap_dqn, wrap_gym, SimpleMonitor
import util.util as U
import multiprocessing as mp
import numpy as np

gym.undo_logger_setup()


class Env(object):
    def __init__(self, game_name, game_type, index=-1):
        """Build a gym environment.
         
        Args:
            game_name (str): game_name (str): Name of the game
            game_type (str): Type of the environment
        """
        self._game_name = game_name
        self._game_type = game_type
        self._index = index
        if game_type == 'gym':
            env = gym.make(game_name + '-v0')
            self.monitored_env = SimpleMonitor(env)
            self.env = wrap_gym(self.monitored_env)  # applies a bunch of modification
        else:
            env = gym.make(game_name + 'NoFrameskip-v4')
            self.monitored_env = SimpleMonitor(env)
            self.env = wrap_dqn(self.monitored_env)  # applies a bunch of modification

    @property
    def name(self):
        return self.env.spec.id

    @property
    def action_num(self):
        return self.env.action_space.n

    def reset(self):
        self.obs = np.array(self.env.reset())
        return (self.obs,)

    def reset_state(self):
        self.monitored_env.reset_state()

    def step(self, action):
        self.obs_, self.reward, self.done, self.info = self.env.step(action)
        return np.array(self.obs_), self.reward, self.done, self.info

    def auto_reset(self):
        if self.done:
            self.obs = np.array(self.env.reset())
            if self.info['rewards']:
                U.env_logger.debug('Environment {} finished with reward {}'.
                                   format(self._index, self.info['rewards'][-1]))
        else:
            self.obs = np.array(self.obs_)
        return self.obs, self.info

    def random_action(self):
        return self.env.action_space.sample()

    def close(self):
        self.env.close()


class EnvPool(object):
    def __init__(self, game_name, game_type, size):
        """Environment pool
        
        Args:
            game_name (str): Name of the game
            game_type (str): Type of the environment
            size (int): Size of environment pool 
        """
        self._game_name = game_name
        self._game_type = game_type
        self._size = size
        self.setup()

    def setup(self):
        self._envs = [Env(self._game_name, self._game_type, i) for i in range(self._size)]
        self._env_queue = [[mp.Queue(), mp.Queue()] for _ in range(self._size)]
        self._ps = [mp.Process(target=EnvPool._env_loop, args=(self._envs[_], self._env_queue[_]), daemon=True)
                    for _ in range(self._size)]
        for p in self._ps:
            p.start()

    @property
    def size(self):
        return self._size

    @property
    def name(self):
        return self._envs[0].name

    @property
    def action_num(self):
        return self._envs[0].action_num

    @staticmethod
    def _env_loop(env, queue):
        while True:
            cmd = queue[0].get()
            if cmd == "close":
                env.close()
                break
            if cmd == "reset":
                queue[1].put(env.reset())
            if cmd == "reset_state":
                env.reset_state()
            if cmd == "step":
                a = queue[0].get()
                queue[1].put(env.step(a))
            if cmd == "auto_reset":
                queue[1].put(env.auto_reset())
            if cmd == "random":
                a = env.random_action()
                queue[1].put((a,))
                queue[1].put(env.step(a))
        queue[0].close()
        queue[1].close()

    def _put(self, cmd, args=None):
        for i in range(self._size):
            self._env_queue[i][0].put_nowait(cmd)
            if not (args is None):
                self._env_queue[i][0].put_nowait(args[i])

    def _get(self, size):
        results = [[] for _ in range(size)]
        for i in range(self._size):
            result = self._env_queue[i][1].get()
            for _ in range(size):
                results[_].append(result[_])
        if size == 1:
            return results[0]
        return results

    def reset(self):
        self._put("reset")
        return self._get(1)

    def reset_state(self):
        self._put("reset_state")

    def step(self, actions):
        assert len(actions) == self._size
        self._put("step", actions)
        return self._get(4)

    def auto_reset(self):
        self._put("auto_reset")
        return self._get(2)

    def random(self):
        self._put("random")
        return self._get(1), self._get(4)

    def close(self):
        self._put("close")
        for p in self._ps:
            p.join()
        return
