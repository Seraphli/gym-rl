import gym
from util.env_wrapper import wrap_dqn, SimpleMonitor
import multiprocessing as mp

gym.undo_logger_setup()


class Env(object):
    def __init__(self, game_name, auto_reset=True):
        self._game_name = game_name
        self._auto_reset = auto_reset
        env = gym.make(game_name + "NoFrameskip-v4")
        monitored_env = SimpleMonitor(env)
        self.env = wrap_dqn(monitored_env)  # applies a bunch of modification

    @property
    def name(self):
        return self.env.spec.id

    @property
    def action_num(self):
        return self.env.action_space.n

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def random_action(self):
        return self.env.action_space.sample()


class EnvPool(object):
    def __init__(self, game_name, size):
        self._game_name = game_name
        self._size = size
        self.setup()

    def setup(self):
        self._envs = [Env(self._game_name) for _ in range(self._size)]
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
            if cmd == "exit":
                break
            if cmd == "reset":
                queue[1].put(env.reset())
            if cmd == "step":
                a = queue[0].get()
                queue[1].put(env.step(a))
            if cmd == "random":
                queue[1].put(env.step(env.random_action()))

    def _put(self, cmd):
        for i in range(self._size):
            self._env_queue[i][0].put_nowait(cmd)

    def _get(self):
        result = []
        for i in range(self._size):
            result.append(self._env_queue[i][1].get())
        return result

    def reset(self):
        self._put("reset")
        return self._get()

    def step(self, actions):
        assert len(actions) == self._size
        for i in range(self._size):
            self._put("step")
            self._put(actions[i])
        return self._get()

    def random(self):
        self._put("random")
        return self._get()

    def exit(self):
        self._put("random")
