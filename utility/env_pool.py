import gym, multiprocessing as mp

gym.undo_logger_setup()


class Env(object):
    def __init__(self, env_name, preprocessor=None, use_lives=True):
        self._env_name = env_name
        self._preprocessor = preprocessor
        self._use_lives = use_lives
        self._env = gym.make(self._env_name)
        self._no_op = 30
        self._last_action = 0
        self._no_op_count = 0
        self.reset()

    @property
    def name(self):
        return self._env.spec.id

    @property
    def action_num(self):
        return self._env.action_space.n

    def reset(self):
        self.state = "Reset"
        self.observation = self._env.reset()
        if self._preprocessor:
            self.observation = self._preprocessor.process(self.observation)
        self.frame = 0
        self.score = .0
        self._lives = -1

    def random_action(self):
        return self._env.action_space.sample()

    def step(self, action):
        if action == self._last_action:
            self._no_op_count += 1
            if self._no_op_count == self._no_op:
                while self.action == self._last_action:
                    self.action = self.random_action()
                self._no_op_count = 0
            else:
                self.action = action
        else:
            self.action = action
        self.observation_, self.reward, self.done, self.info = self._env.step(self.action)
        if self._use_lives:
            if self._lives == -1:
                self._lives = self.info["ale.lives"]
                self.t = False
            elif self._lives > self.info["ale.lives"] or self.done:
                self._lives = self.info["ale.lives"]
                self.t = True
            else:
                self.t = False
        else:
            self.t = self.done
        self.transition = (self.observation, self.action, self.reward, self.t)
        self.frame += 1
        self.score += self.reward
        result = self.done, self.transition, self.frame, self.score
        self.observation = self.observation_
        if self._preprocessor:
            self.observation = self._preprocessor.process(self.observation)
        if self.done:
            self.reset()
        return result


class EnvPool(object):
    def __init__(self, env_name, size, preprocessor=None, use_lives=True):
        self._env_name = env_name
        self._size = size
        self._preprocessor = preprocessor
        self._use_lives = use_lives
        self.setup()

    def setup(self):
        self._envs = [Env(self._env_name, self._preprocessor, self._use_lives) for _ in range(self._size)]
        self._env_queue = [[mp.Queue(), mp.Queue()] for _ in range(self._size)]
        self._ps = [mp.Process(target=EnvPool._env_loop, args=(self._envs[_], self._env_queue[_]), daemon=True)
                    for _ in range(self._size)]
        for p in self._ps:
            p.start()

    @staticmethod
    def _env_loop(env, queue):
        while True:
            cmd = queue[0].get()
            if cmd == "exit":
                break
            if cmd == "reset":
                env.reset()
            if cmd == "before":
                queue[1].put(env.observation)
            if cmd == "step":
                a = queue[0].get()
                queue[1].put(env.step(a))
            if cmd == "random":
                queue[1].put(env.step(env.random_action()))

    def close(self):
        for i in range(self._size):
            self._env_queue[i][0].put_nowait("exit")
        for p in self._ps:
            p.join()

    def reset(self):
        for i in range(self._size):
            self._env_queue[i][0].put_nowait("reset")

    def before_step(self):
        for i in range(self._size):
            self._env_queue[i][0].put_nowait("before")
        result = []
        for i in range(self._size):
            result.append(self._env_queue[i][1].get())
        return result

    def step(self, actions):
        assert len(actions) == self._size
        for i in range(self._size):
            self._env_queue[i][0].put_nowait("step")
            self._env_queue[i][0].put_nowait(actions[i])
        result = []
        for i in range(self._size):
            result.append(self._env_queue[i][1].get())
        return result

    def random(self):
        for i in range(self._size):
            self._env_queue[i][0].put_nowait("random")
        result = []
        for i in range(self._size):
            result.append(self._env_queue[i][1].get())
        return result
