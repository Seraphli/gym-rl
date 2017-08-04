import unittest
from util.env_wrapper import *


class TestEnvPool(unittest.TestCase):
    def setUp(self):
        env = gym.make("PongNoFrameskip-v4")
        self.monitored_env = SimpleMonitor(env)
        self.env = wrap_dqn(self.monitored_env)  # applies a bunch of modification

    def tearDown(self):
        self.env.close()

    def test_reset_state(self):
        self.env.reset()
        self.env.step(1)
        self.monitored_env.reset_state()
        self.env.reset()
        self.env.step(1)


if __name__ == '__main__':
    unittest.main()
