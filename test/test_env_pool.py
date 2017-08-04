import unittest
from util.env_pool import *


class TestEnvPool(unittest.TestCase):
    def setUp(self):
        self.ep = EnvPool("Pong", 2)

    def tearDown(self):
        self.ep.close()

    def test_reset_state(self):
        obs = self.ep.reset()
        obs_, reward, done, info = self.ep.step([1, 2])
        self.ep.reset_state()
        obs = self.ep.reset()
        obs_, reward, done, info = self.ep.step([1, 2])


if __name__ == '__main__':
    unittest.main()
