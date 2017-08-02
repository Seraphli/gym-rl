import unittest
from util.epsilon import *


class TestEps(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_multistage_eps(self):
        eps = MultiStageEpsilon([LinearAnnealEpsilon(1.0, 0.1, int(1e7)),
                                 LinearAnnealEpsilon(0.1, 0.01, int(5e7 - 1e7))])
        self.assertGreater(eps.get(int(1e6)), 0.1)
        self.assertEqual(eps.get(int(1e7)), 0.1)
        self.assertGreater(eps.get(int(2e7)), 0.01)
        self.assertEqual(eps.get(int(5e7)), 0.01)
        self.assertEqual(eps.get(int(6e7)), 0.01)


if __name__ == '__main__':
    unittest.main()
