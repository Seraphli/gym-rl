import tensorflow as tf
import argparse
from utility.tf_layer import tf_layer


class DQN(object):
    def __init__(self):
        self.algo = 'DQN'

    def setup(self):
        pass

    def parse_args(self):
        parser = argparse.ArgumentParser("DQN experiments for Atari games")
        parser.add_argument("--env", type=str, default="Pong", help="name of the game")

        parser.add_argument("--replay-buffer-size", type=int, default=int(1e6), help="replay buffer size")
        parser.add_argument("--lr", type=float, default=1e-4, help="learning rate for Adam optimizer")
        parser.add_argument("--num-steps", type=int, default=int(2e8),
                            help="total number of steps to run the environment for")
        parser.add_argument("--batch-size", type=int, default=32,
                            help="number of transitions to optimize at the same time")
        parser.add_argument("--learning-freq", type=int, default=4,
                            help="number of iterations between every optimization step")
        parser.add_argument("--target-update-freq", type=int, default=40000,
                            help="number of iterations between every target network update")
        parser.add_argument("--save-dir", type=str, default=None,
                            help="directory in which training state and model should be saved.")

        parser.add_argument('--eps', type=float, nargs=3, metavar=('INITIAL', 'FINAL', 'TOTAL'),
                            default=[1.0, 0.1, 1e6],
                            help="define epsilon, changing from initial value to final value in the total step")

        self.args = parser.parse_args()
        return self.args

    def make_session(self):
        self.sess = tf.Session()
        return self.sess

    def _def_network(self):
        self.arch = [
            {'layer': 'conv2d', 'kernel_size': [8, 8], 'input': 4, 'output': 32, 'stride': [4, 4]},
            {'layer': 'relu'},
            {'layer': 'conv2d', 'kernel_size': [4, 4], 'input': 32, 'output': 64, 'stride': [2, 2]},
            {'layer': 'relu'},
            {'layer': 'conv2d', 'kernel_size': [3, 3], 'input': 64, 'output': 64, 'stride': [1, 1]},
            {'layer': 'relu'},
            {'layer': 'flatten'},
            {'layer': 'fc', 'size': 512},
            {'layer': 'relu'},
            {'layer': 'fc', 'size': 10}
        ]

    def _build_network(self, reuse=False, initializer=None, trainable=True, collections=None):
        if not initializer:
            initializer = [tf.contrib.layers.variance_scaling_initializer(), tf.constant_initializer()]

        if collections:
            collections = [collections, tf.GraphKeys.GLOBAL_VARIABLES]
        else:
            collections = [tf.GraphKeys.GLOBAL_VARIABLES]

        ws = []
        ys = []
        for idx, args in enumerate(self.arch):
            args = args.copy().update(locals())
            w, y = tf_layer[args['layer']](args)
            ws += w
            ys.append(y)


agent = DQN()
