import tensorflow as tf
import argparse
from utility.tf_layer import tf_layer
from utility.utility import main_logger
from utility.tf_common import huber_loss, minimize_and_clip


class DQN(object):
    def __init__(self):
        self.algo = 'DQN'

    def setup(self):
        self.action_n = 4
        self.gamma = 0.99
        self._def_network()
        self._def_model()

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
            {'layer': 'fc', 'size': self.action_n}
        ]

    def _build_network(self, x, reuse=False, initializer=None, collections=None):
        if not initializer:
            initializer = [tf.contrib.layers.variance_scaling_initializer(), tf.constant_initializer()]

        if collections:
            collections = [collections, tf.GraphKeys.GLOBAL_VARIABLES]
        else:
            collections = [tf.GraphKeys.GLOBAL_VARIABLES]

        more_arg = {'reuse': reuse, 'initializer': initializer, 'collections': collections}

        y = x
        ws = []
        ys = []
        ms_size = 0
        for idx, args in enumerate(self.arch):
            args.update(more_arg)
            y, w, m_size = tf_layer[args['layer']](idx, y, args)
            ws.append(w)
            ys.append(y)
            ms_size += m_size
        main_logger.info("param: {}, memory size: {:.2f}MB".format(ms_size, ms_size * 4 / 1024 / 1024))
        return y, ws, ys

    def _def_model(self):
        s = tf.placeholder(tf.float32, [None, 84, 84, 4], name='s')
        a = tf.placeholder(tf.float32, [None, ], name='a')
        r = tf.placeholder(tf.float32, [None, ], name='r')
        t = tf.placeholder(tf.float32, [None, ], name='t')
        s_ = tf.placeholder(tf.float32, [None, 84, 84, 4], name='s_')
        with tf.variable_scope("online"):
            q, ws, ys = self._build_network(s, collections="online")
        with tf.variable_scope("target"):
            q_, ws_, ys_ = self._build_network(s_, collections="target")
        a_one_hot = tf.one_hot(a, depth=self.action_n, dtype=tf.float32)
        q_value = tf.reduce_sum(q * a_one_hot, axis=1)
        q_target = r + (1. - t) * self.gamma \
                       * tf.reduce_max(q_, axis=1, name='Qmax_s_')
        td_error = q_target - tf.stop_gradient(q_value)
        errors = huber_loss(td_error)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.args.lr, epsilon=1e-4)
        optimize_expr = minimize_and_clip(optimizer, errors, var_list=[w for w in ws if w])
        with tf.variable_scope('update_params'):
            update_params = tf.group(*[tf.assign(t, o) for t, o in zip([w for w in ws_ if w],
                                                                       [w for w in ws if w])])
        return {'ph': [s, a, r, t, s_], 'opt': optimize_expr, 'update': update_params}


agent = DQN()
agent.setup()
