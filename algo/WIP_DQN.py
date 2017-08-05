import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import argparse
from util.tf_layer import tf_layer
from util.util import main_logger, pretty_num, get_path
from util.tf_common import huber_loss, minimize_and_clip
from util.tf_thread import EnqueueThread, OptThread
from functools import partial
from queue import Queue
import json
import datetime


class DQN(object):
    def __init__(self):
        self.algorithm = 'DQN'

    def setup(self, action_n, replay):
        self.action_n = action_n
        self.replay = replay
        self._train = False
        self.gamma = 0.99
        self._def_net()
        self.model = self._def_model()
        if not self.load_model():
            init = tf.global_variables_initializer()
            self.sess.run(init)
        self.sess.graph.finalize()
        self.sess.run(self.model['update'])

    def parse_args(self):
        """Arguments for command line"""
        parser = argparse.ArgumentParser("DQN experiments for Atari games")
        parser.add_argument("--env", type=str, default="Pong", help="name of the game")
        parser.add_argument("--env-size", type=int, default=8, help="number of the environment")

        parser.add_argument("--replay-buffer-size", type=int, default=int(1e5), help="replay buffer size")
        parser.add_argument("--lr", type=float, default=1e-4, help="learning rate for Adam optimizer")
        parser.add_argument("--num-iters", type=int, default=800,
                            help="total number of iterations to run the environment for")
        parser.add_argument("--batch-size", type=int, default=32,
                            help="number of transitions to optimize at the same time")
        parser.add_argument("--learning-freq", type=int, default=4,
                            help="number of iterations between every optimization step")
        parser.add_argument("--target-update-freq", type=int, default=40000,
                            help="number of iterations between every target network update")

        self.args = parser.parse_args()
        return self.args

    def make_session(self):
        """Make and return a tensorflow session
        
        Returns:
            Session: tensorflow session
        """
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        cfg = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.Session(config=cfg)
        return self.sess

    def _def_net(self):
        """Definition of network architecture"""
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

    def _build_net(self, x, reuse=False, initializer=None, collections=None):
        if not initializer:
            initializer = [tf.contrib.layers.variance_scaling_initializer(), tf.constant_initializer()]

        if collections:
            collections = [collections, tf.GraphKeys.GLOBAL_VARIABLES]
        else:
            collections = [tf.GraphKeys.GLOBAL_VARIABLES]

        more_arg = {'reuse': reuse, 'initializer': initializer, 'collections': collections}

        y = tf.cast(x, tf.float32) / 255.0
        ws = []
        ys = []
        ms_size = 0
        for idx, args in enumerate(self.arch):
            args.update(more_arg)
            y, w, m_size = tf_layer[args['layer']](idx, y, args)
            ws.append(w)
            ys.append(y)
            ms_size += m_size
        main_logger.info("param: {}, memory size: {}B".format(ms_size, pretty_num(ms_size * 4, True)))
        return y, ws, ys

    def _def_input(self):
        s = tf.placeholder(tf.uint8, [None, 84, 84, 4], name='s')
        a = tf.placeholder(tf.uint8, [None, ], name='a')
        r = tf.placeholder(tf.float32, [None, ], name='r')
        t = tf.placeholder(tf.float32, [None, ], name='t')
        s_ = tf.placeholder(tf.uint8, [None, 84, 84, 4], name='s_')
        inputs = s, a, r, t, s_
        self.queue = queue = tf.FIFOQueue(50, [i.dtype for i in inputs])
        replay_sample = partial(self.replay.sample, batch_size=self.args.batch_size)
        self.qt = EnqueueThread(self.sess, queue, replay_sample, inputs)
        sample = queue.dequeue()
        for s, i in zip(sample, inputs):
            s.set_shape(i.get_shape())
        return sample

    def _def_model(self):
        s, a, r, t, s_ = self._def_input()
        with tf.variable_scope('online'):
            q, ws, ys = self._build_net(s, collections='online')
        with tf.variable_scope('target'):
            q_, ws_, ys_ = self._build_net(s_, collections='target')
        o_vars = [_w for w in ws if w for _w in w]
        t_vars = [_w for w in ws_ if w for _w in w]
        q_value = tf.reduce_sum(q * tf.one_hot(a, self.action_n), 1)
        q_target = r + (1. - t) * self.gamma * tf.reduce_max(q_, axis=1, name='Qmax_s_')
        td_error = tf.stop_gradient(q_target) - q_value
        errors = huber_loss(td_error)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.args.lr, epsilon=1e-4)
        optimize_expr = minimize_and_clip(optimizer, errors, var_list=o_vars)
        with tf.variable_scope('update_params'):
            update_expr = [tf.assign(t, o) for t, o in zip(t_vars, o_vars)]
            update_params = tf.group(*update_expr)
        eps = tf.placeholder(tf.float32, [], name='t')
        with tf.variable_scope('action'):
            batch_size = tf.shape(s)[0]
            random_actions = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=self.action_n, dtype=tf.int64)
            deterministic_actions = tf.argmax(q, axis=1)
            chose_random = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=1, dtype=tf.float32) < eps
            actions = tf.where(chose_random, random_actions, deterministic_actions)
        return {'ph': [s, a, r, t, s_], 'eps': eps, 'act': actions, 'opt': optimize_expr, 'update': update_params}

    def take_action(self, observation, epsilon):
        return tf.get_default_session().run(self.model['act'], feed_dict={
            self.model['ph'][0]: observation,
            self.model['eps']: epsilon
        })

    def update_target(self):
        self.sess.run(self.model['update'])

    def train(self, times=1):
        if not self._train:
            self.qt.start()
            self.opt_queue = Queue()
            OptThread(self.sess, self.opt_queue, self.model['opt']).start()
            self._train = True
        for _ in range(times):
            self.opt_queue.put('opt')

    def load_model(self):
        self.saver = tf.train.Saver(max_to_keep=50)
        model_path = get_path('tflog/' + self.algorithm + '/' + self.args.env)
        subdir = next(os.walk(model_path))[1]
        if subdir:
            cmd = input("Found {} saved model(s), do you want to load? [y/N]".format(len(subdir)))
            if 'y' in cmd or 'Y' in cmd:
                if len(subdir) > 1:
                    print("Choose one:")
                    for i in range(len(subdir)):
                        state_fn = model_path + '/' + subdir[i] + '/state.json'
                        with open(state_fn, 'r') as f:
                            state = json.load(f)
                        print("[{}]: Score: {}, Path: {}".format(i, state['score'], subdir[i]))
                    load_path = model_path + '/' + subdir[int(input("Index:"))]
                else:
                    load_path = model_path + '/' + subdir[0]
                state_fn = load_path + '/state.json'
                with open(state_fn, 'r') as f:
                    state = json.load(f)
                checkpoint = tf.train.get_checkpoint_state(load_path)
                if checkpoint and checkpoint.model_checkpoint_path:
                    self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
                    main_logger.info("Successfully loaded model: Score: {}, Path: {}".
                                     format(state['score'], checkpoint.model_checkpoint_path))
                    self.score = state['score']
                    return True
        self.score = None
        main_logger.info("No model loaded")
        return False

    def save_model(self):
        save_path = get_path('tflog/' + self.algorithm
                             + '/' + self.args.env
                             + '/' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
        main_logger.info("Save model at {} with score {:.2f}".format(save_path, self.score))
        self.saver.save(self.sess, save_path + '/model.ckpt')
        with open(save_path + '/state.json', 'w') as f:
            json.dump({'score': self.score, 'args': self.args}, f)


agent = DQN()
