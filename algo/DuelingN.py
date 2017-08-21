from .Double_DQN import DoubleDQN
import tensorflow as tf
from util.tf_common import huber_loss, minimize_and_clip, tensorboard
from util.tf_layer import tf_layer
from util.util import pretty_num
import util.util as U


class DuelingN(DoubleDQN):
    def _def_algorithm(self):
        self.algorithm = 'DuelingN'

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
            {'layer': 'fc', 'size': 1},
            {'layer': 'fc', 'size': self.action_n},
        ]

    def _def_net_summary(self):
        self.arch_sum = [
            {'index': 1, 'vis_type': 'conv', 'tb_type': 'image'},
            {'index': 3, 'vis_type': 'conv', 'tb_type': 'image'},
            {'index': 5, 'vis_type': 'conv', 'tb_type': 'image'},
            {'index': 8, 'vis_type': 'fc', 'tb_type': 'image'},
            {'index': 10, 'vis_type': 'fc', 'tb_type': 'image'}
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
            if idx == 9:
                y, w, m_size = tf_layer[args['layer']](idx, ys[8], args)
            elif idx == 10:
                y, w, m_size = tf_layer[args['layer']](idx, ys[8], args)
            else:
                y, w, m_size = tf_layer[args['layer']](idx, y, args)
            ws.append(w)
            ys.append(y)
            ms_size += m_size
        y = tf.add(ys[10], ys[9] - tf.reduce_mean(ys[10], 1, keep_dims=True))
        U.main_logger.info("param: {}, memory size: {}B".format(ms_size, pretty_num(ms_size * 4, True)))
        return y, ws, ys

    def _def_model(self):
        s, a, r, t, s_ = self._def_input()
        with tf.variable_scope('online'):
            q, ws, ys = self._build_net(s, collections='online')
        with tf.variable_scope('online', reuse=True):
            qs_, _, _ = self._build_net(s_, collections='online')
        self._add_summary(ws, ys)
        with tf.variable_scope('target'):
            q_, ws_, ys_ = self._build_net(s_, collections='target')
        o_vars = [_w for w in ws if w for _w in w]
        t_vars = [_w for w in ws_ if w for _w in w]
        with tf.name_scope('q'):
            q_value = tf.reduce_sum(q * tf.one_hot(a, self.action_n), 1)
            q_max = tf.reduce_max(tf.multiply(
                tf.one_hot(tf.argmax(qs_, axis=1), self.action_n),
                q_), axis=1, name='q_max_s_')
            q_target = r + (1. - t) * self.gamma * q_max
        self.summary.append(tensorboard['scalar']('q_max', q_max[0]))
        with tf.name_scope('grad'):
            td_error = tf.stop_gradient(q_target) - q_value
            errors = huber_loss(td_error)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.args.lr, epsilon=1e-4)
            optimize_expr = minimize_and_clip(optimizer, errors, var_list=o_vars)
        self.summary.append(tensorboard['scalar']('loss', tf.reduce_mean(errors)))
        with tf.variable_scope('update_params'):
            update_expr = [tf.assign(t, o) for t, o in zip(t_vars, o_vars)]
        update_params = tf.group(*update_expr, name='update')
        eps = tf.placeholder(tf.float32, [], name='eps')
        with tf.name_scope('action'):
            batch_size = tf.shape(s)[0]
            random_actions = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=self.action_n, dtype=tf.int64)
            deterministic_actions = tf.argmax(q, axis=1)
            chose_random = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=1, dtype=tf.float32) < eps
        actions = tf.where(chose_random, random_actions, deterministic_actions, name='act')
        return {'ph': [s, a, r, t, s_], 'eps': eps, 'act': actions, 'opt': optimize_expr, 'update': update_params}


agent = DuelingN()
