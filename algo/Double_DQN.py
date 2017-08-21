from .DQN import DQN
import tensorflow as tf
from util.tf_common import huber_loss, minimize_and_clip, tensorboard


class DoubleDQN(DQN):
    def __init__(self):
        super(DoubleDQN, self).__init__()
        self.algorithm = 'DoubleDQN'

    def _def_model(self):
        s, a, r, t, s_ = self._def_input()
        with tf.variable_scope('online'):
            q, ws, ys = self._build_net(s, collections='online')
        self._add_summary(ws, ys)
        with tf.variable_scope('target'):
            q_, ws_, ys_ = self._build_net(s_, collections='target')
        o_vars = [_w for w in ws if w for _w in w]
        t_vars = [_w for w in ws_ if w for _w in w]
        with tf.name_scope('q'):
            q_value = tf.reduce_sum(q * tf.one_hot(a, self.action_n), 1)
            q_max = tf.reduce_max(q_, axis=1, name='q_max_s_')
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
