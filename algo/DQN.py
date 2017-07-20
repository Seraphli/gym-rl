import tensorflow as tf, numpy as np, threading, time
from tensorflow.python.client import timeline
from utility.epsilon import LinearAnnealEpsilon
from utility.exp_replay import Memory
from utility.utility import get_path
from utility.preprocess import *

DQN_REPLAY_SHAPE = (4,
                    ((84, 84, 4), np.uint8),
                    ((), np.uint8),
                    ((), np.int32),
                    ((), np.bool))


class DQN(object):
    """
    This class contain following main parts:
    1. Create network for q learning (_create_network)
    2. Learning step (step)

    Experience Replay locate in utility.exp_replay
    Epsilon decay locate in utility.epsilon
    Tensorflow layer definition locate in utility.tf_layer
    Preprocess is not a part inside DQN, locate in utility.preprocess
    """

    def __init__(self, sess, env, config, logger=None):
        self._sess = sess
        self._env = env
        self._config = config
        self._logger = logger
        self._action_n = self._env["action_n"]
        self.setup()

    def setup(self):
        self._def_algorithm()
        self._def_step_count()
        self._def_agent()
        self._def_tf_summary()
        self._def_tf_sl()

    def _def_algorithm(self):
        self._def_algorithm_name()
        self._def_algorithm_specify()
        self._def_algorithm_extend()

    def _def_algorithm_name(self):
        self._algo_name = 'DQN'

    def _def_tf_sl(self):
        self.saver = tf.train.Saver()

    def _def_tf_summary(self):
        self.train_summary = []
        self.episode_summary = []
        self.test_summary = []
        self.visualize_layers = []
        self.visualize_weights = []

        with tf.variable_scope('summary'):
            self.avg_score = tf.placeholder(tf.float32, [], name='avg')
            self.min_score = tf.placeholder(tf.float32, [], name='min')
            self.max_score = tf.placeholder(tf.float32, [], name='max')
            self.loss_sm = tf.placeholder(tf.float32, [], name='loss')

        self.episode_summary.append(tf.summary.scalar('train/avg', self.avg_score))
        self.test_summary.append(tf.summary.scalar('test/avg', self.avg_score))

        self.episode_summary.append(tf.summary.scalar('train/min', self.min_score))
        self.test_summary.append(tf.summary.scalar('test/min', self.min_score))

        self.episode_summary.append(tf.summary.scalar('train/max', self.max_score))
        self.test_summary.append(tf.summary.scalar('test/max', self.max_score))

        self.train_summary.append(tf.summary.scalar('train/epsilon', self.epsilon))
        self.train_summary.append(tf.summary.scalar('train/loss', self.loss_sm))
        for i in range(len(self.target_vl)):
            self.visualize_layers.append(tf.summary.image('train/layer_%d' % i, self.target_vl[i]))
        self.visualize_weights = self.target_vw

        self.run_metadata = tf.RunMetadata()

        self.save_path = get_path('tf_log/' + self._algo_name + '/' + self._env["name"])
        self.summary_writer = tf.summary.FileWriter(self.save_path, self._sess.graph)

    def _def_agent(self):
        self._def_initializer()
        self._def_queue()
        with tf.device('/gpu'):
            self._def_network()
            self._def_update_params()
            self._def_action()

    def _def_algorithm_specify(self):
        self.replay_start_size = self._config['Algorithm']['ReplayStartSize']
        self.preprocessor = DQNPrep()

    def _def_algorithm_extend(self):
        self.gamma = self._config['Gamma']
        self.lr = self._config['LR']
        self.momentum = self._config['Momentum']
        self.epsilon_cal = LinearAnnealEpsilon(self._config['Epsilon']['Initial'],
                                               self._config['Epsilon']['Final'],
                                               self._config['Epsilon']['Total'])
        self.eval_epsilon = self._config['Epsilon']['Eval']
        self.epsilon = tf.placeholder(tf.float32, [], name='epsilon')
        self.memory = Memory(self._config['ReplaySize'], DQN_REPLAY_SHAPE)
        self.minibatch_size = self._config['MiniBatchSize']
        self.target_net_update_frequency = self._config['TargetNetUpdateFrequency']
        self.update_frequency = self._config['UpdateFrequency']
        self.save_frequency = self._config['SaveFrequency']
        self.log_interval = self._config['LogInterval']

    def _def_step_count(self):
        with tf.variable_scope('step'):
            self.step_count = 0
            self.step_count_tensor = tf.get_variable('step_count', [], tf.int64, tf.zeros_initializer())
            self.step_count_inc = tf.assign(self.step_count_tensor, self.step_count_tensor + 1)

    def _visualize_conv_layer(self, v, shape):
        h, w, c, cy, cx = shape
        v = tf.slice(v, (0, 0, 0, 0), (1, -1, -1, -1))
        v = tf.reshape(v, (h, w, c))
        w += 4
        h += 4
        v = tf.image.resize_image_with_crop_or_pad(v, h, w)
        v = tf.reshape(v, (1, h, w, cy, cx))
        v = tf.transpose(v, (0, 3, 1, 4, 2))
        v = tf.reshape(v, (1, cy * h, cx * w, 1))
        return v

    def _visualize_fc_layer(self, v, shape):
        h, w = shape
        v = tf.slice(v, (0, 0), (1, -1))
        v = tf.reshape(v, (1, h, w, 1))
        return v

    def _def_initializer(self):
        self.w_init, self.b_init = tf.contrib.layers.variance_scaling_initializer(), tf.constant_initializer()

    def _build_network(self, x, name, reuse, trainable, visualize):
        layers_name = ["conv1", "conv2", "conv3", "flatten", "fc1", "fc2"]
        weighted_layers_name = ["conv1", "conv2", "conv3", "fc1", "fc2"]
        weights = []
        layers = []
        visualize_layer = []
        visualize_weight = []
        collect_names = [name, tf.GraphKeys.GLOBAL_VARIABLES]

        y = tf.to_float(x) / 255.0
        layers.append(y)

        with tf.variable_scope(layers_name[0] + "_" + name, reuse=reuse):
            w = tf.get_variable("w", [8, 8, 4, 32], initializer=self.w_init,
                                trainable=trainable, collections=collect_names)
            b = tf.get_variable("b", [32], initializer=self.b_init,
                                trainable=trainable, collections=collect_names)
            y = tf.nn.relu(tf.nn.conv2d(y, w, strides=[1, 4, 4, 1], padding="VALID") + b)
        weights += [w, b]
        layers.append(y)

        with tf.variable_scope(layers_name[1] + "_" + name, reuse=reuse):
            w = tf.get_variable("w", [4, 4, 32, 64], initializer=self.w_init,
                                trainable=trainable, collections=collect_names)
            b = tf.get_variable("b", [64], initializer=self.b_init,
                                trainable=trainable, collections=collect_names)
            y = tf.nn.relu(tf.nn.conv2d(y, w, strides=[1, 2, 2, 1], padding="VALID") + b)
        weights += [w, b]
        layers.append(y)

        with tf.variable_scope(layers_name[2] + "_" + name, reuse=reuse):
            w = tf.get_variable("w", [3, 3, 64, 64], initializer=self.w_init,
                                trainable=trainable, collections=collect_names)
            b = tf.get_variable("b", [64], initializer=self.b_init,
                                trainable=trainable, collections=collect_names)
            y = tf.nn.relu(tf.nn.conv2d(y, w, strides=[1, 1, 1, 1], padding="VALID") + b)
        weights += [w, b]
        layers.append(y)

        with tf.variable_scope(layers_name[3] + "_" + name, reuse=reuse):
            y = tf.reshape(y, [-1, 7 * 7 * 64], name="flatten")

        with tf.variable_scope(layers_name[4] + "_" + name, reuse=reuse):
            w = tf.get_variable("w", [7 * 7 * 64, 512], initializer=self.w_init,
                                trainable=trainable, collections=collect_names)
            b = tf.get_variable("b", [512], initializer=self.b_init,
                                trainable=trainable, collections=collect_names)
            y = tf.nn.relu(tf.matmul(y, w) + b)
        weights += [w, b]
        layers.append(y)

        with tf.variable_scope(layers_name[5] + "_" + name, reuse=reuse):
            w = tf.get_variable("w", [512, self._action_n], initializer=self.w_init,
                                trainable=trainable, collections=collect_names)
            b = tf.get_variable("b", [self._action_n], initializer=self.b_init,
                                trainable=trainable, collections=collect_names)
            y = tf.matmul(y, w) + b
        weights += [w, b]
        layers.append(y)

        if visualize:
            with tf.device('/cpu'):
                with tf.variable_scope("visualize_" + name, reuse=reuse):
                    visualize_layer.append(self._visualize_conv_layer(layers[0], (84, 84, 4, 1, 4)))
                    visualize_layer.append(self._visualize_conv_layer(layers[1], (20, 20, 32, 8, 4)))
                    visualize_layer.append(self._visualize_conv_layer(layers[2], (9, 9, 64, 8, 8)))
                    visualize_layer.append(self._visualize_conv_layer(layers[3], (7, 7, 64, 8, 8)))
                    visualize_layer.append(self._visualize_fc_layer(layers[4], (16, 32)))
                    visualize_layer.append(self._visualize_fc_layer(layers[5], (1, self._action_n)))

                for i in range(len(weighted_layers_name)):
                    visualize_weight.append(tf.summary.histogram(weighted_layers_name[i] + "/w", weights[2 * i]))
                    visualize_weight.append(tf.summary.histogram(weighted_layers_name[i] + "/b", weights[2 * i + 1]))
                    visualize_weight.append(tf.summary.histogram(weighted_layers_name[i] + "/layer", layers[i + 1]))

        return y, weights, layers, visualize_layer, visualize_weight

    def _def_var_queue(self, shape, dtype, name):
        # Variables
        with tf.device('/gpu'):
            var_buffer_out = tf.Variable([[0]], validate_shape=False, dtype=tf.float32, trainable=False)
            var_buffer_in = tf.Variable([0], validate_shape=False, dtype=tf.float32, trainable=False)

        data_in = tf.placeholder(dtype=dtype, shape=[None] + shape, name=name)

        queue = tf.FIFOQueue(10 * self.minibatch_size, [dtype], [shape])

        enqueue_op = queue.enqueue_many([data_in])
        dequeued_data_in = queue.dequeue_many(self.minibatch_size)

        with tf.device('/gpu'):
            move_buffer = tf.assign(var_buffer_out, var_buffer_in, validate_shape=False)
            put_in_buffer = tf.assign(var_buffer_in, tf.cast(dequeued_data_in, tf.float32), validate_shape=False)
            get_from_buffer = tf.cast(tf.identity(var_buffer_out), dtype)

        with tf.control_dependencies([move_buffer]):
            pull = tf.group(put_in_buffer)

        return data_in, enqueue_op, get_from_buffer, pull, put_in_buffer, move_buffer

    def _def_queue(self):
        self.batch_s = self._def_var_queue([84, 84, 4], tf.uint8, 'batch_s')
        self.batch_a = self._def_var_queue([], tf.int32, 'batch_a')
        self.batch_r = self._def_var_queue([], tf.float32, 'batch_r')
        self.batch_t = self._def_var_queue([], tf.bool, 'batch_r')
        self.batch_s_ = self._def_var_queue([84, 84, 4], tf.uint8, 'batch_s_')

    def enqueue_thread(self, sess, coord):
        while not coord.should_stop():
            s, a, r, t, s_ = self.memory.sample()
            sess.run(self.batch_s[1], feed_dict={self.batch_s[0]: s})
            sess.run(self.batch_a[1], feed_dict={self.batch_a[0]: a})
            sess.run(self.batch_r[1], feed_dict={self.batch_r[0]: r})
            sess.run(self.batch_t[1], feed_dict={self.batch_t[0]: a})
            sess.run(self.batch_s_[1], feed_dict={self.batch_s_[0]: s_})

    def _def_network(self):
        self.s = tf.placeholder(tf.uint8, [None, 84, 84, 4], name='s')
        self.a = tf.placeholder(tf.int32, [None, ], name='a')
        self.r = tf.placeholder(tf.float32, [None, ], name='r')
        self.t = tf.placeholder(tf.bool, [None, ], name='t')
        self.s_ = tf.placeholder(tf.uint8, [None, 84, 84, 4], name='s_')

        batch_s, batch_a, batch_r, batch_t, batch_s_ = \
            self.batch_s[2], self.batch_a[2], self.batch_r[2], self.batch_t[2], self.batch_s_[2]
        self.pull = self.batch_s[3], self.batch_a[3], self.batch_r[3], self.batch_t[3], self.batch_s_[3]

        self.q_s, _, _, _, _ \
            = self._build_network(self.s, "online", False, True, False)
        self.batch_q_s, self.online_w, self.online_l, self.online_vl, self.online_vw \
            = self._build_network(batch_s, "online", True, True, False)
        self.batch_q_s_, self.target_w, self.target_l, self.target_vl, self.target_vw \
            = self._build_network(batch_s_, "target", False, False, True)

        with tf.variable_scope('q_target'):
            self.q_target = batch_r + (1. - tf.cast(batch_t, tf.float32)) * self.gamma \
                                      * tf.reduce_max(self.batch_q_s_, axis=1, name='Qmax_s_')

        with tf.variable_scope('loss'):
            a_one_hot = tf.one_hot(batch_a, depth=self._action_n, dtype=tf.float32)
            self.net_q_value = tf.reduce_sum(self.batch_q_s * a_one_hot, axis=1)
            self.loss_op = tf.reduce_mean(tf.squared_difference(self.q_target, self.net_q_value), name='MSE')

        self.train_op = tf.train.RMSPropOptimizer(self.lr, decay=.95, momentum=self.momentum, epsilon=0.01) \
            .minimize(self.loss_op)

    def _def_update_params(self):
        with tf.variable_scope('update_params'):
            self.update_params = [tf.assign(t, o) for t, o in zip(self.target_w, self.online_w)]

    def _def_action(self):
        with tf.variable_scope('action'):
            random_action = tf.random_uniform((), 0, self._action_n, tf.int32)
            should_explore = tf.random_uniform((), 0, 1) < self.epsilon
            best_action = tf.cast(tf.arg_max(self.q_s[0], 0), tf.int32)
            self.action = tf.cond(should_explore, lambda: random_action, lambda: best_action)

    def take_action(self, observation, epsilon):
        return self._sess.run(self.action, feed_dict={self.s: [observation], self.epsilon: epsilon})

    def update_param(self):
        if self.step_count % self.target_net_update_frequency == 0:
            self._sess.run(self.update_params)

    # TODO: still need to speed up
    def _train(self):
        if len(self.memory) < self.replay_start_size:
            return
        if self.step_count % self.update_frequency != 0:
            return
        self._sess.run([self.batch_s[3], self.batch_a[3], self.batch_r[3], self.batch_t[3], self.batch_s_[3]])
        if self._config["TimelineDebug"] and self.step_count % self.log_interval == 0:
            _, loss = self._sess.run(
                [self.pull, self.train_op, self.loss_op],
                options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                run_metadata=self.run_metadata)
            trace = timeline.Timeline(step_stats=self.run_metadata.step_stats)
            trace_file = open(self.save_path + '/timeline ' +
                              time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) +
                              '.ctf.json', 'w')
            trace_file.write(trace.generate_chrome_trace_format())
            trace_file.close()
        else:
            _, loss = self._sess.run([self.train_op, self.loss_op])
        self.add_summary("_Train", loss)
        if self.step_count % self.log_interval == 0:
            self._sess.run([self.batch_s[3], self.batch_a[3], self.batch_r[3], self.batch_t[3], self.batch_s_[3]])
            v, w = self._sess.run([self.visualize_layers, self.visualize_weights])
            self.add_summary("_Visualize", (v, w))

    def load_session(self):
        checkpoint = tf.train.get_checkpoint_state(self.save_path)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self._sess, checkpoint.model_checkpoint_path)
            if self._logger != None:
                self._logger.info('Successfully loaded: %s' % checkpoint.model_checkpoint_path)
            return True
        else:
            if self._logger != None:
                self._logger.info('Could not find old network weights')
            return False

    def save_session(self):
        if len(self.memory) < self.replay_start_size:
            return
        if self.step_count % self.save_frequency == 0:
            self.saver.save(self._sess, self.save_path + '/model.ckpt', global_step=self.step_count)

    def prepare_train(self):
        self.memory.sample_enqueue(self.minibatch_size)
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(sess=self._sess, coord=coord)

        threads = []
        for n in range(4):
            t = threading.Thread(target=self.enqueue_thread, args=(self._sess, coord))
            t.daemon = True
            t.start()
            threads.append(t)

        # Put something in the input-buffer on the GPU.
        self._sess.run([self.batch_s[4], self.batch_a[4], self.batch_r[4], self.batch_t[4], self.batch_s_[4]])
        # Move to the output-side of the buffer on the GPU.
        self._sess.run([self.batch_s[5], self.batch_a[5], self.batch_r[5], self.batch_t[5], self.batch_s_[5]])

    def train(self, observation):
        self._sess.run(self.step_count_inc)
        self.step_count = self._sess.run(self.step_count_tensor)
        self._train()
        self.update_param()
        self.save_session()
        epsilon = self.epsilon_cal.get(self.step_count)
        return self.take_action(observation, epsilon)

    def eval(self, observation):
        return self.take_action(observation, self.eval_epsilon)

    def store_transition(self, transition):
        self.memory.append(transition)

    def preprocess(self, s):
        return self.preprocessor.process(s)

    def add_summary(self, phase, result):
        if phase == "_Train":
            step = self.step_count
            epsilon = self.epsilon_cal.get(step)
            loss = result
            summary = self._sess.run(self.train_summary,
                                     feed_dict={
                                         self.epsilon: epsilon,
                                         self.loss_sm: loss})
        elif phase == "_Visualize":
            step = self.step_count
            v, w = result
            summary = v + w
        elif phase == "Train":
            step, average_score, min_score, max_score = result
            summary = self._sess.run(self.episode_summary,
                                     feed_dict={
                                         self.avg_score: average_score,
                                         self.min_score: min_score,
                                         self.max_score: max_score})
        else:
            step, average_score, min_score, max_score = result
            summary = self._sess.run(self.test_summary,
                                     feed_dict={
                                         self.avg_score: average_score,
                                         self.min_score: min_score,
                                         self.max_score: max_score})
        for s in summary:
            self.summary_writer.add_summary(s, step)
