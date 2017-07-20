import math


class Epsilon(object):
    def get(self, step):
        return


class ConstantEpsilon(Epsilon):
    def __init__(self, value):
        self.value = value

    def get(self, step):
        return self.value


class LinearAnnealEpsilon(Epsilon):
    def __init__(self, initial, final, total):
        self.initial = initial
        self.final = final
        self.total = total
        self.decay = (self.final - self.initial) / self.total

    def get(self, step):
        return max(self.final, min(self.initial + self.decay * step, self.initial))


class ExpDecayEpsilon(Epsilon):
    """
    Using formula f(t) = _N*a^(-_lambda*t)
    """

    def __init__(self, initial, final, total, base):
        with tf.variable_scope('epsilon'):
            self.initial = tf.get_variable('initial', [], tf.float32, tf.constant_initializer(initial))
            self.final = tf.get_variable('final', [], tf.float32, tf.constant_initializer(final))
            self.step = tf.get_variable('step', [], tf.float32, tf.zeros_initializer())
            self.total = tf.get_variable('total', [], tf.float32, tf.constant_initializer(total))
            self.base = tf.get_variable('base', [], tf.float32, tf.constant_initializer(base))
            self._N = self.initial
            self._lambda = -tf.log(0.05) / tf.log(self.base) / self.total
        self.step_op = self._N * tf.pow(self.base, -self._lambda * self.step)

    def run_step(self, _step):
        return tf.get_default_session().run(self.step_op, feed_dict={self.step: _step})
