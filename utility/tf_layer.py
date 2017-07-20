import tensorflow as tf


def conv2d(x, w_shape, w_init, stride, c_names):
    w = tf.get_variable("w", w_shape, initializer=w_init, collections=c_names)
    return tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding="VALID")


def conv_layer(x, w_shape, b_shape, w_init, b_init, stride, l_name, c_names):
    with tf.variable_scope(l_name):
        b = tf.get_variable("b", b_shape, initializer=b_init, collections=c_names)
        return tf.nn.relu(conv2d(x, w_shape, w_init, stride, c_names) + b)


def fc_layer(x, w_shape, b_shape, w_init, b_init, l_name, c_names):
    with tf.variable_scope(l_name):
        w = tf.get_variable("w", w_shape, initializer=w_init, collections=c_names)
        b = tf.get_variable("b", b_shape, initializer=b_init, collections=c_names)
        return tf.matmul(x, w) + b


def fc_relu_layer(x, w_shape, b_shape, w_init, b_init, l_name, c_names):
    with tf.variable_scope(l_name):
        w = tf.get_variable("w", w_shape, initializer=w_init, collections=c_names)
        b = tf.get_variable("b", b_shape, initializer=b_init, collections=c_names)
        return tf.nn.relu(tf.matmul(x, w) + b)
