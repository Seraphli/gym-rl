import tensorflow as tf
from utility.utility import main_logger


def conv2d(idx, x, args):
    l_name = args['layer'] + "_%d" % idx
    w_size = args['kernel_size'] + [args['input'], args['output']]
    b_size = [args['output']]
    m_size = args['kernel_size'][0] * args['kernel_size'][1] * args['input'] * args['output'] + args['output']
    main_logger.info("{}: w: {}, b: {}".format(l_name, w_size, b_size))
    with tf.variable_scope(l_name, reuse=args['reuse']):
        w = tf.get_variable("w", w_size, initializer=args['initializer'][0], collections=args['collections'])
        b = tf.get_variable("b", b_size, initializer=args['initializer'][1], collections=args['collections'])
        y = tf.nn.conv2d(x, w, strides=[1] + args['stride'] + [1], padding="VALID") + b
    return y, [w, b], m_size


def relu(idx, x, args):
    l_name = args['layer'] + "_%d" % idx
    w_size = []
    b_size = []
    m_size = 0
    main_logger.info("{}: w: {}, b: {}".format(l_name, w_size, b_size))
    with tf.variable_scope(l_name, reuse=args['reuse']):
        y = tf.nn.relu(x)
    return y, [], m_size


def flatten(idx, x, args):
    l_name = args['layer'] + "_%d" % idx
    w_size = []
    b_size = []
    m_size = 0
    main_logger.info("{}: w: {}, b: {}".format(l_name, w_size, b_size))
    with tf.variable_scope(l_name, reuse=args['reuse']):
        y = tf.contrib.layers.flatten(x)
    return y, [], m_size


def fc(idx, x, args):
    l_name = args['layer'] + "_%d" % idx
    w_size = [x.get_shape()[1].value, args['size']]
    b_size = [args['size']]
    m_size = x.get_shape()[1].value * args['size'] + args['size']
    main_logger.info("{}: w: {}, b: {}".format(l_name, w_size, b_size))
    with tf.variable_scope(l_name, reuse=args['reuse']):
        w = tf.get_variable("w", w_size, initializer=args['initializer'][0], collections=args['collections'])
        b = tf.get_variable("b", b_size, initializer=args['initializer'][1], collections=args['collections'])
        y = tf.matmul(x, w) + b
    return y, [w, b], m_size


tf_layer = {
    'conv2d': conv2d,
    'relu': relu,
    'flatten': flatten,
    'fc': fc
}
