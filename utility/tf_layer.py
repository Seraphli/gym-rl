import tensorflow as tf


def conv2d(idx, x, args):
    with tf.variable_scope(args['layer'] + "_" + idx, reuse=args['reuse']):
        w = tf.get_variable("w", args['kernel_size'] + [args['input'], args['output']],
                            initializer=args['initializer'][0], trainable=args['trainable'],
                            collections=args['collections'])
        b = tf.get_variable("b", [args['output']], initializer=args['initializer'][1], trainable=args['trainable'],
                            collections=args['collections'])
        y = tf.nn.conv2d(x, w, strides=[1] + args['stride'] + [1], padding="VALID") + b
    return [w, b], y


def relu(idx, x, args):
    with tf.variable_scope(args['layer'] + "_" + idx, reuse=args['reuse']):
        y = tf.nn.relu(x)
    return [], y


def flatten(idx, x, args):
    with tf.variable_scope(args['layer'] + "_" + idx, reuse=args['reuse']):
        y = tf.contrib.layers.flatten(x)
    return [], y


def fc(idx, x, args):
    with tf.variable_scope(args['layer'] + "_" + idx, reuse=args['reuse']):
        w = tf.get_variable("w", [tf.shape(x)[1], args['size']], initializer=args['initializer'][0],
                            trainable=args['trainable'], collections=args['collections'])
        b = tf.get_variable("b", args['size'], initializer=args['initializer'][1], trainable=args['trainable'],
                            collections=args['collections'])
        y = tf.matmul(x, w) + b
    return [w, b], y


tf_layer = {
    'conv2d': conv2d,
    'relu': relu,
    'flatten': flatten,
    'fc': fc
}
