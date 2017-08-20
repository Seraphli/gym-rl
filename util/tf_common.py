import tensorflow as tf
from tensorflow import Tensor


def huber_loss(x, delta=1.0):
    """Reference: https://en.wikipedia.org/wiki/Huber_loss"""
    return tf.where(
        tf.abs(x) < delta,
        tf.square(x) * 0.5,
        delta * (tf.abs(x) - 0.5 * delta)
    )


def minimize_and_clip(optimizer, objective, var_list, clip_val=10):
    """Minimized `objective` using `optimizer` w.r.t. variables in
    `var_list` while ensure the norm of the gradients for each
    variable is clipped to `clip_val`
    """
    gradients = optimizer.compute_gradients(objective, var_list=var_list)
    for i, (grad, var) in enumerate(gradients):
        if grad is not None:
            gradients[i] = (tf.clip_by_norm(grad, clip_val), var)
    return optimizer.apply_gradients(gradients)


def find_factor(n):
    """Find the largest factor
    
    Args:
        n (int): number 

    Returns:
        int: largest factor
        int: another factor that equals n divided by largest factor
    """
    for i in range(int(n ** 0.5) + 1, 1, -1):
        if n % i == 0:
            return n // i, i


def vis_conv_layer(v):
    """Visualize convolution layer
    
    Args:
        v (Tensor): output of the layer,
            with shape (batch, height, width, channels)

    Returns:
        Tensor: combined image of the output

    """
    _, h, w, c = v.get_shape().as_list()
    cy, cx = find_factor(c)
    v = tf.slice(v, (0, 0, 0, 0), (1, -1, -1, -1))
    v = tf.reshape(v, (h, w, c))
    w += 4
    h += 4
    v = tf.image.resize_image_with_crop_or_pad(v, h, w)
    v = tf.reshape(v, (1, h, w, cy, cx))
    v = tf.transpose(v, (0, 3, 1, 4, 2))
    v = tf.reshape(v, (1, cy * h, cx * w, 1))
    return v


def vis_fc_layer(v):
    """Visualize full-connected layer
    
    Args:
        v (Tensor): output of the layer,
            with shape (batch, depth)

    Returns:
        Tensor: one image of the output

    """
    _, depth = v.get_shape().as_list()
    h, w = find_factor(depth)
    v = tf.slice(v, (0, 0), (1, -1))
    v = tf.reshape(v, (1, h, w, 1))
    return v


visualize = {'conv': vis_conv_layer, 'fc': vis_fc_layer}
tensorboard = {'image': tf.summary.image, 'histogram': tf.summary.histogram}
