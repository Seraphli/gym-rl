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


def vis_conv_layer(v, shape):
    """Visualize convolution layer
    
    Args:
        v (Tensor): output of the layer,
            with shape (batch, height, width, channels)
        shape (tuple): a tuple of height, width,
            channels, number of images in one line, number of lines

    Returns:
        Tensor: combined image of the output

    """
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


def vis_fc_layer(v, shape):
    """Visualize full-connected layer
    
    Args:
        v (Tensor): output of the layer,
            with shape (batch, depth)
        shape (tuple): a tuple of height, width

    Returns:
        Tensor: one image of the output

    """
    h, w = shape
    v = tf.slice(v, (0, 0), (1, -1))
    v = tf.reshape(v, (1, h, w, 1))
    return v
