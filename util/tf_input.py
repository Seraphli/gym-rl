import threading
import tensorflow as tf


class EnqueueThread(threading.Thread):
    def __init__(self, sess, queue, sample, inputs):
        super(EnqueueThread, self).__init__()
        self.daemon = True
        self.sess = sess
        self.queue = queue
        self.sample = sample
        self.inputs = inputs

        self.op = self.queue.enqueue(inputs)

    def run(self):
        while True:
            data = self.sample()
            feed_dict = dict(zip(self.inputs, data))
            self.sess.run(self.op, feed_dict=feed_dict)
