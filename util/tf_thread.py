import threading


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


class OptThread(threading.Thread):
    def __init__(self, sess, queue, op):
        super(OptThread, self).__init__()
        self.daemon = True
        self.sess = sess
        self.queue = queue
        self.op = op

    def run(self):
        while self.queue.get() == "opt":
            self.sess.run(self.op)