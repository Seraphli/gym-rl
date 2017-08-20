import threading


class EnqueueThread(threading.Thread):
    def __init__(self, sess, queue, sample, inputs):
        super(EnqueueThread, self).__init__()
        self.daemon = True
        self.sess = sess
        self.queue = queue
        self.sample = sample
        self.inputs = inputs

        self.op = self.queue.enqueue(inputs, name='enqueue')

    def run(self):
        while True:
            data = self.sample()
            feed_dict = dict(zip(self.inputs, data))
            self.sess.run(self.op, feed_dict=feed_dict)


class OpThread(threading.Thread):
    def __init__(self, sess, queue, op):
        super(OpThread, self).__init__()
        self.daemon = True
        self.sess = sess
        self.queue = queue
        self.op = op

    def run(self):
        while self.queue.get() == "run":
            self.sess.run(self.op)


class SummaryThread(threading.Thread):
    def __init__(self, sess, queue, op, sw):
        super(SummaryThread, self).__init__()
        self.daemon = True
        self.sess = sess
        self.queue = queue
        self.op = op
        self.sw = sw
        self.step = 0

    def run(self):
        while self.queue.get() == "run":
            summary = self.sess.run(self.op)
            for s in summary:
                self.sw.add_summary(s, self.step)
            self.step += 1
