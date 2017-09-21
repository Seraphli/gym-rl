import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


class Actor(object):
    def setup(self, action_n, replay):
        self.action_n = action_n
        self.replay = replay


def build_net():
    pass


num_episode = 100

actor_net = build_net()
critic_net = build_net()

env.make()
for _ in range(num_episode):
    done = False
    obs = env.reset()
    replay = Replay()
    t = 0
    while not done:
        act = actor_net.act(obs)
        obs_, reward, done, info = env.step(act)
        replay.add(obs, act, reward, done, obs_)
        t += 1
    R = 0
    for t_i in range(t):
        obs, act, reward, done, obs_ = replay.get(t_i)
        R = reward + gamma * R
        loss_a = actor_net.p(obs) * (R - critic_net.act(obs))
        loss_c = (R - critic_net.act(obs)) ** 2
