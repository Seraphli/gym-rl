#!/usr/bin/env python
from algo.DQN import agent as agent
from util.train import Train


if __name__ == '__main__':
    Train(agent).run()
