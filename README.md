# gym-rl

## Introduction

Fast implement of DQN, playing atari games

Usually take one and half hour to reach >300 score in Breakout or >18 in Pong

Test on GTX1080 and i7-6800K, with 16GB memory

Environment pool size and replay buffer size will effect memory usage

The memory usage using default setting is ~7GB

## How to run

```bash
git clone https://github.com/Seraphli/gym-rl.git
cd gym-rl
PYTHONPATH=. python exp/WIP_DQN_train.py
```

See help for more information and configuration
```
PYTHONPATH=. python exp/WIP_DQN_train.py --help
```

## Thanks

[baselines][1]

[tensorpack][2]

[1]: https://github.com/openai/baselines
[2]: https://github.com/ppwwyyxx/tensorpack

