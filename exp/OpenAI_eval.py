import os
import json
import glob
import argparse
import tensorflow as tf
import gym
from gym import wrappers
import datetime
import numpy as np
from util.util import get_path, boolean_flag, main_logger, Record

cfg_fn = get_path('cfg') + '/OpenAI.json'


def parse_args():
    parser = argparse.ArgumentParser("OpenAI evaluation script")
    parser.add_argument("--algo", type=str, default="DQN", choices=["DQN"], help="name of the algorithm")
    parser.add_argument("--env", type=str, default="Pong", help="name of the game")

    if os.path.exists(cfg_fn):
        with open(cfg_fn, 'r') as f:
            cfg = json.load(f)
        parser.add_argument("--api-key", type=str, default=cfg['APIKey'], help="OpenAI api key")
    else:
        parser.add_argument("--api-key", type=str, required=True, help="OpenAI api key")
    boolean_flag(parser, "save", default=True, help="whether or not to save api key")
    return parser.parse_args()


def load_model(args):
    sess = tf.Session()
    model_path = get_path('model/' + args.algo + '/' + args.env)
    subdir = next(os.walk(model_path))[1]
    if subdir:
        cmd = input("Found {} saved model(s), do you want to load? [y/N]".format(len(subdir)))
        if 'y' in cmd or 'Y' in cmd:
            if len(subdir) > 1:
                print("Choose one:")
                for i in range(len(subdir)):
                    state_fn = model_path + '/' + subdir[i] + '/state.json'
                    with open(state_fn, 'r') as f:
                        state = json.load(f)
                    print("[{}]: Score: {}, Path: {}".format(i, state['score'], subdir[i]))
                load_path = model_path + '/' + subdir[int(input("Index:"))]
            else:
                load_path = model_path + '/' + subdir[0]
            state_fn = load_path + '/state.json'
            with open(state_fn, 'r') as f:
                state = json.load(f)
            saver = tf.train.import_meta_graph(glob.glob(load_path + '/*.meta')[0])
            checkpoint = tf.train.get_checkpoint_state(load_path)
            if checkpoint and checkpoint.model_checkpoint_path:
                saver.restore(sess, checkpoint.model_checkpoint_path)
                main_logger.info("Successfully loaded model: Score: {}, Path: {}".
                                 format(state['score'], checkpoint.model_checkpoint_path))
                return True, sess
    main_logger.info("No model loaded")
    return False, None


def build_graph():
    graph = tf.get_default_graph()
    s = graph.get_tensor_by_name('s')
    a = graph.get_tensor_by_name('a')
    r = graph.get_tensor_by_name('r')
    t = graph.get_tensor_by_name('t')
    s_ = graph.get_tensor_by_name('s_')
    eps = graph.get_tensor_by_name('eps')
    actions = graph.get_tensor_by_name('act')
    optimize_expr = graph.get_tensor_by_name('opt')
    update_params = graph.get_tensor_by_name('update')
    return {'ph': [s, a, r, t, s_], 'eps': eps, 'act': actions, 'opt': optimize_expr, 'update': update_params}


def main():
    args = parse_args()
    with open(cfg_fn, 'w') as f:
        json.dump({'APIKey': args.api_key}, f)
    result, sess = load_model(args)
    if not result:
        main_logger.info("Evaluation exit")
        return
    model = build_graph()
    env = gym.make(args.env + '-v0')
    save_path = get_path('tmp/openai_eval/' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    main_logger.info("Evaluation will store in `{}`".format(save_path))
    env = wrappers.Monitor(env, save_path)
    rewards = []
    for i_episode in range(150):
        observation = env.reset()
        step = 0
        epi_reward = 0
        while True:
            step += 1
            action = sess.run(model['act'], feed_dict={
                model['ph'][0]: observation,
                model['eps']: 0})[0]
            observation, reward, done, info = env.step(action)
            epi_reward += reward
            if done:
                rewards.append(epi_reward)
                record = Record()
                record.add_key_value('Episode', i_episode)
                record.add_key_value('Total step', step)
                record.add_key_value('Episode reward', epi_reward)
                record.add_key_value('Reward (100 epi mean)', np.round(np.mean(rewards[-100:]), 2))
                main_logger.info("\n" + record.dumps())
                break
    gym.upload(save_path, api_key=args.api_key)
    main_logger.info("Evaluation complete")


if __name__ == '__main__':
    main()
