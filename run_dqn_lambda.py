from argparse import ArgumentParser
from distutils.util import strtobool
import itertools
import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'

import numpy as np
import tensorflow as tf

from vrc import atari_env
from vrc.dqnl_agent import DQNLAgent


def allow_gpu_memory_growth():
    try:
        gpu_list = tf.config.list_physical_devices('GPU')
    except AttributeError:
        gpu_list = tf.config.experimental.list_physical_devices('GPU')

    for gpu in gpu_list:
        tf.config.experimental.set_memory_growth(gpu, True)


def parse_kwargs():
    parser = ArgumentParser()
    parser.add_argument('--game', type=str, default='pong')
    parser.add_argument('--virtual_cache', type=strtobool, default=True)
    parser.add_argument('--lambd', type=float, default=0.0)
    parser.add_argument('--timesteps', type=int, default=10_000_000)
    parser.add_argument('--seed', type=int, default=0)
    # TODO: block size, cache size, etc
    return vars(parser.parse_args())


def setup_env(game, seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)

    env = atari_env.make(game)
    env.seed(seed)
    env.action_space.seed(seed)
    return env


def train(env, agent, timesteps):
    state = env.reset()

    for t in itertools.count(start=1):
        if t >= timesteps and info['real_done']:
            env.close()
            break

        action = agent.policy(t, state)
        next_state, reward, done, info = env.step(action)
        agent.update(t, state, action, reward, done)

        state = env.reset() if info['real_done'] else next_state


def main(kwargs):
    game = kwargs.pop('game')
    seed = kwargs.pop('seed')
    timesteps = kwargs.pop('timesteps')

    env = setup_env(game, seed)
    agent = DQNLAgent(env, **kwargs)
    train(env, agent, timesteps)


if __name__ == '__main__':
    allow_gpu_memory_growth()
    kwargs = parse_kwargs()
    main(kwargs)
