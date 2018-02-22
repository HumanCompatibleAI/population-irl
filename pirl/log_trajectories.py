"""
CLI app that takes a given environment and RL algorithm and:
    - 1. trains the RL algorithm on the environment (trajectories discarded).
    - 2. upon convergence, runs the RL algorithm in the environment and logs
       the resulting trajectories.


"""

import argparse
from datetime import datetime
import logging
import tempfile
import os

import numpy as np
import gym

from pirl import envs
from pirl import agents

logger = logging.getLogger('pirl.log_trajectories')

ALGORITHMS = {
    'value_iteration': lambda env: agents.tabular.value_iteration(env)[0],
}

#TODO: refactor once structure is clearer
# Should this be pushed into agents package?
# Will different agents require some setup code?
def make_algo(algo):
    return ALGORITHMS[algo]

def _check_in(cats, kind):
    def f(s):
        if s in cats:
            return s
        else:
            raise argparse.ArgumentTypeError("'{}' is not an {}".format(s, kind))
    return f
algorithm = _check_in(ALGORITHMS.keys(), 'RL algorithm')


def writable_dir(path):
    try:
        testfile = tempfile.TemporaryFile(dir=path)
        testfile.close()
    except OSError as e:
        desc = "Cannot write to '{}': {}".format(path, e)
        raise argparse.ArgumentTypeError(desc)

    return path

def parse_args():
    desc = 'Log trajectories from an RL algorithm.'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--num-trajectories', metavar='num_trajectories',
                        type=int, default=100)
    parser.add_argument('--out_dir', metavar='out_dir', default='./logs',
                        type=writable_dir)
    parser.add_argument('env', metavar='env')
    parser.add_argument('algo', metavar='algo', type=algorithm)

    return parser.parse_args()

def sample(env, policy):
    #TODO: generalize. This is specialised to fully-observable MDPs
    # and assumes policy is a deterministic mapping from states to actions.
    # Could also use Monitor to log this -- although think it's cleaner
    # to do it directly ourselves?

    states = []
    actions = []

    state = env.reset()
    states.append(state)

    done = False
    while not done:
        action = policy[state]
        state, reward, done, _ = env.step(action)
        actions.append(action)
        states.append(state)

    return states, actions

ISO_TIMESTAMP = "%Y%m%d_%H%M%S"

if __name__ == '__main__':
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    env = gym.make(args.env)
    logger.info('Initialising RL algorithm %s', args.algo)
    rl_algo = make_algo(args.algo)

    logger.info('Training %s in %s until convergence', args.algo, args.env)
    # TODO: In general, how should we specify convergence criteria?
    policy = rl_algo(env)

    logger.info('Sampling %d trajectories from trained policy',
                args.num_trajectories)
    # TODO: parallelize?
    trajectories = [sample(env, policy) for _i in range(args.num_trajectories)]
    states, actions = zip(*trajectories)
    states = np.array(states)
    actions = np.array(actions)

    timestamp = datetime.now().strftime(ISO_TIMESTAMP)
    out_dir = '{}-{}-{}.npz'.format(timestamp,
                                    args.env.replace('/', ':'),
                                    args.algo)
    path = os.path.join(args.out_dir, out_dir)
    logger.info('Done -- saving to %s', path)
    np.savez(path, states=states, actions=actions)