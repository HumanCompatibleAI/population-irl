"""
CLI app that takes a given environment and RL algorithm and:
    - 1. trains the RL algorithm on the environment (trajectories discarded).
    - 2. upon convergence, runs the RL algorithm in the environment and logs
       the resulting trajectories.


"""

import argparse

from pirl import envs

ALGORITHMS = {} # TODO

def _check_in(cats, kind):
    def f(s):
        if s in cats:
            return s
        else:
            raise argparse.ArgumentTypeError("'{}' is not an {}".format(s, kind))

algorithm = _check_in(ALGORITHMS.keys(), 'RL algorithm')
environment = _check_in(envs.ENVIRONMENTS.keys(), 'environment')

def parse_args():
    desc = 'Log trajectories from an RL algorithm.'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('env', metavar='env', type=algorithm)
    parser.add_argument('algo', metavar='algo', type=environment)
    parser.add_argument('out_dir', metavar='out_dir', type=str)

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    env = envs.make(args.env)
    # algo = make_algo(args.algo)

    # TODO: train RL algorithm until convergence
    # N.B. (how to specify convergence criteria?)

    # TODO: run RL algorithm, log trajectories, write to args.out_dir