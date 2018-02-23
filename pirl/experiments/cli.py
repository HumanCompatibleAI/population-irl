"""
CLI app that takes a given environment and RL algorithm and:
    - 1. trains the RL algorithm on the environment (trajectories discarded).
    - 2. upon convergence, runs the RL algorithm in the environment and logs
       the resulting trajectories.


"""

import argparse
from datetime import datetime
import logging
import os
import pickle
import tempfile

from pirl.experiments import config, experiments

logger = logging.getLogger('pirl.experiments.cli')

def _check_in(cats, kind):
    def f(s):
        if s in cats:
            return s
        else:
            raise argparse.ArgumentTypeError("'{}' is not an {}".format(s, kind))
    return f
experiment_type = _check_in(config.EXPERIMENTS.keys(), 'experiment')


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
    parser.add_argument('--data_dir', metavar='dir', default='./data',
                        type=writable_dir)
    parser.add_argument('--seed', metavar='N', default=1234, type=int)
    parser.add_argument('experiments', metavar='experiment',
                        type=experiment_type, nargs='+')

    return parser.parse_args()

ISO_TIMESTAMP = "%Y%m%d_%H%M%S"

if __name__ == '__main__':
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    for experiment in args.experiments:
        # reseed so does not matter which order experiments are run in
        res = experiments.run_experiment(experiment, args.seed)
        
        timestamp = datetime.now().strftime(ISO_TIMESTAMP)
        out_dir = '{}-{}.pkl'.format(experiment, timestamp)
        path = os.path.join(args.data_dir, out_dir)
        logger.info('Experiment %s completed. Outcome:\n %s. Saving to %s.',
                    experiment, res['expected_value'], path)
        with open(path, 'wb') as f:
            pickle.dump(res, f)
