"""
CLI app that takes a given environment and RL algorithm and:
    - 1. trains the RL algorithm on the environment (trajectories discarded).
    - 2. upon convergence, runs the RL algorithm in the environment and logs
       the resulting trajectories.


"""

import argparse
from datetime import datetime
import logging.config
import os
import pickle

import git
import numpy as np
import ray

from pirl.experiments import config, experiments
from pirl.utils import get_num_fake_gpus

logger = logging.getLogger('pirl.experiments.cli')


def _check_in(cats, kind):
    def f(s):
        if s in cats:
            return s
        else:
            raise argparse.ArgumentTypeError("'{}' is not an {}".format(s, kind))
    return f
experiment_type = _check_in(config.EXPERIMENTS.keys(), 'experiment')


def parse_args():
    desc = 'Log trajectories from an RL algorithm.'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--data_dir', metavar='dir', default=config.DATA_DIR, type=str)
    parser.add_argument('--seed', metavar='STR', default='foobar', type=str)
    parser.add_argument('--video-every', metavar='N', default=0, type=int,
                        help='video every N episodes; disabled by default.')
    parser.add_argument('--num-cpu', metavar='N', default=None, type=int)
    parser.add_argument('--num-gpu', metavar='N', default=None, type=int)
    parser.add_argument('--ray-cluster', metavar='HOST', default=None, type=str)
    parser.add_argument('experiments', metavar='experiment',
                        type=experiment_type, nargs='+')

    return parser.parse_args()

def git_hash():
    repo = git.Repo(path=os.path.realpath(__file__),
                    search_parent_directories=True)
    return repo.head.object.hexsha

ISO_TIMESTAMP = "%Y%m%d_%H%M%S"

if __name__ == '__main__':
    config.validate_config()  # fail fast and early
    # Argument parsing
    args = parse_args()
    video_every = args.video_every if args.video_every != 0 else None
    logger.info('CLI args: %s', args)

    if args.ray_cluster is None:  # run locally
        num_gpu = get_num_fake_gpus(args.num_gpu)
        ray.init(num_cpus=args.num_cpu, num_gpus=num_gpu,
                 redirect_worker_output=True)
    else:  # connect to existing server (could still be a single machine)
        ray.init(redis_address=args.ray_cluster)

    # Experiment loop
    for experiment in args.experiments:
        # reseed so does not matter which order experiments are run in
        timestamp = datetime.now().strftime(ISO_TIMESTAMP)
        version = git_hash()
        out_dir = '{}-{}-{}'.format(experiment, timestamp, version)
        path = os.path.join(args.data_dir, out_dir)
        os.makedirs(path)

        cfg = config.EXPERIMENTS[experiment]
        res = experiments.run_experiment(cfg, path, video_every, args.seed)

        logger.info('Experiment %s completed. Outcome:\n %s. Saving to %s.',
                    experiment, res['values'], path)
        with open('{}/results.pkl'.format(path), 'wb') as f:
            pickle.dump(res, f)
