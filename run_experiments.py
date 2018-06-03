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
import subprocess

import ray

from pirl import config, experiments

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
    parser.add_argument('--seed', metavar='STR', default='foobar', type=str)
    parser.add_argument('--num-cpu', metavar='N', default=None, type=int)
    parser.add_argument('--num-gpu', metavar='N', default=None, type=int)
    parser.add_argument('--ray-server', metavar='HOST',
                        default=config.RAY_SERVER, type=str)
    parser.add_argument('experiments', metavar='experiment',
                        type=experiment_type, nargs='+')

    return parser.parse_args()

def git_hash():
    hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                   cwd=config.PROJECT_DIR)
    return hash.decode().strip()


# We pretend we have more GPUs to workaround Ray issue #402.
# This can be overridden by specifying --num-gpu.
GPU_MULTIPLIER = 4
# Timestamp for logging
ISO_TIMESTAMP = "%Y%m%d_%H%M%S"

if __name__ == '__main__':
    # Argument parsing
    args = parse_args()
    logger.info('CLI args: %s', args)

    if args.ray_server is None:  # run locally
        num_gpu = args.num_gpu
        if num_gpu is None:
            num_gpu = ray.services._autodetect_num_gpus() * GPU_MULTIPLIER
        ray.init(num_cpus=args.num_cpu, num_gpus=args.num_gpu,
                 redirect_worker_output=True)
    elif args.ray_server == "DEBUG":  # run in "Python" mode (single process)
        ray.init(driver_mode=ray.worker.PYTHON_MODE)
    else:  # connect to existing server (could still be a single machine)
        ray.init(redis_address=args.ray_server)

    # Experiment loop
    for experiment in args.experiments:
        # reseed so does not matter which order experiments are run in
        timestamp = datetime.now().strftime(ISO_TIMESTAMP)
        version = git_hash()
        out_dir = '{}-{}-{}'.format(experiment, timestamp, version)
        path = os.path.join(config.EXPERIMENTS_DIR, out_dir)
        os.makedirs(path)

        cfg = config.EXPERIMENTS[experiment]
        res = experiments.run_experiment(cfg, path, args.seed)

        logger.info('Experiment %s completed. Outcome:\n %s. Saving to %s.',
                    experiment, res['values'], path)
        with open('{}/results.pkl'.format(path), 'wb') as f:
            pickle.dump(res, f)
