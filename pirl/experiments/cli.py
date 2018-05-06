"""
CLI app that takes a given environment and RL algorithm and:
    - 1. trains the RL algorithm on the environment (trajectories discarded).
    - 2. upon convergence, runs the RL algorithm in the environment and logs
       the resulting trajectories.


"""

import argparse
from datetime import datetime
import functools
import logging.config
import multiprocessing
from multiprocessing import Pool, Process, current_process
import os
import pickle
import tempfile
import git

from pirl.experiments import config, experiments

logger = logging.getLogger('pirl.experiments.cli')

class NoDaemonProcess(multiprocessing.Process):
    '''Make daemon attribute always return false.'''
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

class NoDaemonPool(multiprocessing.pool.Pool):
    '''A multiprocessing Pool whose processes do not have the daemon attribute
       set. This lets one have nested Pool's of processes. This is needed as
       this outer pool is used to run experiments in parallel, whereas an inner
       pool is used to perform rollouts in a given environment in parallel.

       This has the negative effect that these processes don't automatically die
       with the main process -- best to not use this unless you need to.'''
    Process = NoDaemonProcess

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
    parser.add_argument('--seed', metavar='STR', default='foobar', type=str)
    parser.add_argument('--video-every', metavar='N', default=100, type=int,
                        help='video every N episodes; 0 to disable.')
    parser.add_argument('--num-cores', metavar='N', default=None, type=int)
    parser.add_argument('experiments', metavar='experiment',
                        type=experiment_type, nargs='+')

    return parser.parse_args()


def init_worker(timestamp):
    current_process().name = current_process().name.replace('NoDaemonPoolWorker-', 'worker')
    logging.config.dictConfig(config.logging(timestamp))
    logger.debug('Worker started')

def git_hash():
    repo = git.Repo(path=os.path.realpath(__file__),
                    search_parent_directories=True)
    return repo.head.object.hexsha

ISO_TIMESTAMP = "%Y%m%d_%H%M%S"

if __name__ == '__main__':
    config.validate_config()  # fail fast and early

    # Logging
    current_process().name = 'master'
    timestamp = datetime.now().strftime(ISO_TIMESTAMP)
    logging.config.dictConfig(config.logging(timestamp))

    # Argument parsing
    args = parse_args()
    video_every = args.video_every if args.video_every != 0 else None
    logger.info('CLI args: %s', args)

    # Pool
    logger.info('Starting pool')
    pool = NoDaemonPool(args.num_cores,
                        initializer=functools.partial(init_worker, timestamp))

    # Experiment loop
    for experiment in args.experiments:
        # reseed so does not matter which order experiments are run in
        timestamp = datetime.now().strftime(ISO_TIMESTAMP)
        version = git_hash()
        out_dir = '{}-{}-{}'.format(experiment, timestamp, version)
        path = os.path.join(args.data_dir, out_dir)
        os.makedirs(path)

        res = experiments.run_experiment(experiment, pool, path,
                                         video_every, args.seed)

        logger.info('Experiment %s completed. Outcome:\n %s. Saving to %s.',
                    experiment, res['values'], path)
        with open('{}/results.pkl'.format(path), 'wb') as f:
            pickle.dump(res, f)

    pool.close()
    pool.join()
