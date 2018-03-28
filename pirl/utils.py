import collections
import functools
import logging
import traceback
import time

from gym.utils import seeding
import numpy as np
import torch

logger = logging.getLogger('pirl.utils')

def getattr_unwrapped(env, attr):
    """Get attribute attr from env, or one of the nested environments.

    Args:
        - env(gym.Wrapper or gym.Env): a (possibly wrapped) environment.
        - attr: name of the attribute

    Returns:
        env.attr, if present, otherwise env.unwrapped.attr and so on recursively.
    """
    try:
        return getattr(env, attr)
    except AttributeError:
        if env.env == env:
            raise
        else:
            return getattr_unwrapped(env.env, attr)


def random_seed(seed=None):
    seed = seeding.create_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def discrete_sample(prob, rng):
    """Sample from discrete probability distribution, each row of prob
       specifies class probabilities."""
    return (np.cumsum(prob) > rng.rand()).argmax()


class TrainingIterator(object):
    def __init__(self, num_iters, name=None,
                 heartbeat_iters=None, heartbeat_time=1.0):
        self._max_iters = num_iters
        self._name = name
        self._h_iters = heartbeat_iters
        self._h_time = heartbeat_time
        self._last_h_time = time.time()
        self._vals = {}

    @property
    def i(self):
        return self._i

    @property
    def heartbeat(self):
        return self._heartbeat

    @property
    def vals(self):
        return self._vals

    @property
    def elapsed(self):
        return time.time() - self._last_h_time

    def status(self):
        msg = '[{}] '.format(self._name) if self._name is not None else ''
        msg += '{}/{}: elapsed {:.2f}'
        return msg.format(self.i, self._max_iters, self.elapsed)

    def __iter__(self):
        last_heartbeat_i = 0
        for i in range(self._max_iters):
            self._i = i
            cur_time = time.time()
            self._heartbeat = False
            if self._h_iters is not None:
                self._heartbeat |= (i - last_heartbeat_i) >= self._h_iters
            if self._h_time is not None:
                self._heartbeat |= (cur_time - self._last_h_time) > self._h_time

            if self.heartbeat:
                logger.debug(self.status())
                self._last_h_time = cur_time
                last_heartbeat_i = i

            yield i

    def record(self, k, v):
        self._vals.setdefault(k, collections.OrderedDict())[self._i] = v

def log_errors(f):
    '''For use with multiprocessing. If an exception occurs, log it immediately
       and then reraise it. This gives early warning in the event of an error
       (by default, multiprocessing will wait on all other executions before
        raising or in any way reporting the exception).'''
    @functools.wraps(f)
    def helper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            logger.error('Error in subprocess: %s', traceback.format_exc())
            raise e
    return helper