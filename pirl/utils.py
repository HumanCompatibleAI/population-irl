import collections
from decorator import decorate
import logging
import random
import traceback
import time
from multiprocessing import pool

from gym.utils import seeding
import numpy as np
import tensorflow as tf
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

def create_seed(seed=None, max_bytes=8):
    return seeding.create_seed(seed, max_bytes=max_bytes)

def random_seed(seed=None):
    seed = create_seed(seed + 'main')
    random.seed(seed)
    np.random.seed(seeding._int_list_from_bigint(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    tf.set_random_seed(seed)
    return seed


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


def _log_errors(f, *args, **kwargs):
    try:
        return f(*args, **kwargs)
    except Exception as e:
        logger.error('Error in subprocess: %s', traceback.format_exc())
        raise e


def log_errors(f):
    '''For use with multiprocessing. If an exception occurs, log it immediately
       and then reraise it. This gives early warning in the event of an error
       (by default, multiprocessing will wait on all other executions before
        raising or in any way reporting the exception).'''
    # Use the decorator module to preseve function signature.
    # In Python 3.5+, functools.wraps also preserves the signature returned by
    # inspect.signature, but not the (deprecated) inspect.getargspec.
    # Unfortunately, some other modules e.g. joblib that we depend on still use
    # the deprecated module.
    return decorate(f, _log_errors)


def nested_async_get(x, fn=lambda y: y):
    '''Invoked on a recursive data structure consisting of dicts and lists,
       returns the same-shaped data structure mapping a leaf node x to
       fn(x.get()) if x is an AsyncResult and fn(x) otherwise.'''
    if isinstance(x, dict):
        return {k: nested_async_get(v, fn) for k, v in x.items()}
    elif isinstance(x, collections.OrderedDict):
        res = [(k, nested_async_get(v, fn)) for k, v in x.items()]
        return collections.OrderedDict(res)
    elif isinstance(x, list):
        return [nested_async_get(v, fn) for v in x]
    elif isinstance(x, pool.AsyncResult):
        return fn(x.get())
    else:
        return fn(x)


def vectorized(x):
    '''Set an attribute to tell experiments pipeline whether to give
       function vectorized inputs.'''
    def helper(f):
        f.is_vectorized = x
        return f
    return helper


def is_vectorized(f):
    if hasattr(f, 'func'):  # handle functools.partial
        return is_vectorized(f.func)
    else:
        return f.is_vectorized