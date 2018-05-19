import collections
import logging
import os
import random
import string
import time

from gym.utils import seeding
import numpy as np
import ray
import ray.services

logger = logging.getLogger('pirl.utils')

# Gym environment helpers

def sanitize_env_name(env_name):
    return env_name.replace('/', '_')

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

# Randomness & sampling

def create_seed(seed=None, max_bytes=8):
    return seeding.create_seed(seed, max_bytes=max_bytes)


def discrete_sample(prob, rng):
    """Sample from discrete probability distribution, each row of prob
       specifies class probabilities."""
    return (np.cumsum(prob) > rng.rand()).argmax()


# Modified from https://stackoverflow.com/questions/2257441/random-string-generation-with-upper-case-letters-and-digits-in-python
def id_generator(size=8):
    choices = random.choices(string.ascii_uppercase + string.digits, k=size)
    return ''.join(choices)

# Logging

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

# Convenience functions for nested dictionaries

# Modified from:
# https://stackoverflow.com/questions/25833613/python-safe-method-to-get-value-of-nested-dictionary
def safeset(dic, keys, value):
    for key in keys[:-1]:
        dic = dic.setdefault(key, collections.OrderedDict())
    dic[keys[-1]] = value


# Modified from:
# https://stackoverflow.com/questions/32935232/python-apply-function-to-values-in-nested-dictionary
def safeget(dct, keys):
    for key in keys:
        return dct.setdefault(key, collections.OrderedDict())
    return dct


def map_nested_dict(ob, func, init=[], level=0):
    if isinstance(ob, collections.Mapping) and (level == 0 or len(init) < level):
        return {k: map_nested_dict(v, func, init + [k]) for k, v in ob.items()}
    else:
        return func(ob, init)


def _get_nested_dict_helper(future, _keys):
    return ray.get(future)

def ray_get_nested_dict(ob, level=0):
    return map_nested_dict(ob, _get_nested_dict_helper, level=level)

# GPU Management

# Ray does not support fractional GPU resources (issue #402)
# Workaround: pretend we have more GPUs than we do!
GPU_MULTIPLIER = 4

def get_num_fake_gpus(max_gpu=1000):
    real_gpus = ray.services._autodetect_num_gpus()
    return min(real_gpus, max_gpu) * GPU_MULTIPLIER

def set_cuda_visible_devices():
    ids = ray.get_gpu_ids()
    # Fractional allocation across GPUs won't work reliably
    # e.g. what if we need a half, and two GPUs are both 3/4's full
    # Let's just handle the simple case of needing 1/GPU_MULTIPLIER of a GPU.
    if len(ids) == 0:
        # Algorithm didn't ask for any GPUs? OK, it won't get any.
        gpus = ''
    elif len(ids) == 1:
        # Algorithm asked for one GPU. Map it onto the appropriate one.
        gpus = str(ids[0] % GPU_MULTIPLIER)
    else:
        # Don't support asking for more than one fraction of the GPU.
        # The issue is Ray isn't guaranteed to allocate us slices from the
        # same GPU.
        raise ValueError("Requested task with >1 GPU (currently unsupported).")
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus