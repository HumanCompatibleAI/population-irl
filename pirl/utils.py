import collections
import functools
import logging
import inspect
import os
import random
import string
import tempfile
import time

from gym.utils import seeding
import hermes.backend.dict
import hermes.backend.redis
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

# Caching
def get_hermes():
    '''Creates a hermes.Hermes instance if one does not already exist;
       otherwise, returns the existing instance. This does two things:
         + Automatically picks between Redis (if available) and dict (if
           no Redis server is running).
         + By adding this extra layer of abstraction, prevents cloudpickle
           from choking on locks/sockets when serializing decorated functions.'''
    if get_hermes.cache is None:
        kwargs = {'ttl': None}
        try:
            host = os.environ.get('RAY_HEAD_IP', 'localhost')
            port = 6380
            db = 0
            get_hermes.cache = hermes.Hermes(hermes.backend.redis.Backend,
                                             host=host, port=port, db=0, **kwargs)
            logger.info('HermesCache: connected to %s:%d [db=%d]',
                        host, port, db)
        except ConnectionError:
            logger.info('HermesCache: no Redis server running on %s:%d, '
                        'falling back to local dict backend.', host, port)
            get_hermes.cache = hermes.Hermes(hermes.backend.dict.Backend)
    return get_hermes.cache
get_hermes.cache = None

def ignore_args(mangler, ignore):
    def name_entry(fn, *args, **kwargs):
        signature = inspect.signature(fn)
        bound = signature.bind(*args, **kwargs)
        for fld in ignore:
            if fld in bound.arguments:
                del bound.arguments[fld]
        return mangler.nameEntry(fn, *bound.args, **bound.kwargs)
    return name_entry

def cache(*oargs, **okwargs):
    '''Cache decorator taking the same arguments as the callable returned by
       hermes.Hermes. This is a hack to prevent cloudpickle choking on
       locks/sockets that are in hermes.Hermes. This decorator simply applies
       the hermes.Hermes decorator to the function, *when it is first called*,
       building the Hermes instance using get_hermes().'''
    def decorator(func):
        @functools.wraps(func)
        def helper(*args, **kwargs):
            if helper.wrapped is None:
                cache = get_hermes()

                if 'ignore' in okwargs:
                    ignore = okwargs.pop('ignore')
                    assert 'key' not in okwargs
                    okwargs['key'] = ignore_args(cache.mangler, ignore)

                helper.wrapped = cache(*oargs, **okwargs)(func)
            return helper.wrapped(*args, **kwargs)
        helper.wrapped = None
        return helper
    return decorator

def cache_and_log(log_to_tmp):
    '''cache_and_log(log_to_tmp) returns a decorator that combines utils.cache
       and log_to_tmp, an instance of utils.log_to_tmp_dir. In particular,
       the function it decorates must take an argument log_dir. In the event of
       a cache miss, this decorator produces the same behavior as a function
       decorated with utils.cache and log_to_tmp sequentially. The advantage
       comes in the case of a cache hit: cache_and_log will update the symlink
       to point to the log directory that the *cached* result output to,
       ensuring a complete set of logs.'''
    def wrapper(*oargs, ** okwargs):
        def decorator(func):
            @functools.wraps(func)
            def add_log_dir(*args, **kwargs):
                signature = inspect.signature(func)
                bound = signature.bind(*args, **kwargs)
                arguments = bound.arguments
                log_dir = arguments['log_dir']
                res = func(*args, **kwargs)
                return res, log_dir

            okwargs['ignore'] = okwargs.get('ignore', []) + ['log_dir']
            cached_fn = cache(*oargs, **okwargs)(add_log_dir)
            return log_to_tmp(cached_fn, returns_log_dir=True)
        return decorator
    return wrapper

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

log_dirs = set()
def log_to_tmp_dir(out_dir):
    '''Decorator for functions taking a parameter log_dir.

       Intercepts the log_dir provided by the callee (write ultimate_symlink),
       and creates a temporary directory (write tmp_log_dir) in config.OBJECT_DIR.
       A symlink is created from ultimate_symlink + a random suffix to tmp_log_dir.
       This tmp_log_dir is then passed as log_dir to the underlying function.
       If the function succeeds, it renames the symlink to ultimate_log_dir,
       unless this already exists.

       The purpose of this is to make logging robust to tasks being retried
       by a cluster manager, either due to failure (in which case ultimate_log_dir
       will not exist) or due to recomputing results evicted from cache (in
       which case ultimate_log_dir will exist).

       Note this should be applied to the function(s) closest to the point
       where logging output is actually produced. In particular, do not apply
       it to two functions that receive the same log_dir!'''
    def decorator(func, returns_log_dir=False):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Inspection & argument extraction
            signature = inspect.signature(func)
            bound = signature.bind(*args, **kwargs)
            arguments = bound.arguments
            ultimate_symlink = os.path.abspath(arguments['log_dir'])

            # Catch common misuse of this decorator
            if ultimate_symlink in log_dirs:
                msg = "Duplicate log directory '{}'".format(ultimate_symlink)
                raise AssertionError(msg)
            log_dirs.add(ultimate_symlink)

            # Make the directories
            tmp_symlink = '{}.{}'.format(ultimate_symlink, id_generator())
            os.makedirs(out_dir, exist_ok=True)
            tmp_log_dir = tempfile.mkdtemp(dir=out_dir)
            os.makedirs(os.path.dirname(tmp_symlink), exist_ok=True)
            os.symlink(tmp_log_dir, tmp_symlink, target_is_directory=True)

            # Call the function
            arguments['log_dir'] = tmp_log_dir
            res = func(*bound.args, **bound.kwargs)
            if returns_log_dir:
                res, new_log_dir = res
                if new_log_dir != tmp_log_dir:
                    os.rmdir(tmp_log_dir)
                    os.unlink(tmp_symlink)
                    os.symlink(new_log_dir, tmp_symlink, target_is_directory=True)

            # Success! (If the function threw an exception, we never reach here.)
            try:
                os.link(tmp_symlink, ultimate_symlink)
                os.unlink(tmp_symlink)
            except FileExistsError:
                logger.warning('Destination %s already exists (attempt to ' 
                               'rename %s). Was this a retried task?',
                               ultimate_symlink, tmp_symlink)

            return res
        return wrapper
    return decorator

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


def map_nested_dict(ob, func, init=[], level=1):
    '''Recurse to level depth in a nested mapping, applying func.'''
    if len(init) < level:
        assert isinstance(ob, collections.Mapping)
        return {k: map_nested_dict(v, func, init + [k], level=level)
                for k, v in ob.items()}
    else:
        return func(ob, init)

def leaf_map_nested_dict(ob, func, init=[]):
    '''Recurse to the leaf node in a nested mapping. Can be variable depth.'''
    if isinstance(ob, collections.Mapping):
        return {k: leaf_map_nested_dict(v, func, init + [k])
                for k, v in ob.items()}
    else:
        return func(ob, init)

def _get_nested_dict_helper(future, _keys):
    return ray.get(future)

def ray_get_nested_dict(ob, level=1):
    return map_nested_dict(ob, _get_nested_dict_helper, level=level)

def ray_leaf_get_nested_dict(ob):
    return leaf_map_nested_dict(ob, _get_nested_dict_helper)

# GPU Management

def set_cuda_visible_devices():
    if ray.worker.global_worker.mode == ray.worker.PYTHON_MODE:
        # Debug mode. get_gpu_ids() not supported -- do nothing.
        return
    ids = ray.get_gpu_ids()
    real_num_gpus = ray.services._autodetect_num_gpus()
    gpus = ','.join([str(x % real_num_gpus) for x in ids])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus