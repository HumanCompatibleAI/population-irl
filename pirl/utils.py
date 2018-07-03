import collections
from distutils.dir_util import copy_tree
import functools
import logging
import inspect
import os
import random
import socket
import string
import sys
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
       otherwise, returns the existing instance.''' 
    if get_hermes.cache is None:
        kwargs = {'ttl': None}
        # Use socket.gethostname() not localhost.
        # If running on the master, the function might get cloudpickle'd
        # and sent to a remote machine, in which case we want
        # the address to point back to us.
        host = os.environ.get('RAY_HEAD_IP', socket.gethostname())
        port = 6380
        db = 0
        get_hermes.cache = hermes.Hermes(hermes.backend.redis.Backend,
                                         host=host, port=port, db=0, **kwargs)
        logger.info('HermesCache: connected to %s:%d [db=%d]',
                    host, port, db)
    return get_hermes.cache
get_hermes.cache = None

def cache_key_func(mangler, func_module, func_name, ignore=None):
    @functools.wraps(mangler.nameEntry)
    def name_entry(fn, *args, **kwargs):
        #TODO: remove the func_name argument once cloudpickle issue #176 is fixed
        #cloudpickle plays havoc with names of decorated functions, but it
        #preserves objects that are in a closure correctly. So patch up the
        #function name (yuck)
        fn.__module__ = func_module
        fn.__name__ = func_name
        signature = inspect.signature(fn)
        bound = signature.bind_partial(*args, **kwargs)
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
        cache = get_hermes()

        ignore = []
        if 'ignore' in okwargs:
            ignore = okwargs.pop('ignore')

        assert 'key' not in okwargs
        key_fn = cache_key_func(cache.mangler, func.__module__,
                                func.__name__, ignore)
        okwargs['key'] = key_fn

        return cache(*oargs, **okwargs)(func)
    return decorator

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

def _temporary_error(exc):
    '''Kills the worker reporting the exception exc. This causes Ray to retry
       the task, whereas if we raise the Exception it will mark the task as
       permanently failed.'''
    #TODO: Must be a better way? See Ray issue #2141
    logger.critical('Fatal error. This is probably a node-specific error; '
                    ' killing worker to force a retry.', exc_info=exc)
    sys.exit(-1)

log_dirs = set()
def cache_and_log(out_dir):
    '''Given an argument out_dir, returns a decorator that will log results to
       out_dir, logging to a temporary directory during execution. Handles
       node failures and caching.

       Specifically, the decorated function must take a parameter log_dir.
       The decorator intercepts the log_dir provided by the callee,
       and creates a temporary directory on the local machine.

       If the task is successful, this temporary directory is copied to out_dir,
       with a unique object id. A symlink to this directory is then made from
       the callee-specified log_dir.

       No entry is made when the task is unsuccessful.

       If it encounters an error that appears to be related to a node failure
       (inability to access out_dir), it will kill the worker, forcing Ray
       to retry the task. (These semantics aren't ideal).

       Note this should be applied to the function(s) closest to the point
       where logging output is actually produced. In particular, do not apply
       it to two functions that receive the same log_dir!'''
    def make_decorator(*oargs, **okwargs):
        def decorator(func):
            @functools.wraps(func)
            def pre_cache_wrapper(*args, **kwargs):
                '''Creates a temporary directory tmp_dir for logging,
                   and calls func(*args, **kwargs, log_dir=tmp_dir).
                   Upon completion of the function, it copies tmp_dir over to
                   a newly created directory in out_dir.
                   The main purpose of this is to isolate errors in the function
                   from errors in accessing out_dir.'''
                with tempfile.TemporaryDirectory(prefix='pirl') as tmp_dir:
                    # Run the function
                    res = func(*args, **kwargs, log_dir=tmp_dir)

                    # Success! (If an exception happens, we never reach here)
                    # Copy results to a new directory in out_dir
                    try:
                        os.makedirs(out_dir, exist_ok=True)
                        permanent_dir = tempfile.mkdtemp(dir=out_dir)
                        os.chmod(permanent_dir, 0o755)
                        copy_tree(tmp_dir, permanent_dir)
                    except OSError as e:
                        # Copying could fail for two reasons.
                        # (1) Node failure -- either it is being preempted,
                        # or has some other error (e.g. network failure).
                        # (2) out_dir is wrong (e.g. permissions errors).
                        # We're going to pretend it's always (1), but complain
                        # loudly in the logs in case it's (2).
                        _temporary_error(e)

                # Append permanent_dir to return value, so caller knows where
                # to look for results.
                return res, permanent_dir

            cached_fn = cache(*oargs, **okwargs)(pre_cache_wrapper)

            @functools.wraps(cached_fn)
            def post_cache_wrapper(*args, **kwargs):
                '''Calls cached_fn(*args, **kwargs_exc) where kwargs_exc has
                   had log_dir removed from it. It adds a symlink at log_dir
                   pointing to the log directory returned by cached_fn, and
                   returns the result returned originally by func.'''
                # Inspection & argument extraction
                signature = inspect.signature(func)
                bound = signature.bind(*args, **kwargs)
                arguments = bound.arguments

                # sym_fname is the user-requested log directory.
                # However, we log to a temporary directory, only making a
                # symbolic link to the temporary directory once finished.
                ultimate_log_dir = arguments.pop('log_dir')
                sym_fname = os.path.abspath(ultimate_log_dir)
                # Catch common misuse of this decorator
                if sym_fname in log_dirs:
                    msg = "Duplicate log directory '{}'".format(sym_fname)
                    raise AssertionError(msg)
                log_dirs.add(sym_fname)

                res, permanent_log_dir = cached_fn(*bound.args, **bound.kwargs)

                try:
                    os.makedirs(os.path.dirname(sym_fname), exist_ok=True)
                    os.symlink(permanent_log_dir, sym_fname,
                               target_is_directory=True)
                except FileExistsError:
                    logger.warning('Destination %s already exists (attempt to '
                                   'link to %s). Did we retry a successful task?',
                                   sym_fname, permanent_log_dir)
                except OSError as e:
                    _temporary_error(e)

                return res

            return post_cache_wrapper
        return decorator
    return make_decorator

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

def autodetect_num_gpus():
    """Attempt to detect the number of GPUs on this machine."""
    # Based on code from Ray
    if 'NUM_GPUS' in os.environ:
        return os.environ['NUM_GPUS']
    else:
        proc_gpus_path = "/proc/driver/nvidia/gpus"
        if os.path.isdir(proc_gpus_path):
            return len(os.listdir(proc_gpus_path))
        return 0

def set_cuda_visible_devices():
    if ray.worker.global_worker.mode == ray.worker.PYTHON_MODE:
        # Debug mode. get_gpu_ids() not supported -- do nothing.
        return
    ids = ray.get_gpu_ids()
    real_num_gpus = ray.services._autodetect_num_gpus()
    gpus = ','.join([str(ids[i % real_num_gpus]) for i in range(len(ids))])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
