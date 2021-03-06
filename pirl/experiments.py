import collections
from contextlib import contextmanager
import functools
import itertools
import inspect
import logging
import os
import os.path as osp

from baselines import bench
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
import gym
import joblib
import ray

from pirl import config, utils
from pirl.utils import create_seed, sanitize_env_name, safeset

logger = logging.getLogger('pirl.experiments.experiments')
cache = utils.cache_and_log(config.OBJECT_DIR)

# Context Managers & Decorators

@contextmanager
def make_envs(env_name, vectorized, parallel, base_seed, log_prefix,
              pre_wrapper=None, post_wrapper=None):
    def helper(i):
        env = gym.make(env_name)
        env = bench.Monitor(env, log_prefix + str(i), allow_early_resets=True)
        if pre_wrapper is not None:
            env = pre_wrapper(env)
        env.seed(base_seed + i)
        return env

    env = None
    try:
        if vectorized:
            env_fns = [functools.partial(helper, i) for i in range(parallel)]
            #SOMEDAY: use SubprocVecEnv when parallel above a certain threshold.
            # Avoided this since (a) it's not much faster, and (b) it fails
            # on some machines for MuJoCo environments. (Never pinpointed
            # exact issue -- import fails in spawned worker, but only after
            # TensorFlow code has run, intermittent issue. Suspect GPU drivers.)
            # If you fix this, change num_cpus in ray_remote_variable_resources.
            env = DummyVecEnv(env_fns)
        else:  # not vectorized
            env = helper(0)

        if post_wrapper is not None:
            # note post_wrapper may be called with Env or VecEnv
            env = post_wrapper(env)

        yield env
    finally:
        if env is not None:
            env.close()


def ray_remote_variable_resources(**kwargs):
    '''A decorator that mimics ray.remote(**kwargs), but creates several remote
       functions varying in their declared resource requirements.
       When the underlying function is called, the arguments are inspected
       to determine the resource requirements, and the relevant cached remote
       is dispatched.

       Note this has the advantage that all the cached tasks are created on
       the driver, and therefore visible on all workers.

       The main downside is it multiplies the number of registered functions
       substantially. This isn't a problem in our application.'''
    def decorator(func):
        parameter_set = collections.OrderedDict([
            ('num_cpus', [1]),
            ('num_gpus', [0,1]),
        ])
        cache = {}
        for vs in itertools.product(*parameter_set.values()):
            # Name mangling to make function ID unique
            parameters = [(k, v) for k, v in zip(parameter_set.keys(), vs)]
            suffix = ','.join(['{}={}'.format(k, v) for k, v in parameters])
            name = func.__name__
            func.__name__ = '{}:{}'.format(name, suffix)
            # Specify max_calls=1 to force GPU memory to be released.
            # This shouldn't be necessary, but the overhead of using a fresh
            # worker each time is minimal as our tasks are long-lived.
            # Without this I've found TensorFlow initialization hangs
            # indefinitely sometimes...
            cache[tuple(vs)] = ray.remote(max_calls=1,
                                          **dict(parameters),
                                          **kwargs)(func)
            func.__name__ = name

        def func_call(*args, **kwargs):
            signature = inspect.signature(func)
            bound = signature.bind(*args, **kwargs)
            arguments = bound.arguments
            rl = arguments.get('rl')
            irl = arguments.get('irl')

            uses_gpu = False
            if rl is not None:
                algo = config.RL_ALGORITHMS[rl]
                uses_gpu |= algo.uses_gpu
            elif irl is not None:
                algo = (config.SINGLE_IRL_ALGORITHMS.get(irl) or
                        config.POPULATION_IRL_ALGORITHMS.get(irl))
                uses_gpu |= algo.uses_gpu
            else:
                raise ValueError("No 'rl' or 'irl' parameters")
            num_gpus = int(uses_gpu)
            # NOTE: If we switch from DummyVecEnv to SubprocVecEnv,
            # should make num_cpus depend on parallel (with some fudge factor)
            num_cpus = 1
            try:
                fn = cache[(num_cpus, num_gpus)]
            except KeyError:
                raise KeyError('Did not expect CPU/GPU combination {}/{}. '
                               'If valid, then update parameter_set.'.format(
                                num_cpus, num_gpus))
            return fn.remote(*args, **kwargs)

        @functools.wraps(func)
        def func_invoker(*args, **kwargs):
            raise Exception("Remote functions cannot be called directly.")
        func_invoker.remote = func_call

        return func_invoker
    return decorator

## Trajectory generation

@ray_remote_variable_resources()
@cache(tags=('expert', ))
def _train_policy(rl, discount, parallel, seed, env_name, log_dir):
    # Setup
    utils.set_cuda_visible_devices()
    logger.debug('[TRAIN] %s [discount=%f, seed=%s, parallel=%d] on %s',
                 rl, discount, seed, parallel, env_name)
    mon_dir = osp.join(log_dir, 'mon')
    os.makedirs(mon_dir, exist_ok=True)
    train_seed = create_seed(seed + 'train')

    # Generate the policy
    rl_algo = config.RL_ALGORITHMS[rl]
    with make_envs(env_name, rl_algo.vectorized, parallel, train_seed,
                   log_prefix=osp.join(mon_dir, 'train')) as envs:
        # This nested parallelism is unfortunate. We're mostly doing this
        # as algorithms differ in their resource reservation.
        policy = rl_algo.train(envs, discount=discount, seed=train_seed,
                               log_dir=log_dir)

    joblib.dump(policy, osp.join(log_dir, 'policy.pkl'))  # save for debugging

    return policy

@ray_remote_variable_resources()
@cache(tags=('expert', ))
def synthetic_data(rl, discount, parallel, seed, env_name, num_trajectories,
                   log_dir, policy):
    '''Precondition: policy produced by RL algorithm rl.'''
    # Setup
    utils.set_cuda_visible_devices()
    logger.debug('[SAMPLE] %s [discount=%f, seed=%s, parallel=%d] '
                 'for %d trajectories on %s',
                 rl, discount, seed, parallel, num_trajectories, env_name)

    # Create and set up logging directory
    mon_dir = osp.join(log_dir, 'mon')
    os.makedirs(mon_dir, exist_ok=True)

    # Sample from policy
    data_seed = create_seed(seed + 'data')
    rl_algo = config.RL_ALGORITHMS[rl]
    with make_envs(env_name, rl_algo.vectorized, parallel, data_seed,
                   log_prefix=osp.join(mon_dir, 'synthetic')) as envs:
        samples = rl_algo.sample(envs, policy, num_trajectories, data_seed)
    return [(obs, acts) for (obs, acts, rews) in samples]

@ray_remote_variable_resources()
@cache(tags=('expert', ))
def _compute_value(rl, discount, parallel, seed, env_name, log_dir, policy):
    utils.set_cuda_visible_devices()
    # Note discount is not used, but is needed as a caching key.
    logger.debug('[VALUE] %s [discount=%f, seed=%s, parallel=%d] on %s',
                 rl, discount, seed, parallel, env_name)
    # Each RL algorithm specifies a method to compute the value of its policy
    rl_algo = config.RL_ALGORITHMS[rl]

    # Create and set up logging directory
    mon_dir = osp.join(log_dir, 'mon')
    os.makedirs(mon_dir, exist_ok=True)

    # Compute value of policy
    eval_seed = create_seed(seed + 'eval')
    with make_envs(env_name, rl_algo.vectorized, parallel, eval_seed,
                   log_prefix=osp.join(mon_dir, 'eval')) as envs:
        value = rl_algo.value(envs, policy, discount=1.00, seed=eval_seed)

    return value


def _expert_trajs(env_name, num_trajectories, rl, discount,
                  parallel, seed, log_dir):
    '''Trains a policy on env_name with rl_name, sampling num_trajectories from
       the policy and computing the value of the policy (typically by sampling,
       but in the tabular case by value iteration).

       Returns a pair of ray object IDs, for the trajectories and value.'''
    #TODO: use different log_dirs for these??
    # Set up logging
    log_dir = osp.join(log_dir, sanitize_env_name(env_name), rl)
    # Train the policy on the environment
    policy_future = _train_policy.remote(rl, discount, parallel, seed,
                                         env_name, osp.join(log_dir, 'train'))
    # Compute the expected value & standard error of the policy
    value_future = _compute_value.remote(rl, discount, parallel, seed,
                                         env_name, osp.join(log_dir, 'value'),
                                         policy_future)
    # Sample from the policy to get expert trajectories
    trajs_future = synthetic_data.remote(rl, discount, parallel, seed, env_name,
                                         num_trajectories,
                                         osp.join(log_dir, 'sample'),
                                         policy_future)
    # Return promises
    return trajs_future, value_future


def expert_trajs(cfg, out_dir, seed):
    log_dir = osp.join(out_dir, 'expert')
    parallel = cfg.get('parallel_rollouts', 1)

    # Construct (env, traj) pairs based on the trajectories and environment configuration
    num_traj = collections.OrderedDict()
    for k in ['train', 'test']:
        trajs = cfg.get('{}_trajectories'.format(k))
        if trajs is not None:
            n = max(trajs)
            envs = cfg.get('{}_environments'.format(k), cfg.get('environments'))
            for env in envs:
                num_traj[env] = max(n, num_traj.get(env, 0))

    # Get futures for trajectories and computed values for each environment
    trajectories = collections.OrderedDict()
    values = collections.OrderedDict()
    for env, traj in num_traj.items():
        t, v = _expert_trajs(env, traj, cfg['expert'], cfg['discount'],
                             parallel, seed, log_dir)
        trajectories[env] = t
        values[env] = v

    return trajectories, values

### IRL

## Population/meta IRL

@ray_remote_variable_resources()
@cache(tags=('irl', 'population_irl'))
def _run_population_irl_meta(irl, parallel, discount, seed, trajs, log_dir):
    # Setup
    utils.set_cuda_visible_devices()
    n = len(list(trajs.values())[0])
    logger.debug('[IRL] meta: algo = %s [discount=%f, seed=%s, parallel=%d], ' 
                 'n = %d, envs = %s',
                 irl, discount, seed, parallel, n, trajs.keys())
    meta_log_dir = osp.join(log_dir, 'meta:{}'.format(n))
    mon_dir = osp.join(meta_log_dir, 'mon')
    os.makedirs(mon_dir)

    # Get algorithm from config
    irl_algo = config.POPULATION_IRL_ALGORITHMS[irl]
    # Customize seeds
    irl_seed = create_seed(seed + 'irlmeta')

    # Set up environments for meta-learning
    ctxs = {}
    for env in trajs.keys():
        log_prefix = osp.join(mon_dir, sanitize_env_name(env) + '-')
        ctxs[env] = make_envs(env, irl_algo.vectorized, parallel,
                              irl_seed, log_prefix=log_prefix)
    meta_envs = {k: v.__enter__() for k, v in ctxs.items()}

    # Run metalearning
    subset = {k: v[:n] for k, v in trajs.items()}
    metainit = irl_algo.metalearn(meta_envs, subset, discount=discount,
                                  seed=irl_seed, log_dir=meta_log_dir)

    # Make sure to exit out of all the environments
    for env in ctxs.values():
        env.__exit__(None, None, None)

    # Save metalearning initialization for debugging
    joblib.dump(metainit, osp.join(log_dir, 'metainit.pkl'))

    return metainit


@ray_remote_variable_resources(num_return_vals=2)
@cache(tags=('irl', 'population_irl'))
def _run_population_irl_finetune(irl, parallel, discount, seed,
                                 env, trajs, metainit, log_dir):
    # Setup
    utils.set_cuda_visible_devices()
    logger.debug('[IRL] finetune: algo = %s [discount=%f, seed=%s, parallel=%d]' 
                 ', m = %d, env = %s',
                 irl, discount, seed, parallel, len(trajs), env)
    finetune_mon_prefix = osp.join(log_dir, 'mon')

    # Get algorithm from config
    irl_algo = config.POPULATION_IRL_ALGORITHMS[irl]
    # Seeding
    finetune_seed = create_seed(seed + 'irlfinetune')

    # Finetune IRL algorithm (i.e. run it) from meta-initialization
    with make_envs(env, irl_algo.vectorized, parallel,
                   finetune_seed,
                   log_prefix=finetune_mon_prefix) as envs:
        r, p = irl_algo.finetune(metainit, envs, trajs, discount=discount,
                                 seed=finetune_seed, log_dir=log_dir)
        joblib.dump(p, osp.join(log_dir, 'policy.pkl'))

    # Compute value of finetuned policy
    with make_envs(env, irl_algo.vectorized, parallel,
                   finetune_seed,
                   log_prefix=finetune_mon_prefix) as envs:
        eval_seed = create_seed(seed + 'eval')
        v = irl_algo.value(envs, p, discount=1.0, seed=eval_seed)

    return r, v


def _run_population_irl_train(irl, parallel, discount, seed,
                              train_trajs, test_trajs, n, ms, log_dir):
    '''Performs metalearning with irl_name on n training trajectories,
       returning a tuple of rewards and values with shape [env][m].'''
    # Metalearn
    meta_log_dir = osp.join(log_dir, 'irl', irl, 'meta:{}'.format(n))
    meta_subset = {k: v[:n] for k, v in train_trajs.items()}
    metainit = _run_population_irl_meta.remote(irl, parallel, discount,
                                               seed, meta_subset, meta_log_dir)

    # Finetune
    rewards = collections.OrderedDict()
    values = collections.OrderedDict()
    for env, trajs in test_trajs.items():
        for m in ms:
            subset = trajs[:m]
            finetune_log_dir = osp.join(meta_log_dir, 'finetune:{}'.format(m),
                                        sanitize_env_name(env))
            r, v = _run_population_irl_finetune.remote(irl, parallel, discount,
                                                       seed, env, subset,
                                                       metainit,
                                                       finetune_log_dir)
            safeset(rewards, [env, m], r)
            safeset(values, [env, m], v)

    return rewards, values


@ray.remote(num_return_vals=2)
def _run_population_irl_helper(irl, parallel, discount, seed,
                               train_envs, test_envs, num_traj,
                               log_dir, envs, *trajectories):
    # Reconstruct trajectories
    trajectories = {k: v for k, v in zip(envs, trajectories)}
    train_trajs = {k: trajectories[k] for k in train_envs}
    test_trajs = {k: trajectories[k] for k in test_envs}

    rewards = collections.OrderedDict()
    values = collections.OrderedDict()

    for n, ms in num_traj.items():
        r, v = _run_population_irl_train(irl, parallel, discount, seed,
                                         train_trajs, test_trajs, n, ms,
                                         log_dir)
        for env in test_trajs.keys():
            safeset(rewards, [env, n], r[env])
            safeset(values, [env, n], v[env])
    # {rewards,value} [env][n][m] -> {reward,value}
    rewards = utils.ray_get_nested_dict(rewards, level=3)
    values = utils.ray_get_nested_dict(values, level=3)
    return rewards, values


def _run_population_irl(irl, parallel, discount, seed, train_envs,
                        test_envs, num_traj, trajectories, out_dir):
    # Flatten trajectories (env -> object ID) to appease ray
    envs = sorted(list(set(train_envs).union(test_envs)))
    trajectories = [trajectories[k] for k in envs]
    return _run_population_irl_helper.remote(irl, parallel, discount, seed,
                                             train_envs, test_envs, num_traj,
                                             out_dir, envs, *trajectories)

## Single-task IRL

@ray_remote_variable_resources(num_return_vals=2)
@cache(tags=('irl', 'single_irl'))
def _run_single_irl_train(irl, parallel, discount, seed,
                          env_name, log_dir, trajectories):
    logger.debug('[IRL] algo = %s [discount=%f, seed=%s, parallel=%d], ' 
                 'env = %s, n = %d',
                 irl, discount, seed, parallel, env_name, len(trajectories))
    # Setup
    utils.set_cuda_visible_devices()
    mon_dir = osp.join(log_dir, 'mon')
    os.makedirs(mon_dir)

    irl_algo = config.SINGLE_IRL_ALGORITHMS[irl]
    irl_seed = create_seed(seed + 'irl')
    with make_envs(env_name, irl_algo.vectorized, parallel, irl_seed,
                   log_prefix=osp.join(mon_dir, 'train')) as envs:
        reward, policy = irl_algo.train(envs, trajectories, discount=discount,
                                        seed=irl_seed, log_dir=log_dir)

    # Save learnt reward & policy for debugging purposes
    joblib.dump(reward, osp.join(log_dir, 'reward.pkl'))
    joblib.dump(policy, osp.join(log_dir, 'policy.pkl'))

    eval_seed = create_seed(seed + 'eval')
    with make_envs(env_name, irl_algo.vectorized, parallel, eval_seed,
                   log_prefix=osp.join(mon_dir, 'eval')) as envs:
        value = irl_algo.value(envs, policy, discount=1.00, seed=eval_seed)

    return reward, value


@ray.remote(num_return_vals=2)
def _run_single_irl_helper(irl, parallel, discount, seed,
                           num_traj, test_envs, log_dir, *trajectories):
    trajectories = {k: v for k, v in zip(test_envs, trajectories)}
    reward_res = collections.OrderedDict()
    value_res = collections.OrderedDict()

    ms = sorted(set(itertools.chain(*num_traj.values())))
    for env, m in itertools.product(test_envs, ms):
        subset = trajectories[env][:m]
        sub_log_dir = osp.join(log_dir, 'irl', irl,
                               sanitize_env_name(env), '{}'.format(m))

        reward, value = _run_single_irl_train.remote(irl, parallel, discount,
                                                     seed, env, sub_log_dir,
                                                     subset)

        for n, ms in num_traj.items():
            if m in ms:
                #SOMEDAY: although this won't duplicate the work,
                #callers might not know the values are identical, and so
                #some work might get duplicated later in the pipeline
                #(e.g. when reoptimizing in value()).
                safeset(reward_res, [env, n, m], reward)
                safeset(value_res, [env, n, m], value)

    # Returns two dictionaries of the form [env][n][m]
    reward_res = utils.ray_get_nested_dict(reward_res, level=3)
    value_res = utils.ray_get_nested_dict(value_res, level=3)
    return reward_res, value_res

def _run_single_irl(irl, num_traj, train_envs, test_envs, parallel,
                    discount, seed, out_dir, trajectories):
    trajectories = [trajectories[k] for k in test_envs]
    return _run_single_irl_helper.remote(irl, parallel, discount, seed,
                                         num_traj, test_envs, out_dir,
                                         *trajectories)

## General IRL

def run_irl(cfg, out_dir, trajectories, seed):
    '''Run experiment in parallel. Returns tuple (reward, value) where each are
       nested OrderedDicts, with key in the format:
        - IRL algo
        - Number of trajectories for other environments
        - Number of trajectories for this environment
        - Environment
       Note that for this experiment type, the second and third arguments are
       always the same.
    '''

    num_traj = collections.OrderedDict()
    if 'train_trajectories' in cfg:  # meta-learning experiment
        for n in cfg['train_trajectories']:
            num_traj[n] = [m for m in cfg['test_trajectories'] if m <= n]
    else:  # no meta-learning
        num_traj[0] = cfg['test_trajectories']

    test_envs = cfg.get('test_environments', cfg.get('environments'))
    train_envs = cfg.get('train_environments', cfg.get('environments'))

    kwargs = {
        'out_dir': out_dir,
        'parallel': cfg.get('parallel_rollouts', 1),
        'discount': cfg['discount'],
        'seed': seed,
        'num_traj': num_traj,
        'train_envs': train_envs,
        'test_envs': test_envs,
        'trajectories': trajectories,
    }

    # Futures shape: irl -> Future([env][n][m])
    reward_futures = collections.OrderedDict()
    value_futures = collections.OrderedDict()
    for irl in cfg['irl']:
        kwds = dict(kwargs)
        kwds.update({'irl': irl})
        if irl in config.SINGLE_IRL_ALGORITHMS:
            rew, val = _run_single_irl(**kwds)
        elif irl in config.POPULATION_IRL_ALGORITHMS:
            rew, val = _run_population_irl(**kwds)
        else:
            assert False  # illegal config
        safeset(reward_futures, [irl], rew)
        safeset(value_futures, [irl], val)

    return reward_futures, value_futures

## Evaluation

#TODO: this actually requires twice as many GPU resources as other tasks
#(for reward wrapper and for the RL policy network).
#No good way to express this in current framework.
@ray_remote_variable_resources()
@cache(tags=('eval', ))
def _value_helper(irl, n, m, rl, parallel, discount, seed, env_name, 
                  reward, log_dir):
    if reward is None:
        # reward will be None if the algorithm is a non-IRL imitation learner.
        # In this case, do not attempt to reoptimize.
        return None

    # Setup
    utils.set_cuda_visible_devices()
    logger.debug('[EVAL] %s [meta=%d, finetune=%d] ' 
                 'by %s [discount=%f, seed=%s, parallel=%d] '
                 'on %s (writing to %s)',
                 irl, n, m,
                 rl, discount, seed, parallel,
                 env_name, log_dir)
    mon_dir = osp.join(log_dir, 'mon')
    os.makedirs(mon_dir)

    if irl in config.SINGLE_IRL_ALGORITHMS:
        reward_wrapper = config.SINGLE_IRL_ALGORITHMS[irl].reward_wrapper
    else:
        reward_wrapper = config.POPULATION_IRL_ALGORITHMS[irl].reward_wrapper
    rw = functools.partial(reward_wrapper, new_reward=reward)
    rl_algo = config.RL_ALGORITHMS[rl]

    train_seed = create_seed(seed + 'eval_train')
    with make_envs(env_name, rl_algo.vectorized, parallel,
                   train_seed, post_wrapper=rw,
                   log_prefix=osp.join(mon_dir, 'train')) as envs:
        p = rl_algo.train(envs, discount=discount,
                          seed=train_seed, log_dir=log_dir)

    joblib.dump(p, osp.join(log_dir, 'policy.pkl'))

    eval_seed = create_seed(seed + 'eval_eval')
    with make_envs(env_name, rl_algo.vectorized, parallel,
                   eval_seed, log_prefix=osp.join(mon_dir, 'eval')) as envs:
        v = rl_algo.value(envs, p, discount=1.00, seed=eval_seed)

    return v

@ray.remote
def _value(irl, rl, parallel, out_dir, reward, discount, seed):
    def mapper(rew, keys):
        env_name, n, m = keys
        log_dir = osp.join(out_dir, 'eval', sanitize_env_name(env_name),
                           '{}:{}:{}'.format(irl, m, n), rl)
        kwargs = {
            'irl': irl,
            'n': n,
            'm': m,
            'rl': rl,
            'parallel': parallel,
            'discount': discount,
            'seed': seed,
            'env_name': env_name,
            'log_dir': log_dir,
            'reward': rew,
        }
        return _value_helper.remote(**kwargs)
    #SOMEDAY: ray.get inside ray.remote is legal but feels yucky
    value_futures = utils.map_nested_dict(reward, mapper, level=3)
    return utils.ray_get_nested_dict(value_futures, level=3)

def value(cfg, out_dir, rewards, seed):
    '''
    Compute the expected value of (a) policies optimized on inferred reward,
    and (b) optimal policies for the ground truth reward. Policies will be
    computed using each RL algorithm specified in cfg['eval'].

    Args:
        - cfg: config.EXPERIMENTS[experiment]
        - out_dir: for logging
        - rewards
        - seed
    Returns:
        tuple, (value, ground_truth) where each is a nested dictionary of the
        same shape as rewards, with the leaf being a dictionary mapping from
        an RL algorithm in cfg['eval'] to a scalar value.
    '''
    discount = cfg['discount']
    parallel = cfg.get('parallel_rollouts', 1)

    def reward_map(rew, keys, rl):
        irl = keys[0]
        kwargs = {
            'irl': irl,
            'rl': rl,
            'parallel': parallel,
            'out_dir': out_dir,
            'reward': rew,
            'discount': discount,
            'seed': seed
        }
        return _value.remote(**kwargs)

    # rewards -> value_futures
    # rewards: [irl_name] -> Future[[env][n][m] -> reward map]
    # value_futures: [rl][irl_name] -> Future[[env][n][m] -> (mean, se)]
    value_futures = collections.OrderedDict()
    for rl in cfg['eval']:
        value_futures[rl] = utils.map_nested_dict(rewards,
                                            functools.partial(reward_map, rl=rl))

    # ground_truth_futures: [rl][env] -> (mean, se)
    #TODO: This is often duplicating the work of expert_trajs.
    #It also feels conceptually confused -- we should probably be computing this in a separate function.
    ground_truth_futures = collections.OrderedDict()
    for rl in cfg['eval']:
        for env_name in cfg.get('test_environments', cfg.get('environments')):
            log_dir = osp.join(out_dir, 'eval', sanitize_env_name(env_name),
                               'gt', rl)
            kwargs = {
                'rl': rl,
                'discount': discount,
                'env_name': env_name,
                'parallel': parallel,
                'seed': seed,
            }
            pol = _train_policy.remote(log_dir=osp.join(log_dir, 'train'),
                                       **kwargs)
            val = _compute_value.remote(policy=pol,
                                        log_dir=osp.join(log_dir, 'eval'),
                                        **kwargs)
            safeset(ground_truth_futures, [rl, env_name], val)

    return value_futures, ground_truth_futures

## General

def _run_experiment(cfg, out_dir, seed):
    # Generate synthetic data
    # trajs: dict, env -> Future[list of np arrays]
    # expert_vals: dict, env -> Future[(mean, s.e.)]
    trajs, expert_vals = expert_trajs(cfg, out_dir, seed)
    # Run IRL
    # rewards: dict, irl -> Future[env -> n -> m -> reward]
    # irl_values: dict, irl -> Future[env -> n -> m -> (mean, s.e.)]
    rewards, irl_values = run_irl(cfg, out_dir, trajs, seed)
    # Run RL with the reward predicted by IRL ("reoptimize")
    # values: dict, rl -> irl -> Future[env -> n -> m -> (mean, se)]
    # ground_truth: dict, rl -> env -> Future[(mean, se)]
    values, ground_truth = value(cfg, out_dir, rewards, seed)

    # Add in the values obtained by the expert & IRL policies
    ground_truth['expert'] = expert_vals
    values['irl'] = irl_values

    res = {
        'trajectories': trajs,
        'rewards': rewards,
        'values': values,
        'ground_truth': ground_truth
    }
    return res


def run_experiment(cfg, out_dir, base_seed):
    '''Run experiment defined in config.EXPERIMENTS.

    Args:
        - experiment(str): experiment name.
        - out_dir(str): path to write logs and results to.
        - video_every(optional[int]): if None, do not record video.
        - base_seed(int)

    Returns:
        dict with key-value pairs:

        - trajectories: synthetic data.
            dict, keyed by environments, with values generated by synthetic_data.
        - rewards: IRL inferred reward.
            nested dict, keyed by environment then IRL algorithm.
        - value: value obtained reoptimizing in the environment.
            Use the RL algorithm used to generate the original synthetic data
            to train a policy on the inferred reward, then compute expected
            discounted value obtained from the resulting policy.
        - ground_truth: value obtained from RL policy.
        - info: info dict from IRL algorithms.
        '''
    res = collections.defaultdict(collections.OrderedDict)
    for i in range(cfg['seeds']):
        log_dir = osp.join(out_dir, str(i))
        seed = base_seed + str(i)
        d = _run_experiment(cfg, log_dir, seed)
        for k, v in d.items():
            res[k][i] = v

    return utils.ray_leaf_get_nested_dict(res)
