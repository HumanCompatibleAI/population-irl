import collections
from contextlib import contextmanager
import functools
import itertools
import joblib
import logging
import os
import os.path as osp

from baselines import bench
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
import gym
from joblib import Memory

from pirl.experiments import config
from pirl.utils import create_seed, is_vectorized, log_errors, nested_async_get

logger = logging.getLogger('pirl.experiments.experiments')
memory = Memory(cachedir=config.CACHE_DIR, verbose=0)


def sanitize_env_name(env_name):
    return env_name.replace('/', '_')


@contextmanager
def _make_envs(env_name, vectorized, parallel, base_seed, log_prefix,
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
            if parallel and len(env_fns) > 1:
                env = SubprocVecEnv(env_fns)
            else:
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


def __train_policy(rl, discount, env_name, parallel, seed, log_dir):
    gen_policy, _sample, compute_value = config.RL_ALGORITHMS[rl]
    mon_dir = osp.join(log_dir, 'mon')
    os.makedirs(mon_dir, exist_ok=True)

    train_seed = create_seed(seed + 'train')
    with _make_envs(env_name, is_vectorized(gen_policy), parallel, train_seed,
                    log_prefix=osp.join(mon_dir, 'train')) as envs:
        p = gen_policy(envs, discount=discount, log_dir=log_dir)
    joblib.dump(p, osp.join(log_dir, 'policy.pkl'))  # save for debugging

    eval_seed = create_seed(seed + 'eval')
    with _make_envs(env_name, is_vectorized(compute_value), parallel, eval_seed,
                    log_prefix=osp.join(mon_dir, 'eval')) as envs:
        v = compute_value(envs, p, discount=1.00, seed=eval_seed)

    return p, v
# avoid name clash in pickling
_train_policy = memory.cache(ignore=['log_dir'])(__train_policy)


@memory.cache(ignore=['log_dir', 'video_every', 'policy'])
def synthetic_data(env_name, rl, num_trajectories, parallel, seed,
                   log_dir, video_every, policy):
    '''Precondition: policy produced by RL algorithm rl.'''
    _, sample, _ = config.RL_ALGORITHMS[rl]

    video_dir = osp.join(log_dir, 'videos')
    if video_every is None:
        video_callable = lambda x: False
    else:
        video_callable = lambda x: x % video_every == 0
    def  monitor(env):
        return gym.wrappers.Monitor(env, video_dir,
                                    video_callable=video_callable, force=True)

    mon_dir = osp.join(log_dir, 'mon')
    os.makedirs(mon_dir, exist_ok=True)

    data_seed = create_seed(seed + 'data')
    with _make_envs(env_name, is_vectorized(sample), parallel, data_seed,
                    log_prefix=osp.join(mon_dir, 'synthetic'),
                    pre_wrapper=monitor) as envs:
        samples = sample(envs, policy, num_trajectories, data_seed)
    return [(observations, actions) for (observations, actions, rewards) in samples]


@log_errors
def _expert_trajs(env_name, num_trajectories, experiment, rl_name, discount,
                  parallel, seed, video_every, log_dir):
    logger.debug('%s: training %s on %s', experiment, rl_name, env_name)
    log_dir = osp.join(log_dir, sanitize_env_name(env_name), rl_name)
    policy, value = _train_policy(rl_name, discount, env_name, parallel, seed, log_dir)

    logger.debug('%s: sampling from %s', experiment, env_name)
    _, sample, _ = config.RL_ALGORITHMS[rl_name]
    trajectories = synthetic_data(env_name, rl_name, num_trajectories, parallel,
                                  seed, log_dir, video_every, policy)

    return trajectories, value


def expert_trajs(experiment, out_dir, cfg, pool, video_every, seed):
    logger.debug('%s: generating synthetic data: training', experiment)
    log_dir = osp.join(out_dir, 'expert')
    os.makedirs(log_dir)
    parallel = cfg.get('parallel_rollouts', 1)

    num_traj = collections.OrderedDict()
    for k in ['train', 'test']:
        n = max(cfg.get('{}_trajectories'.format(k), cfg.get('trajectories')))
        envs = cfg.get('{}_environments'.format(k), cfg.get('environments'))
        for env in envs:
            num_traj[env] = max(n, num_traj.get(env, 0))

    f = functools.partial(_expert_trajs, experiment=experiment,
                          rl_name=cfg['expert'], discount=cfg['discount'],
                          parallel=parallel, seed=seed,
                          video_every=video_every, log_dir=log_dir)
    results = pool.starmap(f, num_traj.items(), chunksize=1)

    trajectories = collections.OrderedDict()
    values = collections.OrderedDict()
    for name, (traj, val) in zip(num_traj.keys(), results):
        trajectories[name] = traj
        values[name] = val

    return trajectories, values


@log_errors
def __run_population_irl(irl_name, train_envs, n, test_envs, ms, experiment,
                         out_dir, parallel, trajectories, discount, seed):
    logger.debug('%s: running IRL algo: %s (meta=%s / finetune=%s)',
                 experiment, irl_name, n, ms)
    log_dir = osp.join(out_dir, 'irl', irl_name)
    train, finetune, _rw, compute_value = config.POPULATION_IRL_ALGORITHMS[irl_name]

    irl_seed = create_seed(seed + 'irlmeta')
    meta_log_dir = osp.join(log_dir, 'meta:{}'.format(n))
    meta_mon_dir = osp.join(meta_log_dir, 'mon')
    os.makedirs(meta_mon_dir)
    meta_subset = {k: trajectories[k][:n] for k in train_envs}
    ctxs = {}
    for env in train_envs:
        log_prefix = osp.join(meta_mon_dir, sanitize_env_name(env) + '-')
        ctxs[env] = _make_envs(env, is_vectorized(train), parallel,
                               irl_seed, log_prefix=log_prefix)
    meta_envs = {k: v.__enter__() for k, v in ctxs.items()}
    metainit = train(meta_envs, meta_subset,
                     discount=discount, log_dir=meta_log_dir)
    for v in ctxs.values():
        v.__exit__(None, None, None)
    joblib.dump(metainit, osp.join(log_dir, 'metainit.pkl'))  # for debugging

    finetune_seed = create_seed(seed + 'irlfinetune')
    # SOMEDAY: parallize this step?
    rewards = collections.OrderedDict()
    policies = collections.OrderedDict()
    values = collections.OrderedDict()
    for m in ms:
        #TODO: parallelize
        for env in test_envs:
            finetune_log_dir = osp.join(meta_log_dir, 'finetune:{}'.format(m),
                                        sanitize_env_name(env))
            os.makedirs(finetune_log_dir)
            finetune_mon_prefix = osp.join(finetune_log_dir, 'mon')
            with _make_envs(env, is_vectorized(finetune), parallel,
                            finetune_seed, log_prefix=finetune_mon_prefix) as envs:
                finetune_subset = trajectories[env][:m]
                r, p = finetune(metainit, envs, finetune_subset,
                                discount=discount, log_dir=finetune_log_dir)
                setdef(rewards, m)[env] = r
                setdef(policies, m)[env] = p
            with _make_envs(env, is_vectorized(finetune), parallel,
                            finetune_seed, log_prefix=finetune_mon_prefix) as envs:
                eval_seed = create_seed(seed + 'eval')
                v = compute_value(envs, p, discount=1.0, seed=eval_seed)
                setdef(values, m)[env] = v

    # Save learnt reward & policy for debugging purposes
    joblib.dump(rewards, osp.join(log_dir, 'rewards.pkl'))
    joblib.dump(policies, osp.join(log_dir, 'policies.pkl'))

    return rewards, values
_run_population_irl = memory.cache(ignore=['out_dir'])(__run_population_irl)
#_run_population_irl = __run_population_irl


@log_errors
def __run_single_irl(irl_name, n, env_name, parallel,
                     experiment, out_dir, trajectories, discount, seed):
    logger.debug('%s: running IRL algo: %s [%s]', experiment, irl_name, n)
    irl_algo, _rw, compute_value = config.SINGLE_IRL_ALGORITHMS[irl_name]
    subset = trajectories[:n]
    log_dir = osp.join(out_dir, 'irl', irl_name,
                       sanitize_env_name(env_name), '{}'.format(n))
    mon_dir = osp.join(log_dir, 'mon')
    os.makedirs(mon_dir)

    irl_seed = create_seed(seed + 'irl')
    with _make_envs(env_name, is_vectorized(irl_algo), parallel, irl_seed,
                    log_prefix=osp.join(mon_dir, 'train')) as envs:
        reward, policy = irl_algo(envs, subset, discount=discount, log_dir=log_dir)

    # Save learnt reward & policy for debugging purposes
    joblib.dump(reward, osp.join(log_dir, 'reward.pkl'))
    joblib.dump(policy, osp.join(log_dir, 'policy.pkl'))

    eval_seed = create_seed(seed + 'eval')
    with _make_envs(env_name, is_vectorized(compute_value), parallel, eval_seed,
                    log_prefix=osp.join(mon_dir, 'eval')) as envs:
        value = compute_value(envs, policy, discount=1.00, seed=eval_seed)

    return reward, value
_run_single_irl = memory.cache(ignore=['out_dir'])(__run_single_irl)
#_run_single_irl = __run_single_irl


def setdef(d, k):
    return d.setdefault(k, collections.OrderedDict())


def run_irl(experiment, out_dir, cfg, pool, trajectories, seed):
    '''Run experiment in parallel. Returns tuple (reward, value) where each are
       nested OrderedDicts, with key in the format:
        - IRL algo
        - Number of trajectories for other environments
        - Number of trajectories for this environment
        - Environment
       Note that for this experiment type, the second and third arguments are
       always the same.
    '''
    kwargs = {
        'experiment': experiment,
        'out_dir': out_dir,
        'parallel': cfg.get('parallel_rollouts', 1),
        'discount': cfg['discount'],
        'seed': seed,
    }

    if 'trajectories' in cfg:
        sin_traj = cfg['trajectories']
        pop_traj = collections.OrderedDict([(n, [n]) for n in cfg['trajectories']])
    else:
        sin_traj = cfg['test_trajectories']
        pop_traj = collections.OrderedDict()
        for n in cfg['train_trajectories']:
            pop_traj[n] = [m for m in cfg['test_trajectories'] if m <= n]
    test_envs = cfg.get('test_environments', cfg.get('environments'))
    sin_res = {}
    for irl_name, m, env in itertools.product(cfg['irl'], sin_traj, test_envs):
        if irl_name in config.SINGLE_IRL_ALGORITHMS:
            if m == 0:
                continue
            kwds = dict(kwargs)
            kwds.update({
                'irl_name': irl_name,
                'n': m,
                'env_name': env,
                'trajectories': trajectories[env],
            })
            delayed = pool.apply_async(_run_single_irl, kwds=kwds)
            setdef(setdef(sin_res, irl_name), m)[env] = delayed

    pop_res = {}
    train_envs = cfg.get('train_environments', cfg.get('environments'))
    for irl_name, n in itertools.product(cfg['irl'], pop_traj.keys()):
        if irl_name in config.POPULATION_IRL_ALGORITHMS:
            kwds = dict(kwargs)
            kwds.update({
                'irl_name': irl_name,
                'n': n,
                'ms': pop_traj[n],
                'train_envs': train_envs,
                'test_envs': test_envs,
                'trajectories': trajectories,
            })
            delayed = pool.apply_async(_run_population_irl, kwds=kwds)
            setdef(pop_res, irl_name)[n] = delayed

    rewards = collections.OrderedDict()
    values = collections.OrderedDict()

    sin_res = nested_async_get(sin_res)
    for irl_name, d in sin_res.items():
        for n, ms in pop_traj.items():
            for m in ms:
                if m == 0:
                    continue
                d2 = d[m]
                for env, (r, v) in d2.items():
                    setdef(setdef(setdef(rewards, irl_name), n), m)[env] = r
                    setdef(setdef(setdef(values, irl_name), n), m)[env] = v

    pop_res = nested_async_get(pop_res)
    for irl_name, d in pop_res.items():
        for n, (dr, dv) in d.items():
            setdef(rewards, irl_name)[n] = dr
            setdef(values, irl_name)[n] = dv

    return rewards, values


@log_errors
def _value(experiment, irl_name, rl_name, env_name, parallel,
           log_dir, reward, discount, seed):
    logger.debug('%s: evaluating %s on %s (writing to %s)',
                 experiment, irl_name, env_name, log_dir)
    gen_policy, _sample, compute_value = config.RL_ALGORITHMS[rl_name]

    if irl_name in config.SINGLE_IRL_ALGORITHMS:
        reward_wrapper = config.SINGLE_IRL_ALGORITHMS[irl_name][1]
    else:
        reward_wrapper = config.POPULATION_IRL_ALGORITHMS[irl_name][2]
    rw = functools.partial(reward_wrapper, new_reward=reward)

    mon_dir = osp.join(log_dir, 'mon')
    os.makedirs(mon_dir)

    train_seed = create_seed(seed + 'eval_train')
    with _make_envs(env_name, is_vectorized(gen_policy), parallel, train_seed,
                    post_wrapper=rw, log_prefix=osp.join(mon_dir, 'train')) as envs:
        p = gen_policy(envs, discount=discount, log_dir=log_dir)

    logger.debug('%s: reoptimized %s on %s, sampling to estimate value',
                 experiment, irl_name, env_name)
    eval_seed = create_seed(seed + 'eval_eval')
    with _make_envs(env_name, is_vectorized(compute_value), parallel, eval_seed,
                    log_prefix=osp.join(mon_dir, 'eval')) as envs:
        v = compute_value(envs, p, discount=1.00, seed=eval_seed)

    return v


def value(experiment, out_dir, cfg, pool, rewards, seed):
    '''
    Compute the expected value of (a) policies optimized on inferred reward,
    and (b) optimal policies for the ground truth reward. Policies will be
    computed using each RL algorithm specified in cfg['eval'].

    Args:
        - experiment
        - cfg: config.EXPERIMENTS[experiment]
        - rewards
    Returns:
        tuple, (value, ground_truth) where each is a nested dictionary of the
        same shape as rewards, with the leaf being a dictionary mapping from
        an RL algorithm in cfg['eval'] to a scalar value.
    '''
    discount = cfg['discount']
    parallel = cfg.get('parallel_rollouts', 1)
    value = collections.OrderedDict()
    ground_truth = {}
    for rl in cfg['eval']:
        for irl_name, reward_by_size in rewards.items():
            res_by_n = collections.OrderedDict()
            for n, reward_by_small_size in reward_by_size.items():
                res_by_m = collections.OrderedDict()
                for m, reward_by_env in reward_by_small_size.items():
                    res_by_env = {}
                    for env_name,r  in reward_by_env.items():
                        log_dir = osp.join(out_dir, 'eval',
                                           sanitize_env_name(env_name),
                                           '{}:{}:{}'.format(irl_name, m, n),
                                           rl)
                        args = (experiment, irl_name, rl, env_name, parallel,
                                log_dir, r, discount, seed)
                        delayed = pool.apply_async(_value, args)
                        res_by_env.setdefault(env_name, {})[rl] = delayed
                    res_by_m[m] = res_by_env
                res_by_n[n] = res_by_m
            value[irl_name] = res_by_n

        for env_name in cfg.get('test_environments', cfg.get('environments')):
            log_dir = osp.join(out_dir, 'eval', sanitize_env_name(env_name),
                               'gt', rl)
            args = (rl, discount, env_name, parallel, seed, log_dir)
            delayed = pool.apply_async(_train_policy, args)
            ground_truth.setdefault(env_name, {})[rl] = delayed

    value = nested_async_get(value)
    ground_truth = nested_async_get(ground_truth, lambda x: x[1])

    return value, ground_truth


def run_experiment(experiment, pool, out_dir, video_every, seed):
    '''Run experiment defined in config.EXPERIMENTS.

    Args:
        - experiment(str): experiment name.
        - pool(multiprocessing.Pool)
        - out_dir(str): path to write logs and results to.
        - video_every(optional[int]): if None, do not record video.
        - seed(int)

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
    cfg = config.EXPERIMENTS[experiment]

    # Generate synthetic data
    trajs, expert_vals = expert_trajs(experiment, out_dir, cfg, pool,
                                      video_every, seed)

    # Run IRL
    rewards, irl_values = run_irl(experiment, out_dir, cfg, pool, trajs, seed)

    # Evaluate IRL by reoptimizing in cfg['evals']
    values, ground_truth = value(experiment, out_dir, cfg, pool, rewards, seed)

    # Add in the value obtained by the expert policy & IRL policy
    for name, val in expert_vals.items():
        ground_truth.setdefault(name, collections.OrderedDict())['expert'] = val
    for irl_name, d1 in irl_values.items():
        for n, d2 in d1.items():
            for m, d3 in d2.items():
                for env, val in d3.items():
                    setdef(setdef(setdef(setdef(values, irl_name), n), m), env)['irl'] = val

    return {
        'trajectories': trajs,
        'rewards': rewards,
        'values': values,
        'ground_truth': ground_truth,
    }
