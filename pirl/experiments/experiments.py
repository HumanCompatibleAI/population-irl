import collections
import functools
import itertools
import logging
import os.path as osp

from joblib import Memory
import gym

from pirl import utils
from pirl.experiments import config

logger = logging.getLogger('pirl.experiments.experiments')
memory = Memory(cachedir=config.CACHE_DIR, verbose=0)

def make_rl_algo(algo):
    return config.RL_ALGORITHMS[algo]


def make_irl_algo(algo):
    return config.IRL_ALGORITHMS[algo]


@memory.cache(ignore=['out_dir', 'video_every', 'policy'])
def synthetic_data(name, env, rl, num_trajectories, seed,
                   out_dir, video_every, policy):
    '''Precondition: policy produced by RL algorithm rl.'''
    _, sample, _ = make_rl_algo(rl)

    video_dir = '{}/{}/videos'.format(out_dir, name.replace('/', '_'))
    if video_every is None:
        video_callable = lambda x: False
    else:
        video_callable = lambda x: x % video_every == 0
    env = gym.wrappers.Monitor(env,
                               video_dir,
                               video_callable=video_callable,
                               force=True)
    samples = sample(env, policy, num_trajectories, seed)
    #TODO: numpy array rather than Python list?
    return [(observations, actions) for (observations, actions, rewards) in samples]


class LearnedRewardWrapper(gym.Wrapper):
    """
    Wrapper for a gym.Env replacing with a new reward matrix.
    Intended for the tabular setting. Will work when the observations are
    discrete and the reward is a function of the observation.
    """
    def __init__(self, env, new_reward):
        self.new_reward = new_reward
        super().__init__(env)

    def step(self, action):
        observation, old_reward, done, info = self.env.step(action)
        #TODO: this won't work if observations are continuous?
        # (It was written only to handle the tabular setting, probably needs
        #  to be extended once we have new environments.)
        new_reward = self.new_reward[observation, action]
        return observation, new_reward, done, info

    @property
    def reward(self):
        return self.new_reward


@utils.log_errors
def _run_irl(irl_name, n, m, small_env, experiment,
            out_dir, envs, trajectories, discount):
    logger.debug('%s: running IRL algo: %s [%s=%s/%s]',
                 experiment, irl_name, small_env, m, n)
    irl_algo, compute_value = make_irl_algo(irl_name)
    subset = {k: v[:n] for k, v in trajectories.items()}
    if small_env is not None:
        subset[small_env] = subset[small_env][:m]
    log_dir = osp.join(out_dir, '{}:{}:{}/{}'.format(irl_name, small_env, m, n))
    rewards, policies = irl_algo(envs, subset, discount=discount, log_dir=log_dir)
    values = {k: compute_value(envs[k], p, discount) for k, p in policies.items()}
    return rewards, values

def run_irl(experiment, out_dir, cfg, pool, envs, trajectories):
    '''Run experiment in parallel. Returns tuple (reward, info) where each are
       nested OrderedDicts, with key in the format:
        - IRL algo
        - Number of trajectories for other environments
        - Number of trajectories for this environment
        - Environment
       Note that for this experiment type, the second and third arguments are
       always the same.
    '''
    f = functools.partial(_run_irl, experiment=experiment, out_dir=out_dir, envs=envs,
                          trajectories=trajectories, discount=cfg['discount'],
                          m=None, small_env=None)
    args = list(itertools.product(cfg['irl'], sorted(cfg['num_trajectories'])))
    results = pool.starmap(f, args, chunksize=1)
    rewards = collections.OrderedDict()
    values = collections.OrderedDict()
    for (irl_name, n), (reward, value) in zip(args, results):
        reward = collections.OrderedDict([(n, reward)])
        value = collections.OrderedDict([(n, value)])
        rewards.setdefault(irl_name, collections.OrderedDict())[n] = reward
        values.setdefault(irl_name, collections.OrderedDict())[n] = value
    return rewards, values


def run_few_shot_irl(experiment, out_dir, cfg, pool, envs, trajectories):
    '''Same spec as run_irl.'''
    f = functools.partial(_run_irl, experiment=experiment, out_dir=out_dir, envs=envs,
                          trajectories=trajectories, discount=cfg['discount'])
    args = list(itertools.product(cfg['irl'],
                                  sorted(cfg['num_trajectories']),
                                  sorted(cfg['few_shot']),
                                  envs.keys()))
    results = pool.starmap(f, args, chunksize=1)
    rewards = collections.OrderedDict()
    values = collections.OrderedDict()
    for (irl_name, n, m, env), (reward, value) in zip(args, results):
        if irl_name not in rewards:
            rewards[irl_name] = collections.OrderedDict()
            values[irl_name] = collections.OrderedDict()
        if n not in rewards[irl_name]:
            rewards[irl_name][n] = collections.OrderedDict()
            values[irl_name][n] = collections.OrderedDict()
        if m not in rewards[irl_name][n]:
            rewards[irl_name][n][m] = collections.OrderedDict()
            values[irl_name][n][m] = collections.OrderedDict()
        rewards[irl_name][n][m][env] = reward[env]
        values[irl_name][n][m][env] = value
    return rewards, values

def _value(experiment, cfg, envs, rewards):
    '''
    Compute the expected value of (a) policies optimized on inferred reward,
    and (b) optimal policies for the ground truth reward. Policies will be 
    computed using each RL algorithm specified in cfg['eval'].

    Args:
        - experiment
        - cfg: config.EXPERIMENTS[experiment]
        - envs
        - rewards
    Returns:
        tuple, (value, ground_truth) where each is a nested dictionary of the
        same shape as rewards, with the leaf being a dictionary mapping from
        an RL algorithm in cfg['eval'] to a scalar value.
    '''
    discount = cfg['discount']
    value = collections.OrderedDict()
    for rl in cfg['eval']:
        #TODO: parallelize?
        gen_policy, _sample, compute_value = make_rl_algo(rl)

        for irl_name, reward_by_size in rewards.items():
            res_by_n = collections.OrderedDict()
            for n, reward_by_small_size in reward_by_size.items():
                res_by_m = collections.OrderedDict()
                for m, reward_by_env in reward_by_small_size.items():
                    res_by_env = {}
                    for env_name, env in envs.items():
                        logger.debug('%s: evaluating %s on %s with %d/%d trajectories',
                                     experiment, irl_name, env_name, m, n)
                        r = reward_by_env[env_name]
                        # TODO: alternately, we could pass the new reward directly
                        # to gen_policy as an override -- unsure which is cleaner?
                        wrapped_env = LearnedRewardWrapper(env, r)
                        p = gen_policy(wrapped_env, discount=discount)
                        v = compute_value(wrapped_env, p, discount=discount)
                        res_by_env.setdefault(env_name, {})[rl] = v
                    res_by_m[m] = res_by_env
                res_by_n[n] = res_by_m
            value[irl_name] = res_by_n

        ground_truth = {}
        for env_name, env in envs.items():
            p = gen_policy(env, discount=discount)
            v = compute_value(env, p, discount=discount)
            ground_truth.setdefault(env_name, {})[rl] = v

        return value, ground_truth

@memory.cache(ignore=['env', 'out_dir'])
def _train_policy(rl, discount, env_name, seed, env, out_dir):
    gen_policy, _sample, compute_value = make_rl_algo(rl)
    log_dir = osp.join(out_dir, env_name)

    env.seed(seed)
    p = gen_policy(env, discount=discount, log_dir=log_dir)
    v = compute_value(env, p, discount=discount)

    return p, v

def _expert_trajs(experiment, out_dir, cfg, video_every, seed, envs):
    logger.debug('%s: generating synthetic data: training', experiment)
    rl_name = cfg['expert']
    policies = collections.OrderedDict()
    values = collections.OrderedDict()
    for name, env in envs.items():
        p, v = _train_policy(rl_name, cfg['discount'], name, seed, env, out_dir)
        policies[name] = p
        values[name] = v

    logger.debug('%s: generating synthetic data: sampling', experiment)
    _, sample, _ = make_rl_algo(cfg['expert'])
    max_trajectories = max(cfg['num_trajectories'])
    trajectories = collections.OrderedDict(
        (name, synthetic_data(name, env, cfg['expert'], max_trajectories, seed,
                              out_dir, video_every, policies[name]))
        for name, env in envs.items()
    )

    return trajectories, values

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
    #TODO: Should I be closing/resetting environments?
    seed = utils.random_seed(seed)
    cfg = config.EXPERIMENTS[experiment]

    # Generate synthetic data
    logger.debug('%s: creating environments %s', experiment, cfg['environments'])
    # I use sorted collections to make experiments (closer to) deterministic
    envs = collections.OrderedDict(
        (name, gym.make(name))
        for name in cfg['environments']
    )

    expert_trajs, expert_vals = _expert_trajs(experiment, out_dir, cfg,
                                              video_every, seed, envs)

    # Run IRL
    fn = run_few_shot_irl if 'few_shot' in cfg else run_irl
    rewards, irl_values = fn(experiment, out_dir, cfg, pool, envs, expert_trajs)

    # Evaluate IRL by reoptimizing in cfg['evals']
    values, ground_truth = _value(experiment, cfg, envs, rewards)

    # Add in the value obtained by the expert policy & IRL policy
    for name, val in expert_vals.items():
        ground_truth[name]['expert'] = val
    for irl_name, d1 in irl_values.items():
        for n, d2 in d1.items():
            for m, d3 in d2.items():
                for env, val in d3.items():
                    values[irl_name][n][m][env]['irl'] = val

    return {
        'trajectories': expert_trajs,
        'rewards': rewards,
        'values': values,
        'ground_truth': expert_vals,
    }
