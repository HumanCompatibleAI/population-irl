import collections
import functools
import itertools
import logging
import os

import git
import gym
from gym.utils import seeding

from pirl import utils
from pirl.experiments import config

logger = logging.getLogger('pirl.experiments.experiments')

#TODO: refactor once structure is clearer
# Should this be pushed into agents package?
# Will different agents require some setup code?
def make_rl_algo(algo):
    return config.RL_ALGORITHMS[algo]


def make_irl_algo(algo):
    return config.IRL_ALGORITHMS[algo]


def sample(env, policy, rng):
    #TODO: generalize. This is specialised to fully-observable MDPs,
    # with a stochastic policy matrix.

    states = []
    actions = []

    state = env.reset()

    done = False
    while not done:
        states.append(state)
        action_dist = policy[state]
        action = utils.discrete_sample(action_dist, rng)
        actions.append(action)
        state, reward, done, _ = env.step(action)

    return states, actions


def synthetic_data(env, policy, num_trajectories, seed):
    rng, _ = seeding.np_random(seed)
    env.seed(seed)
    trajectories = [sample(env, policy, rng) for _i in range(num_trajectories)]
    return trajectories

def slice_trajectories(trajectories, max_length):
    return {k: [(states[:max_length], actions[:max_length])
                for states, actions in env_trajectories]
            for k, env_trajectories in trajectories.items()}

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

def _run_irl(irl_name, n, experiment, envs, trajectories, discount):
    logger.debug('%s: running IRL algo: %s [%d]', experiment, irl_name, n)
    irl_algo = make_irl_algo(irl_name)
    subset = slice_trajectories(trajectories, n)
    return irl_algo(envs, subset, discount=discount)

def run_irl(experiment, cfg, pool, envs, trajectories):
    '''Run experiment in parallel. Returns tuple (reward, info) where each are
       nested OrderedDicts, with key in the format:
        - IRL algo
        - Number of trajectories for other environments
        - Number of trajectories for this environment
        - Environment
       Note that for this experiment type, the second and third arguments are
       always the same.
    '''
    f = functools.partial(_run_irl, experiment=experiment, envs=envs,
                          trajectories=trajectories, discount=cfg['discount'])
    args = list(itertools.product(cfg['irl'], sorted(cfg['num_trajectories'])))
    results = pool.starmap(f, args, chunksize=1)
    rewards = collections.OrderedDict()
    infos = collections.OrderedDict()
    for (irl_name, n), (reward, info) in zip(args, results):
        reward = collections.OrderedDict([(n, reward)])
        info = collections.OrderedDict([(n, info)])
        rewards.setdefault(irl_name, collections.OrderedDict())[n] = reward
        infos.setdefault(irl_name, collections.OrderedDict())[n] = info
    return rewards, info


def _run_few_shot_irl(irl_name, n, m, small_env,
                      experiment, envs, trajectories, discount):
    logger.debug('%s: running IRL algo: %s [%s=%d/%d]',
                 experiment, irl_name, small_env, m, n)
    irl_algo = make_irl_algo(irl_name)
    subset = slice_trajectories(trajectories, n)
    small = slice_trajectories({'x': trajectories[small_env]}, m)['x']
    subset[small_env] = small
    return irl_algo(envs, subset, discount=discount)

def run_few_shot_irl(experiment, cfg, pool, envs, trajectories):
    '''Same spec as run_irl.'''
    f = functools.partial(_run_few_shot_irl, experiment=experiment, envs=envs,
                          trajectories=trajectories, discount=cfg['discount'])
    args = list(itertools.product(cfg['irl'],
                                  sorted(cfg['num_trajectories']),
                                  sorted(cfg['few_shot']),
                                  envs.keys()))
    results = pool.starmap(f, args, chunksize=1)
    rewards = collections.OrderedDict()
    infos = collections.OrderedDict()
    for (irl_name, n, m, env), (reward, info) in zip(args, results):
        if irl_name not in rewards:
            rewards[irl_name] = collections.OrderedDict()
            infos[irl_name] = collections.OrderedDict()
        if n not in rewards[irl_name]:
            rewards[irl_name][n] = collections.OrderedDict()
            infos[irl_name][n] = collections.OrderedDict()
        if m not in rewards[irl_name][n]:
            rewards[irl_name][n][m] = collections.OrderedDict()
            infos[irl_name][n][m] = collections.OrderedDict()
        rewards[irl_name][n][m][env] = reward[env]
        infos[irl_name][n][m][env] = info
    return rewards, infos


def git_hash():
    repo = git.Repo(path=os.path.realpath(__file__),
                    search_parent_directories=True)
    return repo.head.object.hexsha

def run_experiment(experiment, pool, seed):
    '''Run experiment defined in config.EXPERIMENTS.

    Args:
        - experiment(str): experiment name.
        - pool(multiprocessing.Pool)
        - seed(int)

    Returns:
        tuple, (trajectories, rewards, expected_value), where:

        - trajectories: synthetic data.
            dict, keyed by environments, with values generated by synthetic_data.
        - rewards: IRL inferred reward.
            nested dict, keyed by environment then IRL algorithm.
        - expected_value: value obtained reoptimizing in the environment.
            Use the RL algorithm used to generate the original synthetic data
            to train a policy on the inferred reward, then compute expected
            discounted value obtained from the resulting policy.
        '''
    utils.random_seed(seed)
    cfg = config.EXPERIMENTS[experiment]

    # Generate synthetic data
    logger.debug('%s: creating environments %s', experiment, cfg['environments'])
    # To make experiments (more) deterministic, I use sorted collections.
    envs = collections.OrderedDict()
    for name in cfg['environments']:
        env = gym.make(name)
        envs[name] = env

    logger.debug('%s: generating synthetic data: training', experiment)
    gen_policy, gen_optimal_policy, compute_value = make_rl_algo(cfg['rl'])
    policies = collections.OrderedDict(
        (name, gen_policy(env, discount=cfg['discount']))
        for name, env in envs.items()
    )
    logger.debug('%s: generating synthetic data: sampling', experiment)
    trajectories = collections.OrderedDict(
        (k, synthetic_data(e, policies[k], max(cfg['num_trajectories']), seed))
        for k, e in envs.items()
    )

    # Run IRL
    fn = run_few_shot_irl if 'few_shot' in cfg else run_irl
    rewards, infos = fn(experiment, cfg, pool, envs, trajectories)

    # Evaluate results
    # Note the expected value is estimated, and the accuracy of this may depend
    # on the RL algorithm. For value iteration, for example, this is computed
    # directly; for many other algorithms, a sample-based approach is adopted.
    expected_value = {}
    for irl_name, reward_by_size in rewards.items():
        res = collections.OrderedDict()
        for n, reward_by_small_size in reward_by_size.items():
            for m, reward_by_env in reward_by_small_size.items():
                for env_name, env in envs.items():
                    logger.debug('%s: evaluating %s on %s with %d/%d trajectories',
                                 experiment, irl_name, env_name, m, n)
                    r = reward_by_env[env_name]
                    # TODO: alternately, we could pass the new reward directly
                    # to gen_policy as an override -- unsure which is cleaner?
                    wrapped_env = LearnedRewardWrapper(env, r)
                    reoptimized_policy = gen_optimal_policy(wrapped_env,
                                            discount=cfg['discount'])
                    value = compute_value(env, reoptimized_policy,
                                          discount=cfg['discount'])
                    res.setdefault(env_name, {})[n] = value
                expected_value[irl_name] = res
    ground_truth = {env_name: compute_value(env, policies[env_name],
                                            discount=cfg['discount'])}

    return {
        'trajectories': trajectories,
        'reward': rewards,
        'expected_value': expected_value,
        'ground_truth': ground_truth,
        'info': infos,
        'version': git_hash(),
    }
