import collections
import functools
import itertools
import logging
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


def synthetic_data(env, policy, num_trajectories, path, name, seed):
    env = gym.wrappers.Monitor(env, '{}/{}/videos'.format(path, name.replace('/', '_')), video_callable=lambda x: True, force=True)
    rng, _ = seeding.np_random(seed)
    env.seed(seed)
    trajectories = [sample(env, policy, rng) for _i in range(num_trajectories)]
    return trajectories


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
def _run_irl(irl_name, n, experiment, envs, trajectories, discount):
    logger.debug('%s: running IRL algo: %s [%d]', experiment, irl_name, n)
    irl_algo = make_irl_algo(irl_name)
    subset = {k: v[:n] for k, v in trajectories.items()}
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


@utils.log_errors
def _run_few_shot_irl(irl_name, n, m, small_env,
                      experiment, envs, trajectories, discount):
    logger.debug('%s: running IRL algo: %s [%s=%d/%d]',
                 experiment, irl_name, small_env, m, n)
    irl_algo = make_irl_algo(irl_name)
    subset = {k: v[:n] for k, v in trajectories.items()}
    subset[small_env] = subset[small_env][:m]
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

def _value(experiment, cfg, envs, rewards, rl_algo):
    '''
    Compute the expected value of (a) policies optimized on inferred reward,
    and (b) optimal policies for the ground truth reward. In the first case,
    policies will be computed using (i) an optimal planner (prefix 'opt_') and
    (ii) the planner assumed by the IRL algorithm (prefix 'pla_').

    (i) is more principled, but it can be a noisy metric: small changes in
    reward may drastically change the optimal policy and, thus, the value.
    By contrast, (ii) is more robust, but can have surprising properties.
    For example, a Boltzmann rational planner will become closer to optimal
    when the reward is larger in magnitude, so the best policy is not attained
    on the ground truth reward (but rather a scaled-up version).

    Args:
        - envs:
        - rewards
        - rl_algo: return value of make_rl_algo.
    Returns:
        tuple, (value, ground_truth) where:

            - value: nested dictionaries of same shape as rewards,
              with leafs containing a pair of values corresponding to (i) & (ii).
            - ground_truth: dictionary of same shape as environment,
              with leafs containing value of optimal ground truth policy.
    '''
    gen_policy, gen_optimal_policy, compute_value = rl_algo
    discount = cfg['discount']
    value = collections.OrderedDict()
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
                    optimal_p = gen_optimal_policy(wrapped_env,
                                                   discount=cfg['discount'])
                    optimal_v = compute_value(env, optimal_p,
                                              discount=cfg['discount'])
                    planner_p = gen_policy(wrapped_env,
                                           discount=cfg['discount'])
                    planner_v = compute_value(env, planner_p,
                                              discount=cfg['discount'])

                    res_by_env[env_name] = (optimal_v, planner_v)
                res_by_m[m] = res_by_env
            res_by_n[n] = res_by_m
        value[irl_name] = res_by_n

    ground_truth = {}
    for env_name, env in envs.items():
        policy = gen_optimal_policy(env, discount=discount)
        ground_truth[env_name] = compute_value(env, policy, discount=discount)

    return value, ground_truth


def run_experiment(experiment, pool, path, seed):
    '''Run experiment defined in config.EXPERIMENTS.

    Args:
        - experiment(str): experiment name.
        - pool(multiprocessing.Pool)
        - seed(int)

    Returns:
        tuple, (trajectories, rewards, value), where:

        - trajectories: synthetic data.
            dict, keyed by environments, with values generated by synthetic_data.
        - rewards: IRL inferred reward.
            nested dict, keyed by environment then IRL algorithm.
        - value: value obtained reoptimizing in the environment.
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
    rl_algo = make_rl_algo(cfg['rl'])
    gen_policy = rl_algo[0]
    policies = collections.OrderedDict(
        (name, gen_policy(env, discount=cfg['discount']))
        for name, env in envs.items()
    )
    logger.debug('%s: generating synthetic data: sampling', experiment)
    trajectories = collections.OrderedDict(
        (k, synthetic_data(e, policies[k], max(cfg['num_trajectories']), path, k, seed))
        for k, e in envs.items()
    )

    # Run IRL
    fn = run_few_shot_irl if 'few_shot' in cfg else run_irl
    rewards, infos = fn(experiment, cfg, pool, envs, trajectories)

    value, ground_truth = _value(experiment, cfg, envs, rewards, rl_algo)

    return {
        'trajectories': trajectories,
        'reward': rewards,
        'value': value,
        'ground_truth': ground_truth,
        'info': infos,
    }
