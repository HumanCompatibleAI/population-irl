import functools
import itertools
import os.path as osp
import sys

import gym
import tensorflow as tf

from pirl import agents, envs, irl

# General
PROJECT_DIR = osp.dirname(osp.dirname(osp.dirname(osp.realpath(__file__))))
CACHE_DIR = osp.join(PROJECT_DIR, 'data', 'cache')

# ML Framework Config
def make_tf_config():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return config
TENSORFLOW = make_tf_config()

# Logging
def logging(identifier):
    return {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '(%(processName)s) %(asctime)s [%(levelname)s] %(name)s: %(message)s',
            },
        },
        'handlers': {
            'stream': {
                'level': 'DEBUG',
                'formatter': 'standard',
                'class': 'logging.StreamHandler',
            },
            'file': {
                'level': 'DEBUG',
                'formatter': 'standard',
                'filename': 'logs/pirl-{}.log'.format(identifier),
                'maxBytes': 100 * 1024 * 1024,
                'backupCount': 3,
                'class': 'logging.handlers.RotatingFileHandler',
            },
        },
        'loggers': {
            '': {
                'handlers': ['stream', 'file'],
                'level': 'DEBUG',
                'propagate': True
            },
        }
    }

# RL Algorithms

# Values take form (gen_policy, sample, compute_value).
#
# gen_policy has signature (env, discount, log_dir), where env is a gym.Env,
# discount is a float and log_dir is a writable directory.
# They return a policy (algorithm-specific object).
#
# sample has signature (env, policy, num_episodes, seed) where
# num_episodes is the number of trajectories to sample, and seed is used
# to sample deterministically. It returns a list of 3-tuples
# (states, actions, rewards), each of which is a list.
#
# compute_value has signature (env, policy, discount).
# It returns (mean, se) where mean is the estimated reward and se is the
# standard error (0 for exact methods).
RL_ALGORITHMS = {
    'value_iteration': (
        agents.tabular.env_wrapper(agents.tabular.q_iteration_policy),
        agents.tabular.sample,
        agents.tabular.value_of_policy,
    ),
    'max_ent': (
        agents.tabular.env_wrapper(irl.tabular_maxent.max_ent_policy),
        agents.tabular.sample,
        agents.tabular.value_of_policy,
    ),
    'max_causal_ent': (
        agents.tabular.env_wrapper(irl.tabular_maxent.max_causal_ent_policy),
        agents.tabular.sample,
        agents.tabular.value_of_policy,
    ),
}

ppo_cts_pol = functools.partial(agents.ppo.train_continuous,
                                tf_config=TENSORFLOW,
                                num_timesteps=1e6)
ppo_sample = functools.partial(agents.ppo.sample, tf_config=TENSORFLOW)
ppo_value = functools.partial(agents.continuous.value, ppo_sample)
RL_ALGORITHMS['ppo_cts'] = (ppo_cts_pol, ppo_sample, ppo_value)

ppo_cts_pol_quick = functools.partial(agents.ppo.train_continuous,
                                      tf_config=TENSORFLOW,
                                      num_timesteps=1e4)
RL_ALGORITHMS['ppo_cts_quick'] = (ppo_cts_pol_quick, ppo_sample, ppo_value)

# IRL Algorithms
def traditional_to_single(fs):
    irl_algo, reward_wrapper, compute_value = fs
    @functools.wraps(irl_algo)
    def helper(envs, trajectories, **kwargs):
        #SOMEDAY: parallelize
        res = {k: irl_algo(envs[k], v, **kwargs) for k, v in trajectories.items()}
        rewards = {k: v[0] for k, v in res.items()}
        policies = {k: v[1] for k, v in res.items()}
        return rewards, policies
    return helper, reward_wrapper, compute_value


def traditional_to_concat(fs):
    irl_algo, reward_wrapper, compute_value = fs
    @functools.wraps(irl_algo)
    def helper(envs, trajectories, **kwargs):
        concat_trajectories = list(itertools.chain(*trajectories.values()))
        # Pick an environment arbitrarily. In the typical use case,
        # they are all the same up to reward anyway.
        env = list(envs.values())[0]
        reward, policy = irl_algo(env, concat_trajectories, **kwargs)
        rewards = {k: reward for k in trajectories.keys()}
        policies = {k: policy for k in trajectories.keys()}
        return rewards, policies
    return helper, reward_wrapper, compute_value


TRADITIONAL_IRL_ALGORITHMS = {
    # Maximum Causal Entropy (Ziebart 2010)
    'mce': (irl.tabular_maxent.irl,
            agents.tabular.TabularRewardWrapper,
            agents.tabular.value_of_policy),
    # Maximum Entropy (Ziebart 2008)
    'me': (functools.partial(irl.tabular_maxent.irl, planner=irl.tabular_maxent.max_ent_policy),
           agents.tabular.TabularRewardWrapper,
           agents.tabular.value_of_policy),
}

airl_irl = functools.partial(irl.airl.irl, tf_cfg=TENSORFLOW)
airl_quick_irl = functools.partial(airl_irl, irl_cfg={'n_itr': 10})
airl_reward = functools.partial(irl.airl.AIRLRewardWrapper, tf_cfg=TENSORFLOW)
airl_value = functools.partial(agents.continuous.value,
                functools.partial(irl.airl.sample, tf_cfg=TENSORFLOW))
TRADITIONAL_IRL_ALGORITHMS['airl'] = (airl_irl, airl_reward, airl_value)
TRADITIONAL_IRL_ALGORITHMS['airl_quick'] = (airl_quick_irl, airl_reward, airl_value)

MY_IRL_ALGORITHMS = dict()
for reg in range(-2,3):
    fn = functools.partial(irl.tabular_maxent.population_irl,
                           individual_reg=10 ** reg)
    MY_IRL_ALGORITHMS['mcep_reg1e{}'.format(reg)] = fn, agents.tabular.value_of_policy
MY_IRL_ALGORITHMS['mcep_reg0'] = (
    functools.partial(irl.tabular_maxent.population_irl, individual_reg=0),
    agents.tabular.TabularRewardWrapper,
    agents.tabular.value_of_policy)

# Values take the form: (irl, reward_wrapper, compute_value).
#
# irl signature (env, trajectories, discount, log_dir) where:
# - env is a gym.Env.
# - trajectories is a dict of environment IDs to lists of trajectories.
# - discount is a float in [0,1].
# - log_dir is a directory which may be used for logging or other temporary output.
# It returns a tuple (reward, policy), both of which are algorithm-specific
# objects. reward must be comprehensible to RL algorithms (if any) specified in
# the 'eval' key in the experimental config.
#
# reward_wrapper is a class with signature __init__(env, reward).
# It wraps environment and overrides step() to return the reward learnt by
# the IRL algorithm.
#
# compute_value has signature (env, policy, discount) where:
# - env is a gym.Env.
# - policy is as returned by the IRL algorithm.
# - discount is a float in [0,1].
# It returns (mean, se) where mean is the estimated reward and se is the
# standard error (0 for exact methods).
IRL_ALGORITHMS = dict()
IRL_ALGORITHMS.update(MY_IRL_ALGORITHMS)
for name, algo in TRADITIONAL_IRL_ALGORITHMS.items():
    IRL_ALGORITHMS[name + 's'] = traditional_to_single(algo)
    IRL_ALGORITHMS[name + 'c'] = traditional_to_concat(algo)

EXPERIMENTS = {}

# ONLY FOR TESTING CODE! Not real experiments.
EXPERIMENTS['dummy-test'] = {
    'environments': ['pirl/GridWorld-Simple-v0'],
    'discount': 1.00,
    'expert': 'value_iteration',
    'eval': ['value_iteration'],
    'irl': ['mcep_reg0', 'mces'],
    'num_trajectories': [20, 10],
}
EXPERIMENTS['few-dummy-test'] = {
    'environments': ['pirl/GridWorld-Simple-v0',
                     'pirl/GridWorld-Simple-Deterministic-v0'],
    'discount': 1.00,
    'expert': 'value_iteration',
    'eval': ['value_iteration'],
    'irl': ['mces', 'mcec', 'mcep_reg0'],
    'num_trajectories': [20],
    'few_shot': [1, 5],
}
EXPERIMENTS['dummy-test-deterministic'] = {
    'environments': ['pirl/GridWorld-Simple-Deterministic-v0'],
    'discount': 1.00,
    'expert': 'value_iteration',
    'eval': ['value_iteration'],
    'irl': ['mces', 'mcep_reg0'],
    'num_trajectories': [20, 10],
}
EXPERIMENTS['dummy-continuous-test'] = {
    'environments': ['Reacher-v2'],
    'discount': 0.99,
    'expert': 'ppo_cts_quick',
    'eval': ['ppo_cts_quick'],
    'irl': ['airl_quicks'],
    'num_trajectories': [10, 20],
}
EXPERIMENTS['dummy-continuous-test-slow'] = {
    'environments': ['Reacher-v2'],
    'discount': 0.99,
    'expert': 'ppo_cts',
    'eval': ['ppo_cts'],
    'irl': ['airls'],
    'num_trajectories': [10, 100, 1000],
}

# Jungle gridworld experiments
EXPERIMENTS['jungle'] = {
    'environments': ['pirl/GridWorld-Jungle-9x9-{}-v0'.format(k)
                     for k in ['Soda', 'Water', 'Liquid']],
    'discount': 1.00,
    'expert': 'max_causal_ent',
    'eval': ['value_iteration'],
    'irl': [
        'mces',
        'mcec',
        'mcep_reg0',
        'mcep_reg1e-2',
        'mcep_reg1e-1',
        'mcep_reg1e0',
    ],
    'num_trajectories': [1000, 500, 200, 100, 50, 30, 20, 10, 5],
}
EXPERIMENTS['jungle-small'] = {
    'environments': ['pirl/GridWorld-Jungle-4x4-{}-v0'.format(k)
                     for k in ['Soda', 'Water', 'Liquid']],
    'discount': 1.00,
    'expert': 'max_causal_ent',
    'eval': ['value_iteration'],
    'irl': [
        'mces',
        'mcec',
        'mcep_reg0',
        'mcep_reg1e-2',
        'mcep_reg1e-1',
        'mcep_reg1e0',
    ],
    'num_trajectories': [500, 200, 100, 50, 30, 20, 10, 5],
}

# Test different planner combinations
EXPERIMENTS['unexpected-optimal'] = {
    'environments': ['pirl/GridWorld-Jungle-4x4-Soda-v0'],
    'discount': 1.00,
    'expert': 'max_causal_ent',
    'eval': ['value_iteration'],
    'irl': [
        'mces',
        'mes',
    ],
    'num_trajectories': [200],
}

# Few-shot learning
EXPERIMENTS['few-jungle'] = {
    'environments': ['pirl/GridWorld-Jungle-9x9-{}-v0'.format(k)
                     for k in ['Soda', 'Water', 'Liquid']],
    'discount': 1.00,
    'expert': 'max_causal_ent',
    'eval': ['value_iteration'],
    'irl': [
        'mces',
        'mcec',
        'mcep_reg0',
        'mcep_reg1e-2',
        'mcep_reg1e-1',
        'mcep_reg1e0',
    ],
    'num_trajectories': [1000],
    'few_shot': [1, 2, 5, 10, 20, 50, 100],
}
EXPERIMENTS['few-jungle-small'] = {
    'environments': ['pirl/GridWorld-Jungle-4x4-{}-v0'.format(k)
                     for k in ['Soda', 'Water', 'Liquid']],
    'discount': 1.00,
    'expert': 'max_causal_ent',
    'eval': ['value_iteration'],
    'irl': [
        'mces',
        'mcec',
        'mcep_reg0',
        'mcep_reg1e-2',
        'mcep_reg1e-1',
        'mcep_reg1e0',
    ],
    'num_trajectories': [1000],
    'few_shot': [1, 2, 5, 10, 20, 50, 100],
}

# Continuous control
EXPERIMENTS['billiards'] = {
    'environments': ['pirl/Billiards-seed{}-v0'.format(i) for i in range(4)],
    'discount': 0.99,
    'expert': 'ppo_cts',
    'eval': ['ppo_cts'],
    'irl': [
        'airls',
    ],
    'num_trajectories': [1000],
}

def validate_config():
    for k, v in EXPERIMENTS.items():
        try:
            gym.envs.registry.spec('pirl/GridWorld-Jungle-4x4-Liquid-v0')
            float(v['discount'])
            RL_ALGORITHMS[v['expert']]
            for rl in v.get(eval, []):
                RL_ALGORITHMS[rl]
            for irl in v['irl']:
                IRL_ALGORITHMS[irl]
            [int(t) for t in v['num_trajectories']]
            [int(t) for t in v.get('few_shot', [])]
        except Exception as e:
            msg = 'In experiment ' + k + ': ' + str(e)
            raise type(e)(msg).with_traceback(sys.exc_info()[2])
