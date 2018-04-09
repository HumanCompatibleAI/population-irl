import functools
import itertools
import sys

import gym
import tensorflow as tf

from pirl import agents, envs, irl

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
ppo_cts = functools.partial(agents.ppo.train_continuous, num_timesteps=1e6)

RL_ALGORITHMS = {
    # Values take form (gen_policy, gen_optimal_policy, compute_value, sample).
    #
    # Both gen_* functions have signature (env, discount) where env is a gym.Env
    # and discount is a float. They return a policy (algorithm-specific object).
    #
    # compute_value has signature (env, policy, discount).
    # It returns (mean, se) where mean is the estimated reward and se is the
    # standard error (0 for exact methods).
    #
    # sample has signature (env, policy, num_episodes, seed) where
    # num_episodes is the number of trajectories to sample, and seed is used
    # to sample deterministically. It returns a list of 3-tuples
    # (states, actions, rewards), each of which is a list.
    'value_iteration': (
        agents.tabular.env_wrapper(agents.tabular.q_iteration_policy),
        agents.tabular.env_wrapper(agents.tabular.q_iteration_policy),
        agents.tabular.value_of_policy,
        agents.tabular.sample,
    ),
    'max_ent': (
        agents.tabular.env_wrapper(irl.tabular_maxent.max_ent_policy),
        agents.tabular.env_wrapper(agents.tabular.q_iteration_policy),
        agents.tabular.value_of_policy,
        agents.tabular.sample,
    ),
    'max_causal_ent': (
        agents.tabular.env_wrapper(irl.tabular_maxent.max_causal_ent_policy),
        agents.tabular.env_wrapper(agents.tabular.q_iteration_policy),
        agents.tabular.value_of_policy,
        agents.tabular.sample,
    ),
    'ppo_cts': (
        ppo_cts,
        ppo_cts,
        agents.ppo.value,
        agents.ppo.sample,
    )
}

# IRL Algorithms
def traditional_to_single(f):
    @functools.wraps(f)
    def helper(envs, trajectories, **kwargs):
        #TODO: parallelize
        res = {k: f(envs[k], v, **kwargs) for k, v in trajectories.items()}
        values = {k: v[0] for k, v in res.items()}
        info = {k: v[1] for k, v in res.items()}
        return values, info
    return helper


def traditional_to_concat(f):
    @functools.wraps(f)
    def helper(envs, trajectories, **kwargs):
        concat_trajectories = list(itertools.chain(*trajectories.values()))
        # Pick an environment arbitrarily. In the typical use case,
        # they are all the same up to reward anyway.
        env = list(envs.values())[0]
        value, info = f(env, concat_trajectories, **kwargs)
        value = {k: value for k in trajectories.keys()}
        return value, info
    return helper


TRADITIONAL_IRL_ALGORITHMS = {
    # Maximum Causal Entropy (Ziebart 2010)
    'mce': irl.tabular_maxent.irl,
    # Maximum Entropy (Ziebart 2008)
    'me': functools.partial(irl.tabular_maxent.irl,
                            planner=irl.tabular_maxent.max_ent_policy),
}

MY_IRL_ALGORITHMS = dict()
for reg in range(-2,3):
    fn = functools.partial(irl.tabular_maxent.population_irl,
                           individual_reg=10 ** reg)
    MY_IRL_ALGORITHMS['mcep_reg1e{}'.format(reg)] = fn
MY_IRL_ALGORITHMS['mcep_reg0'] = functools.partial(
    irl.tabular_maxent.population_irl, individual_reg=0)

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
    'rl': 'value_iteration',
    'irl': ['mcep_reg0', 'mces'],
    'num_trajectories': [20, 10],
}
EXPERIMENTS['few-dummy-test'] = {
    'environments': ['pirl/GridWorld-Simple-v0',
                     'pirl/GridWorld-Simple-Deterministic-v0'],
    'discount': 1.00,
    'rl': 'value_iteration',
    'irl': ['mces', 'mcec', 'mcep_reg0'],
    'num_trajectories': [20],
    'few_shot': [1, 5],
}
EXPERIMENTS['dummy-test-deterministic'] = {
    'environments': ['pirl/GridWorld-Simple-Deterministic-v0'],
    'discount': 1.00,
    'rl': 'value_iteration',
    'irl': ['mces', 'mcep_reg0'],
    'num_trajectories': [20, 10],
}
EXPERIMENTS['dummy-continuous-test'] = {
    'environments': ['airl/TwoDMaze-v0'],
    'discount': 0.99,
    'rl': 'ppo',
    'irl': ['gcls'],
    'num_trajectories': [10, 20],
}

# Jungle gridworld experiments
EXPERIMENTS['jungle'] = {
    'environments': ['pirl/GridWorld-Jungle-9x9-{}-v0'.format(k)
                     for k in ['Soda', 'Water', 'Liquid']],
    'discount': 1.00,
    'rl': 'max_causal_ent',
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
    'rl': 'max_causal_ent',
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
    'rl': 'value_iteration',
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
    'rl': 'max_causal_ent',
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
    'rl': 'max_causal_ent',
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

def validate_config():
    for k, v in EXPERIMENTS.items():
        try:
            gym.envs.registry.spec('pirl/GridWorld-Jungle-4x4-Liquid-v0')
            float(v['discount'])
            RL_ALGORITHMS[v['rl']]
            for irl in v['irl']:
                IRL_ALGORITHMS[irl]
            [int(t) for t in v['num_trajectories']]
            [int(t) for t in v.get('few_shot', [])]
        except Exception as e:
            msg = 'In experiment ' + k + ': ' + str(e)
            raise type(e)(msg).with_traceback(sys.exc_info()[2])
