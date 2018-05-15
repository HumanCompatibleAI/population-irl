import functools
import itertools
import os
import os.path as osp
import sys

import gym
import tensorflow as tf

from pirl import agents, envs, irl

# General
PROJECT_DIR = osp.dirname(osp.dirname(osp.dirname(osp.realpath(__file__))))
DATA_DIR = osp.join(PROJECT_DIR, 'data')
CACHE_DIR = osp.join(PROJECT_DIR, 'cache')
LOG_DIR = osp.join(PROJECT_DIR, 'logs')

try:
    from pirl.experiments.config_local import *
except ImportError:
    pass

# ML Framework Config
def make_tf_config():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return config
TENSORFLOW = make_tf_config()

# Logging
def logging(identifier):
    os.makedirs(LOG_DIR, exist_ok=True)
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
                'filename': '{}/pirl-{}.log'.format(LOG_DIR, identifier),
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
        agents.tabular.value_in_mdp,
    ),
    'max_ent': (
        agents.tabular.env_wrapper(irl.tabular_maxent.max_ent_policy),
        agents.tabular.sample,
        agents.tabular.value_in_mdp,
    ),
    'max_causal_ent': (
        agents.tabular.env_wrapper(irl.tabular_maxent.max_causal_ent_policy),
        agents.tabular.sample,
        agents.tabular.value_in_mdp,
    ),
}

def ppo_cts_pol(num_timesteps):
    return functools.partial(agents.ppo.train_continuous,
                             tf_config=TENSORFLOW,
                             num_timesteps=num_timesteps)
ppo_sample = functools.partial(agents.ppo.sample, tf_config=TENSORFLOW)
ppo_value = functools.partial(agents.sample.value, ppo_sample)
RL_ALGORITHMS['ppo_cts'] = (ppo_cts_pol(1e6), ppo_sample, ppo_value)
RL_ALGORITHMS['ppo_cts_short'] = (ppo_cts_pol(1e5), ppo_sample, ppo_value)
RL_ALGORITHMS['ppo_cts_shortest'] = (ppo_cts_pol(1e4), ppo_sample, ppo_value)


# IRL Algorithms

## Single environment IRL algorithms (not population)

SINGLE_IRL_ALGORITHMS = {
    # Maximum Causal Entropy (Ziebart 2010)
    'mce': (irl.tabular_maxent.irl,
            agents.tabular.TabularRewardWrapper,
            agents.tabular.value_in_mdp),
    'mce_quick': (functools.partial(irl.tabular_maxent.irl, num_iter=500),
                  agents.tabular.TabularRewardWrapper,
                  agents.tabular.value_in_mdp),
    # Maximum Entropy (Ziebart 2008)
    'me': (functools.partial(irl.tabular_maxent.irl, planner=irl.tabular_maxent.max_ent_policy),
           agents.tabular.TabularRewardWrapper,
           agents.tabular.value_in_mdp),
}

AIRL_ALGORITHMS = {
    'so': dict(),
    'sa': dict(training_cfg={'state_only': False}),
    'random': dict(policy_cfg={'policy': irl.airl.GaussianPolicy}),
}
airl_reward = functools.partial(irl.airl.airl_reward_wrapper, tf_cfg=TENSORFLOW)
airl_value = functools.partial(agents.sample.value,
                               functools.partial(irl.airl.sample, tf_cfg=TENSORFLOW))
for k, kwargs in AIRL_ALGORITHMS.items():
    irl_fn = functools.partial(irl.airl.irl, tf_cfg=TENSORFLOW, **kwargs)
    SINGLE_IRL_ALGORITHMS['airl_{}'.format(k)] = (irl_fn, airl_reward, airl_value)

    training_cfg = kwargs.get('training_cfg', {})
    training_cfg['n_itr'] = 10
    kwargs['training_cfg'] = training_cfg
    irl_fn = functools.partial(irl.airl.irl, tf_cfg=TENSORFLOW, **kwargs)
    SINGLE_IRL_ALGORITHMS['airl_{}_quick'.format(k)] = (irl_fn, airl_reward, airl_value)

## Population IRL algorithms

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
POPULATION_IRL_ALGORITHMS = dict()
def pop_maxent(**kwargs):
    return (
        functools.partial(irl.tabular_maxent.metalearn, **kwargs),
        functools.partial(irl.tabular_maxent.finetune, **kwargs),
        agents.tabular.TabularRewardWrapper,
        agents.tabular.value_in_mdp
    )
for reg in range(-2,3):
    algo = pop_maxent(individual_reg = 10**reg)
    POPULATION_IRL_ALGORITHMS['mcep_reg1e{}'.format(reg)] = algo
POPULATION_IRL_ALGORITHMS['mcep_reg0'] = pop_maxent(individual_reg=0)
POPULATION_IRL_ALGORITHMS['mcep_quick_reg0'] = pop_maxent(individual_reg=0, num_iter=500)

AIRLP_ALGORITHMS = {
    'so': dict(),
    'so_quick': dict(training_cfg={'n_itr': 2}, outer_itr=2),
}
for lr in range(1, 4):
    AIRLP_ALGORITHMS['so_lr1e-{}'.format(lr)] = dict(lr=10 ** (-lr))
AIRLP_ALGORITHMS
for k, kwargs in AIRLP_ALGORITHMS.items():
    metalearn_fn = functools.partial(irl.airl.metalearn, tf_cfg=TENSORFLOW, **kwargs)
    finetune_fn = functools.partial(irl.airl.finetune, tf_cfg=TENSORFLOW, **kwargs)
    entry = (metalearn_fn, finetune_fn, airl_reward, airl_value)
    POPULATION_IRL_ALGORITHMS['airlp_{}'.format(k)] = entry

def traditional_to_concat(fs):
    irl_algo, reward_wrapper, compute_value = fs
    def metalearner(env_fns, trajectories, discount, log_dir):
        return list(itertools.chain(*trajectories.values()))
    @functools.wraps(irl_algo)
    def finetune(train_trajectories, env_fns, test_trajectories, **kwargs):
        concat_trajectories = train_trajectories + test_trajectories
        return irl_algo(env_fns, concat_trajectories, **kwargs)
    return metalearner, finetune, reward_wrapper, compute_value

for name, algo in SINGLE_IRL_ALGORITHMS.items():
    POPULATION_IRL_ALGORITHMS[name + 'c'] = traditional_to_concat(algo)

# Experiments

EXPERIMENTS = {}

# ONLY FOR TESTING CODE! Not real experiments.
EXPERIMENTS['dummy-test'] = {
    'environments': ['pirl/GridWorld-Simple-v0'],
    'discount': 1.00,
    'expert': 'value_iteration',
    'eval': ['value_iteration'],
    'irl': ['mcep_quick_reg0', 'mce_quick'],
    'trajectories': [20, 10],
}
EXPERIMENTS['few-dummy-test'] = {
    'train_environments': ['pirl/GridWorld-Simple-v0'],
    'test_environments': ['pirl/GridWorld-Simple-Deterministic-v0'],
    'discount': 1.00,
    'expert': 'value_iteration',
    'eval': ['value_iteration'],
    'irl': ['mce_quick', 'mce_quickc', 'mcep_quick_reg0'],
    'train_trajectories': [20],
    'test_trajectories': [0, 1, 5],
}
EXPERIMENTS['dummy-test-deterministic'] = {
    'environments': ['pirl/GridWorld-Simple-Deterministic-v0'],
    'discount': 1.00,
    'expert': 'value_iteration',
    'eval': ['value_iteration'],
    'irl': ['mce_quick', 'mcep_quick_reg0'],
    'trajectories': [20, 10],
}
EXPERIMENTS['dummy-continuous-test'] = {
    'environments': ['Reacher-v2'],
    'parallel_rollouts': 4,
    'discount': 0.99,
    'expert': 'ppo_cts_shortest',
    'eval': ['ppo_cts_shortest'],
    'irl': ['airl_so_quick', 'airl_sa_quick', 'airl_random_quick'],
    'trajectories': [10, 20],
}
EXPERIMENTS['dummy-continuous-test-medium'] = {
    'environments': ['Reacher-v2'],
    'parallel_rollouts': 4,
    'discount': 0.99,
    'expert': 'ppo_cts_short',
    'eval': ['ppo_cts_short'],
    'irl': ['airl_so'],
    'trajectories': [10, 100, 1000],
}
EXPERIMENTS['dummy-continuous-test-slow'] = {
    'environments': ['Reacher-v2'],
    'parallel_rollouts': 4,
    'discount': 0.99,
    'expert': 'ppo_cts',
    'eval': ['ppo_cts'],
    'irl': ['airl_so'],
    'trajectories': [10, 100, 1000],
}

# Jungle gridworld experiments
EXPERIMENTS['jungle'] = {
    'environments': ['pirl/GridWorld-Jungle-9x9-{}-v0'.format(k)
                     for k in ['Soda', 'Water', 'Liquid']],
    'discount': 1.00,
    'expert': 'max_causal_ent',
    'eval': ['value_iteration'],
    'irl': [
        'mce',
        'mcec',
        'mcep_reg0',
        'mcep_reg1e-2',
        'mcep_reg1e-1',
        'mcep_reg1e0',
    ],
    'trajectories': [1000, 500, 200, 100, 50, 30, 20, 10, 5],
}
EXPERIMENTS['jungle-small'] = {
    'environments': ['pirl/GridWorld-Jungle-4x4-{}-v0'.format(k)
                     for k in ['Soda', 'Water', 'Liquid']],
    'discount': 1.00,
    'expert': 'max_causal_ent',
    'eval': ['value_iteration'],
    'irl': [
        'mce',
        'mcec',
        'mcep_reg0',
        'mcep_reg1e-2',
        'mcep_reg1e-1',
        'mcep_reg1e0',
    ],
    'trajectories': [500, 200, 100, 50, 30, 20, 10, 5],
}

# Test different planner combinations
EXPERIMENTS['unexpected-optimal'] = {
    'environments': ['pirl/GridWorld-Jungle-4x4-Soda-v0'],
    'discount': 1.00,
    'expert': 'max_causal_ent',
    'eval': ['value_iteration'],
    'irl': [
        'mce',
        'me',
    ],
    'trajectories': [200],
}

# Few-shot learning
jungle_types = ['Soda', 'Water', 'Liquid']
for shape in ['9x9', '4x4']:
    for few_shot in jungle_types:
        EXPERIMENTS['few-jungle-{}-{}'.format(shape, few_shot)] = {
            'train_environments': ['pirl/GridWorld-Jungle-{}-{}-v0'.format(shape, k)
                                   for k in jungle_types if k != few_shot],
            'test_environments': ['pirl/GridWorld-Jungle-{}-{}-v0'.format(shape, few_shot)],
            'discount': 1.00,
            'expert': 'max_causal_ent',
            'eval': ['value_iteration'],
            'irl': [
                'mce',
                'mcec',
                'mcep_reg0',
                'mcep_reg1e-2',
                'mcep_reg1e-1',
                'mcep_reg1e0',
            ],
            'trajectories': [1000],
            'few_shot': [0, 1, 2, 5, 10, 20, 50, 100],
        }

# Continuous control
EXPERIMENTS['continuous-baselines-classic'] = {
    # continuous state space but (mostly) discrete action spaces
    'environments': [
        'MountainCarContinuous-v0',
        # below are discrete which AIRL cannot currently work with
        # 'Acrobot-v1',
        # 'CartPole-v1',
        # 'MountainCar-v0',
        # 'Pendulum-v0',
    ],
    'parallel_rollouts': 4,
    'discount': 0.99,
    'expert': 'ppo_cts',
    'eval': ['ppo_cts'],
    'irl': ['airl_so', 'airl_random'],
    'trajectories': [1000],
}
EXPERIMENTS['continuous-baselines-easy'] = {
    'environments': [
        'Reacher-v2',
        'InvertedPendulum-v2',
        'InvertedDoublePendulum-v2'
    ],
    'parallel_rollouts': 4,
    'discount': 0.99,
    'expert': 'ppo_cts',
    'eval': ['ppo_cts'],
    'irl': ['airl_so', 'airl_random'],
    'trajectories': [1000],
}
EXPERIMENTS['continuous-baselines-medium'] = {
    'environments': [
        'Swimmer-v2',
        'Hopper-v2',
        'HalfCheetah-v2',
    ],
    'parallel_rollouts': 4,
    'discount': 0.99,
    'expert': 'ppo_cts',
    'eval': ['ppo_cts'],
    'irl': ['airl_so', 'airl_random'],
    'trajectories': [1000],
}
EXPERIMENTS['billiards'] = {
    'environments': ['pirl/Billiards{}-seed{}-v0'.format(n, i)
                     for n in [2,3,4] for i in range(1)],
    'parallel_rollouts': 4,
    'discount': 0.99,
    'expert': 'ppo_cts',
    'eval': [],
    'irl': ['airl_so'],
    'trajectories': [1000],
}
EXPERIMENTS['reacher-env-comparisons'] = {
    'environments': ['Reacher-v2', 'pirl/Reacher-baseline-seed0-v0',
                     'pirl/Reacher-variable-hidden-goal-seed0-v0',
                     'pirl/Reacher-fixed-hidden-goal-seed0-v0'],
    'parallel_rollouts': 4,
    'discount': 0.99,
    'expert': 'ppo_cts',
    'eval': ['ppo_cts'],
    'irl': [],
    'trajectories': [1000],
}

# Few-shot continuous control
EXPERIMENTS['reacher-metalearning'] = {
    'train_environments': ['pirl/Reacher-fixed-hidden-goal-seed{}-v0'.format(seed) for seed in range(0,5)],
    'test_environments': ['pirl/Reacher-fixed-hidden-goal-seed{}-v0'.format(seed) for seed in range(5, 10)],
    'parallel_rollouts': 4,
    'discount': 0.99,
    'expert': 'ppo_cts',
    'eval': ['ppo_cts'],
    'irl': ['airl_so'] + ['airlp_so_lr1e-{}'.format(i) for i in range(1,4)],
    'train_trajectories': [1000],
    'test_trajectories': [1, 5, 10, 100],
}
EXPERIMENTS['dummy-reacher-metalearning'] = {
    'train_environments': ['pirl/Reacher-fixed-hidden-goal-seed{}-v0'.format(seed) for seed in range(0,2)],
    'test_environments': ['pirl/Reacher-fixed-hidden-goal-seed{}-v0'.format(seed) for seed in range(5,6)],
    'parallel_rollouts': 4,
    'discount': 0.99,
    'expert': 'ppo_cts_shortest',
    'eval': ['ppo_cts_shortest'],
    'irl': ['airl_so_quick', 'airlp_so_quick'],
    'train_trajectories': [1000],
    'test_trajectories': [5],
}

# Test of RL parallelism
for n in [1, 4, 8, 16]:
    EXPERIMENTS['parallel-cts-easy-{}'.format(n)] = {
        'environments': [
            'Reacher-v2',
            'InvertedPendulum-v2',
            'InvertedDoublePendulum-v2'
        ],
        'parallel_rollouts': n,
        'discount': 0.99,
        'expert': 'ppo_cts',
        'eval': [],#['ppo_cts'],
        'irl': ['airl_so'],
        'trajectories': [1000],
    }
    EXPERIMENTS['parallel-cts-easy-fast-{}'.format(n)] = {
        'environments': [
            'Reacher-v2',
            'InvertedPendulum-v2',
            'InvertedDoublePendulum-v2'
        ],
        'parallel_rollouts': n,
        'discount': 0.99,
        'expert': 'ppo_cts',
        'eval': [],
        'irl': ['airl_so_quick'],
        'trajectories': [1000],
    }
    EXPERIMENTS['parallel-cts-reacher-{}'.format(n)] = {
        'environments': [
            'Reacher-v2',
        ],
        'parallel_rollouts': n,
        'discount': 0.99,
        'expert': 'ppo_cts',
        'eval': [],
        'irl': ['airl_so'],
        'trajectories': [1000],
    }
    EXPERIMENTS['parallel-cts-reacher-fast-{}'.format(n)] = {
        'environments': [
            'Reacher-v2',
        ],
        'parallel_rollouts': n,
        'discount': 0.99,
        'expert': 'ppo_cts',
        'eval': [],
        'irl': ['airl_so_quick'],
        'trajectories': [1000],
    }
    EXPERIMENTS['parallel-cts-reacher-fast-rl-{}'.format(n)] = {
        'environments': [
            'Reacher-v2',
        ],
        'parallel_rollouts': n,
        'discount': 0.99,
        'expert': 'ppo_cts_shortest',
        'eval': [],
        'irl': [],
        'trajectories': [10],
    }
    EXPERIMENTS['parallel-cts-humanoid-fast-rl-{}'.format(n)] = {
        'environments': [
            'Humanoid-v2',
        ],
        'parallel_rollouts': n,
        'discount': 0.99,
        'expert': 'ppo_cts_shortest',
        'eval': [],
        'irl': [],
        'trajectories': [10],
    }


def validate_config():
    for k, v in EXPERIMENTS.items():
        try:
            for env in v.get('train_environments', v.get('environments')):
                gym.envs.registry.spec(env)
            for env in v.get('test_environments', v.get('environments')):
                gym.envs.registry.spec(env)
            float(v['discount'])
            RL_ALGORITHMS[v['expert']]
            for rl in v.get(eval, []):
                RL_ALGORITHMS[rl]
            intersect = set(POPULATION_IRL_ALGORITHMS.keys()).intersection(SINGLE_IRL_ALGORITHMS.keys())
            for irl, algo in SINGLE_IRL_ALGORITHMS.items():
                assert len(algo) == 3, k
            for irl, algo in POPULATION_IRL_ALGORITHMS.items():
                assert len(algo) == 4, k
            assert len(intersect) == 0
            for irl in v['irl']:
                assert (irl in POPULATION_IRL_ALGORITHMS or irl in SINGLE_IRL_ALGORITHMS)
            [int(t) for t in v.get('train_trajectories', v.get('trajectories'))]
            [int(t) for t in v.get('test_trajectories', v.get('trajectories'))]
        except Exception as e:
            msg = 'In experiment ' + k + ': ' + str(e)
            raise type(e)(msg).with_traceback(sys.exc_info()[2])
