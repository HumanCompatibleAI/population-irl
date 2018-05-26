import functools
import itertools
import os.path as osp

import tensorflow as tf

from pirl import agents, irl
from pirl.config.types import RLAlgorithm, IRLAlgorithm, MetaIRLAlgorithm

# Overrideable defaults
PROJECT_DIR = osp.dirname(osp.dirname(osp.dirname(osp.realpath(__file__))))
DATA_DIR = osp.join(PROJECT_DIR, 'data')
RAY_SERVER = None # Scheduler IP

try:
    from pirl.config.config_local import *
except ImportError:
    pass

# Directory locations

EXPERIMENTS_DIR = osp.join(DATA_DIR, 'experiments')
OBJECT_DIR = osp.join(DATA_DIR, 'objects')
CACHE_DIR = osp.join(DATA_DIR, 'cache')

# ML Framework Config

def make_tf_config():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return config
TENSORFLOW = make_tf_config()

# Logging
LOG_CFG = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            },
        },
        'handlers': {
            'stream': {
                'level': 'DEBUG',
                'formatter': 'standard',
                'class': 'logging.StreamHandler',
            },
        },
        'loggers': {
            '': {
                'handlers': ['stream'],
                'level': 'DEBUG',
                'propagate': True
            },
        }
    }

# RL Algorithms

# Values take form (gen_policy, sample, compute_value).
#
# gen_policy has signature (env, discount, seed, log_dir), where env is a gym.Env,
# discount is a float, seed is an integer and log_dir is a writable directory.
# They return a policy (algorithm-specific object).
#
# sample has signature (env, policy, num_episodes, seed) where
# num_episodes is the number of trajectories to sample, and seed is used
# to sample deterministically. It returns a list of 3-tuples
# (states, actions, rewards), each of which is a list.
#
# compute_value has signature (env, policy, discount, seed).
# It returns (mean, se) where mean is the estimated reward and se is the
# standard error (0 for exact methods).
RL_ALGORITHMS = {
    'value_iteration': RLAlgorithm(
        train=agents.tabular.policy_env_wrapper(agents.tabular.q_iteration_policy),
        sample=agents.tabular.sample,
        value=agents.tabular.value_in_mdp,
        vectorized=False,
        uses_gpu=False,
    ),
    'max_ent': RLAlgorithm(
        train=agents.tabular.policy_env_wrapper(irl.tabular_maxent.max_ent_policy),
        sample=agents.tabular.sample,
        value=agents.tabular.value_in_mdp,
        vectorized=False,
        uses_gpu=False,
    ),
    'max_causal_ent': RLAlgorithm(
        train=agents.tabular.policy_env_wrapper(irl.tabular_maxent.max_causal_ent_policy),
        sample=agents.tabular.sample,
        value=agents.tabular.value_in_mdp,
        vectorized=False,
        uses_gpu=False,
    ),
}

def ppo_cts_pol(num_timesteps):
    train = functools.partial(agents.ppo.train_continuous,
                              tf_config=TENSORFLOW,
                              num_timesteps=num_timesteps)
    sample = functools.partial(agents.ppo.sample, tf_config=TENSORFLOW)
    value = functools.partial(agents.sample.value, sample)
    return RLAlgorithm(train, sample, value, vectorized=True, uses_gpu=True)
RL_ALGORITHMS['ppo_cts'] = ppo_cts_pol(1e6)
RL_ALGORITHMS['ppo_cts_500k'] = ppo_cts_pol(5e5)
RL_ALGORITHMS['ppo_cts_200k'] = ppo_cts_pol(2e5)
RL_ALGORITHMS['ppo_cts_short'] = ppo_cts_pol(1e5)
RL_ALGORITHMS['ppo_cts_shortest'] = ppo_cts_pol(1e4)

# IRL Algorithms

## Single environment IRL algorithms (not population)

# Values take the form: (irl, reward_wrapper, compute_value).
#
# irl signature (env, trajectories, discount, seed, log_dir) where:
# - env is a gym.Env.
# - trajectories is a dict of environment IDs to lists of trajectories.
# - discount is a float in [0,1].
# - seed is an integer.
# - log_dir is a directory which may be used for logging or other temporary output.
# It returns a tuple (reward, policy), both of which are algorithm-specific
# objects. reward must be comprehensible to RL algorithms (if any) specified in
# the 'eval' key in the experimental config.
#
# reward_wrapper is a class with signature __init__(env, reward).
# It wraps environment (that may be a vector environment) and overrides step()
# to return the reward learnt by the IRL algorithm.
#
# compute_value has signature (env, policy, discount, seed) where:
# - env is a gym.Env.
# - policy is as returned by the IRL algorithm.
# - discount is a float in [0,1].
# - seed is an integer.
# It returns (mean, se) where mean is the estimated reward and se is the
# standard error (0 for exact methods).
SINGLE_IRL_ALGORITHMS = {
    # Maximum Causal Entropy (Ziebart 2010)
    'mce': IRLAlgorithm(
        train=irl.tabular_maxent.irl,
        reward_wrapper=agents.tabular.TabularRewardWrapper,
        value=agents.tabular.value_in_mdp,
        vectorized=False,
        uses_gpu=False,
    ),
    'mce_shortest': IRLAlgorithm(
        train=functools.partial(irl.tabular_maxent.irl, num_iter=500),
        reward_wrapper=agents.tabular.TabularRewardWrapper,
        value=agents.tabular.value_in_mdp,
        vectorized=False,
        uses_gpu=False,
    ),
    # Maximum Entropy (Ziebart 2008)
    'me': IRLAlgorithm(
        train=functools.partial(irl.tabular_maxent.irl,
                                planner=irl.tabular_maxent.max_ent_policy),
        reward_wrapper=agents.tabular.TabularRewardWrapper,
        value=agents.tabular.value_in_mdp,
        vectorized=False,
        uses_gpu=False,
    ),
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
    train = functools.partial(irl.airl.irl, tf_cfg=TENSORFLOW, **kwargs)
    SINGLE_IRL_ALGORITHMS['airl_{}'.format(k)] = IRLAlgorithm(
        train=train,
        reward_wrapper=airl_reward,
        value=airl_value,
        vectorized=True,
        uses_gpu=True,
    )

    for k2, v in {'short': 100, 'shortest': 10}.items():
        kwds = dict(kwargs)
        training_cfg = dict(kwds.get('training_cfg', dict()))
        training_cfg['n_itr'] = v
        kwds['training_cfg'] = training_cfg
        train = functools.partial(irl.airl.irl, tf_cfg=TENSORFLOW, **kwds)
        SINGLE_IRL_ALGORITHMS['airl_{}_{}'.format(k, k2)] = IRLAlgorithm(
            train=train,
            reward_wrapper=airl_reward,
            value=airl_value,
            vectorized=True,
            uses_gpu=True,
        )

## Population IRL algorithms

# Values take the form: (metalearn, finetune, reward_wrapper, compute_value).
#
# metalearn has signature (envs, trajectories, discount, seed, log_dir), where:
# - envs is a dictionary mapping to gym.Env
# - trajectories is a dictionary mapping to trajectories
# - discount, seed and log_dir are as in the single-IRL case.
# It returns an algorithm-specific object.
#
# finetune has signature (metainit, env, trajectories, discount, seed, log_dir),
# where metainit is the return value of metalearn; the remaining arguments and
# the return value are as in the single-IRL case.
#
# reward_wrapper and compute_value are the same as in the single-IRL case.
POPULATION_IRL_ALGORITHMS = dict()
def pop_maxent(**kwargs):
    return MetaIRLAlgorithm(
        metalearn=functools.partial(irl.tabular_maxent.metalearn, **kwargs),
        finetune=functools.partial(irl.tabular_maxent.finetune, **kwargs),
        reward_wrapper=agents.tabular.TabularRewardWrapper,
        value=agents.tabular.value_in_mdp,
        vectorized=False,
        uses_gpu=False,
    )
for reg in range(-4,3):
    algo = pop_maxent(regularize=10**reg)
    POPULATION_IRL_ALGORITHMS['mcep_reg1e{}'.format(reg)] = algo
POPULATION_IRL_ALGORITHMS['mcep_reg0'] = pop_maxent(regularize=0)
POPULATION_IRL_ALGORITHMS['mcep_shortest_reg0'] = pop_maxent(regularize=0,
                                                             num_iter=500)

AIRLP_ALGORITHMS = {
    # 3-tuple with elements:
    # - common
    # - metalearn only
    # - finetune only
    'so': (dict(), dict(), dict()),
    'so_short': (dict(outer_itr=100), dict(), dict()),
    'so_10fine': (dict(), dict(), dict(training_cfg={'n_itr': 10})),
    'so_short_10fine': (dict(outer_itr=100), dict(), dict(training_cfg={'n_itr': 10})),
    'so_shortest': (dict(training_cfg={'n_itr': 2}, outer_itr=2), dict(), dict()),
}
for lr in range(1, 4):
    AIRLP_ALGORITHMS['so_lr1e-{}'.format(lr)] = (dict(lr=10 ** (-lr)), dict(), dict())
for k, (common, meta, fine) in AIRLP_ALGORITHMS.items():
    meta = dict(meta, **common)
    fine = dict(fine, **common)
    metalearn_fn = functools.partial(irl.airl.metalearn, tf_cfg=TENSORFLOW, **meta)
    finetune_fn = functools.partial(irl.airl.finetune, tf_cfg=TENSORFLOW, **fine)
    entry = MetaIRLAlgorithm(metalearn_fn, finetune_fn, airl_reward, airl_value,
                             vectorized=True, uses_gpu=True)
    POPULATION_IRL_ALGORITHMS['airlp_{}'.format(k)] = entry

def traditional_to_concat(singleirl):
    def metalearner(envs, trajectories, discount, seed, log_dir):
        return list(itertools.chain(*trajectories.values()))
    @functools.wraps(singleirl.train)
    def finetune(train_trajectories, envs, test_trajectories, discount, seed, **kwargs):
        concat_trajectories = train_trajectories + test_trajectories
        return singleirl.train(envs, concat_trajectories, discount, seed, **kwargs)
    return MetaIRLAlgorithm(metalearner, finetune,
                            singleirl.reward_wrapper, singleirl.value,
                            singleirl.vectorized, singleirl.uses_gpu)

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
    'irl': ['mce_shortest', 'mcep_shortest_reg0'],
    'train_trajectories': [20, 10],
    'test_trajectories': [20, 10],
    'seeds': 2,
}
EXPERIMENTS['few-dummy-test'] = {
    'train_environments': ['pirl/GridWorld-Simple-v0'],
    'test_environments': ['pirl/GridWorld-Simple-Deterministic-v0'],
    'discount': 1.00,
    'expert': 'value_iteration',
    'eval': ['value_iteration'],
    'irl': ['mce_shortest', 'mce_shortestc', 'mcep_shortest_reg0'],
    'train_trajectories': [20, 10],
    'test_trajectories': [0, 1, 5],
    'seeds': 2,
}
EXPERIMENTS['dummy-test-deterministic'] = {
    'environments': ['pirl/GridWorld-Simple-Deterministic-v0'],
    'discount': 1.00,
    'expert': 'value_iteration',
    'eval': ['value_iteration'],
    'irl': ['mce_shortest', 'mcep_shortest_reg0'],
    'train_trajectories': [20, 10],
    'test_trajectories': [20, 10],
    'seeds': 2,
}
EXPERIMENTS['dummy-continuous-test'] = {
    'environments': ['Reacher-v2'],
    'parallel_rollouts': 4,
    'discount': 0.99,
    'expert': 'ppo_cts_shortest',
    'eval': ['ppo_cts_shortest'],
    'irl': ['airl_so_shortest', 'airl_random_shortest'],
    'test_trajectories': [10, 20],
    'seeds': 2,
}
EXPERIMENTS['few-dummy-continuous-test'] = {
    'train_environments': ['pirl/Reacher-seed{}-0.1-v0'.format(i)
                           for i in range(0, 2)],
    'test_environments': ['pirl/Reacher-seed{}-0.1-v0'.format(i)
                          for i in range(1, 3)],
    'parallel_rollouts': 4,
    'discount': 0.99,
    'expert': 'ppo_cts_shortest',
    'eval': ['ppo_cts_shortest'],
    'irl': ['airl_so_shortest', 'airlp_so_shortest'],
    'train_trajectories': [10, 20],
    'test_trajectories': [10, 20],
    'seeds': 2,
}
EXPERIMENTS['dummy-continuous-test-medium'] = {
    'environments': ['Reacher-v2'],
    'parallel_rollouts': 4,
    'discount': 0.99,
    'expert': 'ppo_cts_short',
    'eval': ['ppo_cts_short'],
    'irl': ['airl_so'],
    'test_trajectories': [10, 100, 1000],
}
EXPERIMENTS['dummy-continuous-test-slow'] = {
    'environments': ['Reacher-v2'],
    'parallel_rollouts': 4,
    'discount': 0.99,
    'expert': 'ppo_cts',
    'eval': ['ppo_cts'],
    'irl': ['airl_so'],
    'test_trajectories': [10, 100, 1000],
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
    'test_trajectories': [200],
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
                'mcep_reg1e-4',
                'mcep_reg1e-3',
                'mcep_reg1e-2',
                'mcep_reg1e-1',
                'mcep_reg1e0',
            ],
            'train_trajectories': [1000],
            'test_trajectories': [0, 1, 2, 5, 10, 20, 50, 100],
        }
EXPERIMENTS['few-jungle-quick-tmp'] = {
    'train_environments': ['pirl/GridWorld-Jungle-9x9-{}-v0'.format(k)
                           for k in jungle_types if k != 'Water'],
    'test_environments': ['pirl/GridWorld-Jungle-9x9-Water-v0'],
    'discount': 1.00,
    'expert': 'max_causal_ent',
    'eval': ['value_iteration'],
    'irl': [
        'mcep_reg1e-1',
    ],
    'train_trajectories': [1000],
    'test_trajectories': [0, 1, 2, 5, 10, 20, 50, 100],
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
    'expert': 'ppo_cts_short',
    'eval': ['ppo_cts_short'],
    'irl': ['airl_so_short', 'airl_random_short'],
    'test_trajectories': [1000],
}
EXPERIMENTS['continuous-reacher'] = {
    'environments': ['Reacher-v2'],
    'parallel_rollouts': 4,
    'discount': 0.99,
    'expert': 'ppo_cts',
    'eval': ['ppo_cts'],
    'irl': ['airl_so', 'airl_random'],
    'test_trajectories': [1000],
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
    'test_trajectories': [1000],
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
    'test_trajectories': [1000],
}
EXPERIMENTS['billiards'] = {
    'environments': ['pirl/Billiards{}-seed{}-v0'.format(n, i)
                     for n in [2,3,4] for i in range(1)],
    'parallel_rollouts': 4,
    'discount': 0.99,
    'expert': 'ppo_cts',
    'eval': [],
    'irl': ['airl_so'],
    'test_trajectories': [1000],
}
EXPERIMENTS['mountain-car-single'] = {
    'environments': ['pirl/MountainCarContinuous-left-0-0.1-v0'],
    'parallel_rollouts': 4,
    'discount': 0.99,
    # simple environment, small number of iterations sufficient to converge
    'expert': 'ppo_cts_short',
    'eval': ['ppo_cts_short'],
    'irl': ['airl_so_short', 'airl_so',
            'airl_sa_short', 'airl_sa',
            'airl_random_short', 'airl_random'],
    'test_trajectories': [1, 2, 5, 100],
}
EXPERIMENTS['mountain-car-vel'] = {
    'environments': ['pirl/MountainCarContinuous-left-{}-{}-v0'.format(vel, initial_noise)
                     for vel in [0, 0.1, 0.5, 1]
                     for initial_noise in [0.1, 0.2, 0.5]
                    ],
    'parallel_rollouts': 4,
    'discount': 0.99,
    # simple environment, small number of iterations sufficient to converge
    'expert': 'ppo_cts_short',
    'eval': [],#'ppo_cts_short'],
    'irl': ['airl_so_short'], #'airl_sa_short', 'airl_random_short'],
    'test_trajectories': [1, 2, 5, 100],
}
EXPERIMENTS['reacher-var'] = {
    'environments': ['pirl/Reacher-seed{}-{}-v0'.format(seed, noise)
                     for seed in range(0,3) for noise in [0.1, 0.5, 1.0]],
    'parallel_rollouts': 4,
    'discount': 0.99,
    # simple environment, small number of iterations sufficient to converge
    'expert': 'ppo_cts_200k',
    'irl': ['airl_so_short'],
    'eval': [],
    'test_trajectories': [1, 2, 5, 100],
}

# Few-shot continuous control
EXPERIMENTS['reacher-metalearning'] = {
    'train_environments': ['pirl/Reacher-seed{}-0.1-v0'.format(seed) for seed in range(0,5)],
    'test_environments': ['pirl/Reacher-seed{}-0.1-v0'.format(seed) for seed in range(5, 10)],
    'parallel_rollouts': 4,
    'discount': 0.99,
    'expert': 'ppo_cts',
    'eval': ['ppo_cts'],
    'irl': ['airl_so'] + ['airlp_so_lr1e-{}'.format(i) for i in range(1,4)],
    'train_trajectories': [1000],
    'test_trajectories': [1, 5, 10, 100],
}
EXPERIMENTS['dummy-reacher-metalearning'] = {
    'train_environments': ['pirl/Reacher-seed{}-0.1-v0'.format(seed) for seed in range(0,2)],
    'test_environments': ['pirl/Reacher-seed{}-0.1-v0'.format(seed) for seed in range(5,6)],
    'parallel_rollouts': 4,
    'discount': 0.99,
    'expert': 'ppo_cts_shortest',
    'eval': ['ppo_cts_shortest'],
    'irl': ['airl_so_shortest', 'airlp_so_shortest'],
    'train_trajectories': [1000],
    'test_trajectories': [5],
}
EXPERIMENTS['mountain-car-meta'] = {
    'environments': ['pirl/MountainCarContinuous-{}-0-0.1-v0'.format(side)
                     for side in ['left', 'right']],
    'parallel_rollouts': 4,
    'discount': 0.99,
    # simple environment, small number of iterations sufficient to converge
    'expert': 'ppo_cts_short',
    'eval': ['ppo_cts_short'],
    'irl': ['airl_so_short', 'airl_sa_short', 'airl_random_short',
            'airlp_so_short', 'airlp_so_short_10fine'],
    'train_trajectories': [100],
    'test_trajectories': [1, 2, 5, 10, 50, 100],
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
        'test_trajectories': [1000],
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
        'irl': ['airl_so_shortest'],
        'test_trajectories': [1000],
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
        'test_trajectories': [1000],
    }
    EXPERIMENTS['parallel-cts-reacher-fast-{}'.format(n)] = {
        'environments': [
            'Reacher-v2',
        ],
        'parallel_rollouts': n,
        'discount': 0.99,
        'expert': 'ppo_cts',
        'eval': [],
        'irl': ['airl_so_shortest'],
        'test_trajectories': [1000],
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
        'test_trajectories': [10],
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
        'test_trajectories': [10],
    }