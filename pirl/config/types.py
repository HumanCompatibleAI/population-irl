from collections import namedtuple
import sys

import gym

# Algorithms
RES_FLDS = ['sample', 'vectorized', 'uses_gpu']
RES_FLDS_DOC = '''\n
sample has signature (env, policy, num_episodes, seed) where
num_episodes is the number of trajectories to sample, and seed is used
to sample deterministically. It returns a list of 3-tuples
(states, actions, rewards), each of which is a list.

vectorized is a boolean flag indicating if the algorithm takes VecEnv's.

uses_gpu is a boolean flag indicating whether the algorithm requires a GPU.
'''

RLAlgorithm = namedtuple('RLAlgorithm', ['train', 'value'] + RES_FLDS)
RLAlgorithm.__doc__ = '''\
train has signature (env, discount, seed, log_dir), where env is a gym.Env,
discount is a float, seed is an integer and log_dir is a writable directory.
They return a policy (algorithm-specific object).

value has signature (env, policy, discount, seed).
It returns (mean, se) where mean is the estimated reward and se is the
standard error (0 for exact methods).''' + RES_FLDS_DOC
IRLAlgorithm = namedtuple('IRLAlgorithm',
                          ['train', 'reward_wrapper', 'value'] + RES_FLDS)
IRLAlgorithm.__doc__ = '''\
train signature (env, trajectories, discount, seed, log_dir) where:
- env is a gym.Env.
- trajectories is a dict of environment IDs to lists of trajectories.
- discount is a float in [0,1].
- seed is an integer.
- log_dir is a directory which may be used for logging or other temporary output.
It returns a tuple (reward, policy), both of which are algorithm-specific
objects. reward must be comprehensible to RL algorithms (if any) specified in
the 'eval' key in the experimental config.

reward_wrapper is a class with signature __init__(env, reward).
It wraps environment (that may be a vector environment) and overrides step()
to return the reward learnt by the IRL algorithm.

value has signature (env, policy, discount, seed) where:
- env is a gym.Env.
- policy is as returned by the IRL algorithm.
- discount is a float in [0,1].
- seed is an integer.
It returns (mean, se) where mean is the estimated reward and se is the
standard error (0 for exact methods).''' + RES_FLDS_DOC
MetaIRLAlgorithm = namedtuple('MetaIRLAlgorithm',
                              ['metalearn', 'finetune',
                               'reward_wrapper', 'value'] + RES_FLDS)
MetaIRLAlgorithm.__doc__ = '''\
Values take the form: (metalearn, finetune, reward_wrapper, compute_value).

metalearn has signature (envs, trajectories, discount, seed, log_dir), where:
- envs is a dictionary mapping to gym.Env
- trajectories is a dictionary mapping to trajectories
- discount, seed and log_dir are as in the single-IRL case.
It returns an algorithm-specific object.

finetune has signature (metainit, env, trajectories, discount, seed, log_dir),
where metainit is the return value of metalearn; the remaining arguments and
the return value are as in the single-IRL case.

reward_wrapper and compute_value are the same as for IRLAlgorithm.'''

def validate_config(rl_algos, single_irl_algos, population_irl_algos):
    '''Checks the defined algorithms are of the appropriate type,
       and there is no ambiguity based on their keys (for single v.s.
       population IRL algorithms.)'''
    # Check algorithms
    pop_keys = set(population_irl_algos.keys())
    intersection = pop_keys.intersection(single_irl_algos.keys())
    assert len(intersection) == 0

    for rl, algo in rl_algos.items():
        assert isinstance(algo, RLAlgorithm), rl
    for irl, algo in single_irl_algos.items():
        assert isinstance(algo, IRLAlgorithm), irl
    for irl, algo in population_irl_algos.items():
        assert isinstance(algo, MetaIRLAlgorithm), irl

# Per-experiment configuration

def _list_of(converter):
    def f(xs):
        return list(map(converter, xs))
    return f

# All fields in the parsed configuration.
# 'key': type_converter
FIELD_TYPES = {
    # (Lists of) algorithms
    'expert': str,
    'irl': _list_of(str),
    'eval': _list_of(str),
    # Lists of environments
    'train_environments': _list_of(str),
    'test_environments': _list_of(str),
    # Lists of trajectories
    'train_trajectories': _list_of(int),
    'test_trajectories': _list_of(int),
    'discount': float,
    # Number of seeds to use
    'seeds': int,
    'parallel_rollouts': int,
}

MANDATORY_FIELDS = ['expert', 'irl', 'eval', 'test_trajectories']
OPTIONAL_FIELDS = {
    # 'field': default_value
    'discount': 0.99,
    'seeds': 3,
    'train_trajectories': None,
    # note parallel_rollouts is ignored for non-vectorized (I)RL algorithms
    'parallel_rollouts': 4,
}

def parse_config(experiment, cfg,
                 rl_algos, single_irl_algos, population_irl_algos):
    '''Returns a canonical configuration from user-specified configuration
       dictionary cfg. Fills in defaults from OPTIONAL_FIELDS, verifies all
       MANDATORY_FIELDS are present, type checks in FIELD_TYPES, and performs
       some additional custom validation.'''
    try:
        # Fill in defaults
        res = {}
        for k in MANDATORY_FIELDS:
            res[k] = cfg[k]
        for k, default in OPTIONAL_FIELDS.items():
            v = cfg.get(k, default)
            if v is not None:
                res[k] = v
        for k in ['train_environments', 'test_environments']:
            if 'environments' in cfg:
                assert k not in cfg
                res[k] = cfg['environments']
            else:
                res[k] = cfg[k]

        # Type checking/conversion
        for fld, converter in FIELD_TYPES.items():
            if fld in res:
                res[fld] = converter(res[fld])

        # Check environments are registered
        for fld in ['train_environments', 'test_environments']:
            for env in res[fld]:
                gym.envs.registry.spec(env)

        # Check RL & IRL algorithms are registered
        rl_algos[res['expert']]
        for rl in res['eval']:
            rl_algos[rl]
        for irl in res['irl']:
            assert (irl in population_irl_algos or irl in single_irl_algos)

        # train_trajectories only makes sense with a meta-IRL algorithm
        meta_irl = any([irl in population_irl_algos for irl in res['irl']])
        assert ('train_trajectories' in res) == meta_irl

        return res
    except Exception as e:
        fstr = "Error parsing config for experiment '{}': {}"
        msg = fstr.format(experiment, str(e))
        raise type(e)(msg).with_traceback(sys.exc_info()[2])