from collections import namedtuple
import sys

import gym

# Algorithms
RES_FLDS = ['vectorized', 'uses_gpu']
RLAlgorithm = namedtuple('RLAlgorithm', ['train', 'sample', 'value'] + RES_FLDS)
IRLAlgorithm = namedtuple('IRLAlgorithm',
                          ['train', 'reward_wrapper', 'value'] + RES_FLDS)
MetaIRLAlgorithm = namedtuple('MetaIRLAlgorithm',
                              ['metalearn', 'finetune',
                               'reward_wrapper', 'value'] + RES_FLDS)

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
}

MANDATORY_FIELDS = ['discount', 'expert', 'irl', 'eval', 'test_trajectories']
OPTIONAL_FIELDS = {
    # 'field': default_value
    'train_trajectories': None,
    'seeds': 5,
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