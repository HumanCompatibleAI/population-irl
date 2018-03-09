import functools
import itertools

import torch

from pirl import agents, irl

# Experiments
EXPERIMENTS = {
    # ONLY FOR TESTING CODE! Not real experiments.
    'dummy-test': {
        'environments': ['pirl/GridWorld-Simple-v0'],
        'discount': 1.00,
        'rl': 'value_iteration',
        'irl': ['mes', 'mep_orig'],
        'num_trajectories': [100, 10],
    },
    'dummy-test-deterministic': {
        'environments': ['pirl/GridWorld-Simple-Deterministic-v0'],
        'discount': 1.00,
        'rl': 'value_iteration',
        'irl': ['mes', 'mep_orig'],
        'num_trajectories': [100, 10],
    },
    # Real experiments below
    'jungle': {
        'environments': ['pirl/GridWorld-Jungle-9x9-{}-v0'.format(k)
                         for k in ['Soda', 'Water', 'Liquid']],
        'discount': 1.00,
        'rl': 'value_iteration',
        'irl': [
            'mep_orig_scale1_reg0',
            # 'mep_orig_scale1_reg0.1',
            # 'mep_orig_scale1_reg1',
            # 'mep_orig_scale2_reg0',
            # 'mep_orig_scale2_reg0.1',
            # 'mep_orig_scale2_reg1',
            # 'mep_demean',
            'mes',
        ],
        'num_trajectories': [200, 100, 50, 30, 20, 10],
    },
    'jungle-small': {
        'environments': ['pirl/GridWorld-Jungle-4x4-{}-v0'.format(k)
                         for k in ['Soda', 'Water', 'Liquid']],
        'discount': 1.00,
        'rl': 'value_iteration',
        'irl': [
            'mep_orig_scale1_reg0',
            'mes',
        ],
        'num_trajectories': [200, 100, 50, 30, 20, 10],
    },
    'jungle-small-maxent': {
        'environments': ['pirl/GridWorld-Jungle-4x4-{}-v0'.format(k)
                         for k in ['Soda', 'Water', 'Liquid']],
        'discount': 1.00,
        'rl': 'max_ent',
        'irl': [
            'mep_orig_scale1_reg0',
            'mes',
        ],
        'num_trajectories': [200, 100, 50, 30, 20, 10],
    },
    'jungle-small-maxcausalent': {
        'environments': ['pirl/GridWorld-Jungle-4x4-{}-v0'.format(k)
                         for k in ['Soda', 'Water', 'Liquid']],
        'discount': 1.00,
        'rl': 'max_causal_ent',
        'irl': [
            'mep_orig_scale1_reg0',
            'mes',
        ],
        'num_trajectories': [200, 100, 50, 30, 20, 10],
    },
}

# RL Algorithms
RL_ALGORITHMS = {
    # Values take form (gen_policy, compute_value).
    # Both functions take a gym.Env as their single argument.
    # compute_value moreover takes as the second argument a return value from
    # gen_policy, which may have been computed on an environment with a
    # different reward.
    'value_iteration': (
        agents.tabular.env_wrapper(agents.tabular.q_iteration_policy),
        agents.tabular.value_of_policy,
    ),
    'max_ent': (
        agents.tabular.env_wrapper(irl.tabular_maxent.max_ent_policy),
        agents.tabular.value_of_policy,
    ),
    'max_causal_ent': (
        agents.tabular.env_wrapper(irl.tabular_maxent.max_causal_ent_policy),
        agents.tabular.value_of_policy,
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
        concat_trajectories = itertools.chain(*trajectories.values())
        # Pick an environment arbitrarily. In the typical use case,
        # they are all the same up to reward anyway.
        env = list(envs.values())[0]
        value, info = f(env, concat_trajectories, **kwargs)
        value = {k: value for k in trajectories.keys()}
        return value, info
    return helper


TRADITIONAL_IRL_ALGORITHMS = {
    'me': functools.partial(irl.tabular_maxent.irl, discount=0.99),
}

# demean vs non demean
# without demeaning, change scale, regularization
MY_IRL_ALGORITHMS = dict()
for reg, scale in itertools.product([0, 1e-1, 1], [1, 2]):
    fn = functools.partial(irl.tabular_maxent.population_irl,
                           discount=0.99,
                           demean=False,
                           common_scale=scale,
                           individual_reg=reg)
    MY_IRL_ALGORITHMS['mep_orig_scale{}_reg{}'.format(scale, reg)] = fn
MY_IRL_ALGORITHMS['mep_orig'] = MY_IRL_ALGORITHMS['mep_orig_scale1_reg0']
MY_IRL_ALGORITHMS['mep_demean'] = functools.partial(irl.tabular_maxent.population_irl,
                                                    discount=0.99)

adam_optim = functools.partial(torch.optim.Adam, lr=1e-2)
MY_IRL_ALGORITHMS['mep_orig_adam'] = functools.partial(irl.tabular_maxent.population_irl,
                                                       discount=0.99,
                                                       demean=False,
                                                       optimizer=adam_optim)

IRL_ALGORITHMS = dict()
IRL_ALGORITHMS.update(MY_IRL_ALGORITHMS)
for name, algo in TRADITIONAL_IRL_ALGORITHMS.items():
    IRL_ALGORITHMS[name + 's'] = traditional_to_single(algo)
    IRL_ALGORITHMS[name + 'c'] = traditional_to_concat(algo)