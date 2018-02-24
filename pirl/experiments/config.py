import functools
import itertools

from pirl import agents, irl

# Experiments
EXPERIMENTS = {
    # ONLY FOR TESTING CODE! Not real experiments.
    'dummy-test': {
        #TODO: change to be a fixed environment and allow overloading
        #reward? This is closer to the semantic meaning we want to express,
        #but is awkward to go with Gym's abstraction.
        'environments': ['pirl/GridWorld-Simple-v0'],
        'rl': 'value_iteration',
        'irl': ['max_ent_single', 'max_ent_population'],
        'num_trajectories': 100,
    },
    'dummy-test-deterministic': {
        'environments': ['pirl/GridWorld-Simple-Deterministic-v0'],
        'rl': 'value_iteration',
        'irl': ['max_ent_single'],
        'num_trajectories': 100,
    },
    # Real experiments below
    'jungle': {
        'environments': ['pirl/GridWorld-Jungle-{}-v0'.format(k)
                         for k in ['Soda', 'Water', 'Liquid']],
        'rl': 'value_iteration',
        'irl': ['max_ent_single', 'max_ent_population'],
        'num_trajectories': 200,
    },
}

# RL Algorithms
RL_ALGORITHMS = {
    # Values take form (gen_policy, compute_value).
    # Both functions take a gym.Env as their single argument.
    # compute_value moreover takes as the second argument a return value from
    # gen_policy, which may have been computed on an environment with a
    # different reward.
    'value_iteration': (lambda env: agents.tabular.value_iteration(env)[0],
                        agents.tabular.value_of_policy)
}

# IRL Algorithms
def traditional_to_single(f):
    @functools.wraps(f)
    def helper(envs, trajectories, **kwargs):
        #TODO: parallelize
        return {k: f(envs[k], v, **kwargs) for k, v in trajectories.items()}
    return helper


def traditional_to_concat(f):
    @functools.wraps(f)
    def helper(envs, trajectories, **kwargs):
        concat_trajectories = itertools.chain(*trajectories.values())
        # Pick an environment arbitrarily. In the typical use case,
        # they are all the same up to reward anyway.
        env = list(envs.values())[0]
        res = f(env, concat_trajectories, **kwargs)
        return {k: res for k in trajectories.keys()}
    return helper


TRADITIONAL_IRL_ALGORITHMS = {
    'max_ent': functools.partial(irl.tabular_maxent.maxent_irl, discount=0.99),
}

MY_IRL_ALGORITHMS = {
    'max_ent_population': functools.partial(irl.tabular_maxent.maxent_population_irl,
                                            discount=0.99),
}

IRL_ALGORITHMS = dict()
IRL_ALGORITHMS.update(MY_IRL_ALGORITHMS)
for name, algo in TRADITIONAL_IRL_ALGORITHMS.items():
    IRL_ALGORITHMS[name + '_single'] = traditional_to_single(algo)
    IRL_ALGORITHMS[name + '_concat'] = traditional_to_concat(algo)