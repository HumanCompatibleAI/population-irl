import numpy as np
from gym.envs.registration import register

from pirl.envs import gridworld, tabular_mdp
from pirl.envs.gridworld import GridWorldMdpEnv
from pirl.envs.tabular_mdp import TabularMdpEnv
from pirl.envs.mountain_car import ContinuousMountainCarPopulationEnv
from pirl.envs.reacher import ReacherPopulationEnv
from pirl.envs.billiards import BilliardsEnv
from pirl.envs.seaquest import SeaquestPopulationEnv

### Gridworlds

## Only intended for testing code, entirely unrealistic
register(
    id='pirl/GridWorld-Simple-v0',
    entry_point='pirl.envs:GridWorldMdpEnv.from_string',
    max_episode_steps=100,
    kwargs={
        'grid': ['A X1',
                 '    ',
                 ' 1X ',
                 'A X9'],
        'default_reward': 0.1,
        'noise': 0.2,
    },
)

register(
    id='pirl/GridWorld-Simple-Deterministic-v0',
    entry_point='pirl.envs:GridWorldMdpEnv.from_string',
    max_episode_steps=100,
    kwargs={
        'grid': ['A X1',
                 '    ',
                 ' 1X ',
                 'A X9'],
        'default_reward': 0.1,
        'noise': 0.0,
    },
)

## Jungle
# Key: A = initial state, X = wall, R = road, L = lava, S = soda, W = water
# Rewards: default = -1, road = 0, lava = -10
# Rewards: soda = 1 or 0, water = 1 or 0 (depending on agent preference)
jungle_topology = {
    '9x9': [
        'AAAX  W  ',
        ' R X   S ',
        ' R X XXL ',
        ' R    LL ',
        ' RRRRRRR ',
        ' R    LL ',
        ' R X XXL ',
        ' R X   W ',
        'AAAX  S  ',
    ],
    '4x4': [
        'ARLW',
        'RRR ',
        'RRR ',
        'ARLS',
    ],
}
jungle_default_reward = -1
jungle_topology = {k: np.array([list(x) for x in v])
                   for k, v in jungle_topology.items()}
cfg = {'Soda': ['S'], 'Water': ['W'], 'Liquid': ['S', 'W']}
for kind, cells in cfg.items():
    reward_map = {'R': 0, 'L': -10}
    for k in cells:
        reward_map[k] = 1
    fn = np.vectorize(lambda x: reward_map.get(x, jungle_default_reward))
    for scale, topology in jungle_topology.items():
        reward = fn(topology)
        register(
            id='pirl/GridWorld-Jungle-{}-{}-v0'.format(scale, kind),
            entry_point='pirl.envs:GridWorldMdpEnv',
            max_episode_steps=100,
            kwargs={
                'walls': topology == 'X',
                'reward': reward,
                'initial_state': gridworld.create_initial_state(topology),
                'terminal': np.zeros_like(topology, dtype=bool),
                'noise': 0.2,
            }
        )

## MountainCar
for name, sign in {'left': -1, 'right': 1}.items():
    for vel_penalty in [0, 0.1, 0.5, 1]:
        register(
            id='pirl/MountainCarContinuous-{}-{}-v0'.format(name, vel_penalty),
            entry_point='pirl.envs:ContinuousMountainCarPopulationEnv',
            max_episode_steps=999,
            reward_threshold=90.0,
            kwargs={'side': sign, 'vel_penalty': vel_penalty},
        )

## Reacher
for seed in range(10):
    register(
        id='pirl/Reacher-baseline-seed{}-v0'.format(seed),
        entry_point='pirl.envs:ReacherPopulationEnv',
        max_episode_steps=50,
        kwargs={
            'seed': seed,
            'start_variance': 0.1,
            'goal_state_pos': 'variable',
            'goal_state_access': True,
        }
    )

    register(
        id='pirl/Reacher-variable-hidden-goal-seed{}-v0'.format(seed),
        entry_point='pirl.envs:ReacherPopulationEnv',
        max_episode_steps=50,
        kwargs={
            'seed': seed,
            'start_variance': 0.1,
            'goal_state_pos': 'variable',
            'goal_state_access': False,
        },
    )

    register(
        id='pirl/Reacher-fixed-hidden-goal-seed{}-v0'.format(seed),
        entry_point='pirl.envs:ReacherPopulationEnv',
        max_episode_steps=50,
        kwargs={
            'seed': seed,
            'start_variance': 0.1,
            'goal_state_pos': 'fixed',
            'goal_state_access': False,
        },
    )

## Billiards
billiard_params = [
    (0, 1),
    (1, 1),
    (5, 2),
    (-10, 1)
]
for seed in range(10):
    for num_balls in range(1, len(billiard_params) + 1):
        register(
            id='pirl/Billiards{}-seed{}-v0'.format(num_balls, seed),
            entry_point='pirl.envs:BilliardsEnv',
            max_episode_steps=200,
            kwargs={
                'params': billiard_params,
                'num_balls': num_balls,
                'seed': seed,
            },
        )

## Seaquest
register(
    id='pirl/SeaquestPopulation-v0',
    entry_point='pirl.envs:SeaquestPopulationEnv',
    max_episode_steps=100000,
    kwargs={},
)