import numpy as np
from gym.envs.registration import register

from pirl.envs import gridworld, tabular_mdp_env
from pirl.envs.gridworld import GridWorldMdp
from pirl.envs.tabular_mdp_env import TabularMdpEnv

### Gridworlds

## Only intended for testing code, entirely unrealistic
register(
    id='pirl/GridWorld-Simple-v0',
    entry_point='pirl.envs:GridWorldMdp.from_string',
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
    entry_point='pirl.envs:GridWorldMdp.from_string',
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
jungle_topology = [
    'AAAX  W  ',
    ' R X   S ',
    ' R X XXL ',
    ' R    LL ',
    ' RRRRRRR ',
    ' R    LL ',
    ' R X XXL ',
    ' R X   W ',
    'AAAX  S  ',
]
jungle_default_reward = -1
jungle_topology = np.array([list(x) for x in jungle_topology])
cfg = {'Soda': ['S'], 'Water': ['W'], 'Liquid': ['S', 'W']}
for kind, cells in cfg.items():
    reward_map = {'R': 0, 'L': -10}
    for k in cells:
        reward_map[k] = 1
    fn = np.vectorize(lambda x: reward_map.get(x, jungle_default_reward))
    reward = fn(jungle_topology)
    register(
        id='pirl/GridWorld-Jungle-{}-v0'.format(kind),
        entry_point='pirl.envs:GridWorldMdp',
        max_episode_steps=100,
        kwargs={
            'walls': jungle_topology == 'X',
            'reward': reward,
            'initial_state': gridworld.create_initial_state(jungle_topology),
            'terminal': np.zeros_like(jungle_topology, dtype=bool),
            'noise': 0.2,
        }
    )