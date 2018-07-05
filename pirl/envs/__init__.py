import numpy as np
from gym.envs.registration import register
from pirl.envs import gridworld

### Gridworlds

## Only intended for testing code, entirely unrealistic
register(
    id='pirl/GridWorld-Simple-v0',
    entry_point='pirl.envs.gridworld:GridWorldMdpEnv.from_string',
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
    entry_point='pirl.envs.gridworld:GridWorldMdpEnv.from_string',
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
            entry_point='pirl.envs.gridworld:GridWorldMdpEnv',
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
for num_peaks in [2, 3, 4]:
    for vel_penalty in [0, 0.1, 0.5, 1]:
        for initial_noise in [0.05, 0.1, 0.25]:
            GOAL_POS = {'left': [0.01], 'right': [num_peaks - 1.01], 'random': None}
            for side, pos in GOAL_POS.items():
                register(
                    id='pirl/MountainCarContinuous-{}-{}-{}-{}-v0'.format(
                        num_peaks, side, vel_penalty, initial_noise),
                    entry_point='pirl.envs.mountain_car:ContinuousMountainCarPopulationEnv',
                    max_episode_steps=999,
                    reward_threshold=90.0,
                    kwargs={
                        'num_peaks': num_peaks,
                        'goal_reward': [100],
                        'goal_position': pos,
                        'vel_penalty': vel_penalty,
                        'initial_noise': initial_noise
                    },
                )

            TWO_FIXED_POS = {
                'left-target': [100, -100],
                'right-target': [-100, 100]
            }
            for side, reward in TWO_FIXED_POS.items():
                register(
                    id='pirl/MountainCarContinuous-{}-{}-{}-{}-v0'.format(
                        num_peaks, side, vel_penalty, initial_noise),
                    entry_point='pirl.envs.mountain_car:ContinuousMountainCarPopulationEnv',
                    max_episode_steps=999,
                    reward_threshold=90.0,
                    kwargs={
                        'num_peaks': num_peaks,
                        'goal_reward': reward,
                        'goal_position': [0.01, num_peaks - 1.01],
                        'vel_penalty': vel_penalty,
                        'initial_noise': initial_noise
                    },
                )

            TWO_VARIABLE_POS = {'red': [100, -100], 'blue': [-100, 100]}
            for good_goal, reward in TWO_VARIABLE_POS.items():
                register(
                    id='pirl/MountainCarContinuous-{}-{}-{}-{}-v0'.format(
                        num_peaks, good_goal, vel_penalty, initial_noise),
                    entry_point='pirl.envs.mountain_car:ContinuousMountainCarPopulationEnv',
                    max_episode_steps=999,
                    reward_threshold=90.0,
                    kwargs={
                        'num_peaks': num_peaks,
                        'goal_reward': reward,
                        'vel_penalty': vel_penalty,
                        'initial_noise': initial_noise
                    },
                )

## Reacher
for start_variance in [0.1, 0.5, 1.0]:
    for seed in range(10):
        register(
            id='pirl/ReacherGoal-seed{}-{}-v0'.format(seed, start_variance),
            entry_point='pirl.envs.reacher_goal:ReacherGoalEnv',
            max_episode_steps=50,
            kwargs={
                'seed': seed,
                'start_variance': start_variance * np.pi,
                'goal_state_pos': 'fixed',
                'goal_state_access': False,
            }
        )

        for steps in [50, 100]:
            register(
                id='pirl/ReacherWall-seed{}-{}-{}-v0'.format(seed, steps, start_variance),
                entry_point='pirl.envs.reacher_wall:ReacherWallEnv',
                max_episode_steps=steps,
                kwargs={
                    'wall_penalty': 0.4*steps,
                    'wall_seed': seed,
                    'start_variance': start_variance * np.pi,
                }
            )

    for steps in [50, 100]:
        register(
            id='pirl/ReacherWall-nowall-{}-{}-v0'.format(steps, start_variance),
            entry_point='pirl.envs.reacher_wall:ReacherWallEnv',
            max_episode_steps=steps,
            kwargs={
                'wall_seed': None,
                'start_variance': start_variance * np.pi,
            }
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
            entry_point='pirl.envs.billiards:BilliardsEnv',
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
    entry_point='pirl.envs.seaquest:SeaquestPopulationEnv',
    max_episode_steps=100000,
    kwargs={},
)