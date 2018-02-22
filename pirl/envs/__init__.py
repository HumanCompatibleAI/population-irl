from pirl.envs.tabular_mdp_env import TabularMdpEnv
from pirl.envs.gridworld import GridWorldMdp

from gym.envs.registration import register

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
        'noise': 0,
    },
)