"""
Defines a set of pre-specified environments. Some additional customization
may be possible,
"""

from pirl.envs.gridworld import GridWorldMdp

def simple_gridworld():
    """Hardcoded 4x4 gridworld. Only intended for code testing."""
    grid = ['A X1',
            '    ',
            ' 1X ',
            'A X9']
    return GridWorldMdp.from_string(grid, default_reward=-0.1, noise=0)


ENVIRONMENTS = {
    'GridWorld-Simple': simple_gridworld,
}


def make(name, kwargs):
    return ENVIRONMENTS[name](**kwargs)