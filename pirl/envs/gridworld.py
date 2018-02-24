from io import StringIO
import sys

import numpy as np
from gym import utils

from pirl.envs.tabular_mdp_env import TabularMdpEnv

def _create_transition(walls, noise):
    width, height = walls.shape
    walls = walls.flatten()

    nS = walls.shape[0]
    nA = len(Direction.ALL_DIRECTIONS)
    transition = np.zeros((nS, nA, nS))

    def move(start, dir):
        oldx, oldy = start % width, start // width
        newx, newy = Direction.move_in_direction((oldx, oldy), dir)
        newx = max(0, min(newx, width - 1))
        newy = max(0, min(newy, height - 1))
        idx = newy * width + newx
        return start if walls[idx] else idx

    for idx, wall in enumerate(walls):
        for a, dir in enumerate(Direction.ALL_DIRECTIONS):
            if wall:  # wall
                # Can never get into a wall, but TabularMdpEnv
                # insists transition be a probability distribution,
                # so make it an absorbing state.
                transition[idx, a, idx] = 1
            else:  # unobstructed space
                if dir == Direction.STAY:
                    transition[idx, a, idx] = 1
                else:
                    transition[idx, a, move(idx, dir)] = 1 - noise
                    for noise_dir in Direction.get_adjacent_directions(dir):
                        transition[idx, a, move(idx, noise_dir)] += noise / 2

    return transition

def _create_reward(grid, default_reward):
    def convert(cfg):
        if cfg in ['X', ' ', 'A']:
            return -default_reward
        else:
            return float(cfg)
    rewards = [[convert(cfg) for cfg in row] for row in grid]
    return np.array(rewards)

def create_initial_state(grid):
    cfg = np.array(grid, dtype='object')
    initial_state = cfg == 'A'
    return initial_state / initial_state.sum()


class GridWorldMdp(TabularMdpEnv):
    """A grid world where the objective is to navigate to one of many rewards.

    Specifies all of the static information that an agent has access to when
    playing in the given grid world, including the state space, action space,
    transition probabilities, rewards, start state, etc.

    The agent can take any of the four cardinal directions as an action, getting
    a living reward (typically negative in order to incentivize shorter
    paths). It can also take the STAY action, in which case it does not receive
    the living reward.
    """
    def __init__(self, walls, reward, initial_state, terminal, noise=0.2):
        """Create an N*M grid world of the specified structure.

        Args:
            - walls(N*M bool matrix): specifies cells with walls
            - reward(N*M float matrix): reward obtained when entering cell.
            - initial_state(N*M float matrix): probability distribution.
            - terminal(N*M bool matrix): does entering cell end episode?
            - noise(float): probability intended action does not take place.
        """
        # Check dimensions
        assert walls.shape == reward.shape
        assert walls.shape == initial_state.shape

        # Setup state
        self.walls = walls  # used only for rendering
        transition = _create_transition(walls, noise)
        reward = reward.flatten()
        initial_state = initial_state.flatten()
        terminal = terminal.flatten()
        super().__init__(transition, reward, initial_state, terminal)

    @staticmethod
    def from_string(grid, noise=0.2, default_reward=0.0):
        """Create an N*M grid world from an N-length array of M-length arrays
           of characters or floats (M-length string also permissible).

        Each cell should be one of:
            - 'X': a wall.
            - 'A': an initial state, with reward default_reward.
            - ' ': a cell, with reward default_reward.
            - a numeric character ('1', '4', etc) or something castable to a
              a float (5, 4.2, etc), specifying the given reward.
        """
        grid = [list(x) for x in grid]
        walls = np.array(grid, dtype='U1') == 'X'
        reward = _create_reward(grid, default_reward)
        initial_state = create_initial_state(grid)
        terminal = np.zeros_like(walls, dtype=bool)
        return GridWorldMdp(walls, reward, initial_state, terminal, noise)

    def render(self, mode='human'):
        #TODO: PNG/X11 rendering?
        """Returns a string representation of this grid world.

        The returned string has a line for every row, and each space is exactly
        one character. These are encoded in the same way as the grid input to
        the constructor -- walls are 'X', empty spaces are ' ', the start state
        is 'A', and rewards are their own values. However, rewards like 3.5 or
        -9 cannot be represented with a single character. Such rewards are
        encoded as 'R' (if positive) or 'N' (if negative).
        """
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        walls = self.walls
        initial_state = self.initial_states.reshape(walls.shape)
        reward = self.reward.reshape(walls.shape)
        width, height = walls.shape

        current_x, current_y = self.state % width, self.state // width
        def get_char(x, y):
            if walls[y, x]:
                res = 'X'
            elif initial_state[y, x] > 0:
                #TODO: show where we actually started?
                res = 'A'
            else:
                # TODO: handle default reward more elegantly
                r = reward[y, x]
                # Convert to an int if it would not lose information
                r = int(r) if int(r) == r else r
                posneg_char = 'R' if r >= 0 else 'N'
                reward_str = str(r)
                res = reward_str if len(reward_str) == 1 else posneg_char
            if (x, y) == (current_x, current_y):
                res = utils.colorize(res, 'red', highlight=True)
            return res

        def get_row_str(y):
            return ''.join([get_char(x, y) for x in range(width)])

        outfile.write('\n'.join([get_row_str(y) for y in range(height)]))

        if mode == 'ansi':
            return outfile


class Direction(object):
    """A class that contains the five actions available in Gridworlds.

    Includes definitions of the actions as well as utility functions for
    manipulating them or applying them.
    """
    NORTH = (0, -1)
    SOUTH = (0, 1)
    EAST  = (1, 0)
    WEST  = (-1, 0)
    STAY = (0, 0)
    INDEX_TO_DIRECTION = [NORTH, SOUTH, EAST, WEST, STAY]
    DIRECTION_TO_INDEX = { a:i for i, a in enumerate(INDEX_TO_DIRECTION) }
    ALL_DIRECTIONS = INDEX_TO_DIRECTION

    @staticmethod
    def move_in_direction(point, direction):
        """Takes a step in the given direction and returns the new point.

        point: Tuple (x, y) representing a point in the x-y plane.
        direction: One of the Directions, except not Direction.STAY or
                   Direction.SELF_LOOP.
        """
        x, y = point
        dx, dy = direction
        return (x + dx, y + dy)

    @staticmethod
    def get_adjacent_directions(direction):
        """Returns the directions within 90 degrees of the given direction.

        direction: One of the Directions, except not Direction.STAY.
        """
        if direction in [Direction.NORTH, Direction.SOUTH]:
            return [Direction.EAST, Direction.WEST]
        elif direction in [Direction.EAST, Direction.WEST]:
            return [Direction.NORTH, Direction.SOUTH]
        raise ValueError('Invalid direction: %s' % direction)

    @staticmethod
    def get_number_from_direction(direction):
        return Direction.DIRECTION_TO_INDEX[direction]

    @staticmethod
    def get_direction_from_number(number):
        return Direction.INDEX_TO_DIRECTION[number]