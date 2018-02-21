from io import StringIO

import numpy as np

from pirl.tabular_mdp_env import TabularMdpEnv

def _create_transition(walls, noise):
    width, height = walls.shape
    walls = walls.flatten()

    nS = walls.shape[0]
    nA = len(Direction.ALL_DIRECTIONS)
    transition = np.zeros(nS, nA, nS)

    def move(start, dir):
        oldx, oldy = start % width, start // width
        newx, newy = Direction.move_in_direction((oldx, oldy), dir)
        newx = max(0, min(newx, width - 1))
        newy = max(0, min(newy, height - 1))
        return start if walls[newy][newx] else (newx * width + newy)

    for idx, cfg in enumerate(walls):
        for a, dir in enumerate(Direction.ALL_DIRECTIONS):
            if cfg == 'X':  # wall
                # Can never get into a wall, but TabularMdpEnv
                # insists transition be a probability distribution,
                # so make it an absorbing state.
                transition[idx, a, idx] = 1
            else:  # unobstructed space
                if dir == Direction.STAY:
                    transition[idx, a, idx] = 1
                else:
                    transition[idx, a, move(idx, dir)] = 1 - 2 * noise
                    for noise_dir in Direction.get_adjacent_directions(dir):
                        transition[idx, a, move(idx, noise_dir)] += noise

    return transition

def _create_reward(grid, default_reward):
    def convert(cfg):
        if cfg in ['X', ' ', 'A']:
            return -default_reward
        else:
            return float(cfg)
    rewards = [[convert(cfg) for cfg in row] for row in grid]
    return np.array(rewards).flatten()

def _create_initial_state(grid):
    cfg = np.array(grid, dtype='object').flatten()
    initial_state = cfg == 'A'
    return initial_state / initial_state.sum()


class GridworldMdp(TabularMdpEnv):
    """A grid world where the objective is to navigate to one of many rewards.

    Specifies all of the static information that an agent has access to when
    playing in the given grid world, including the state space, action space,
    transition probabilities, rewards, start state, etc.

    The agent can take any of the four cardinal directions as an action, getting
    a living reward (typically negative in order to incentivize shorter
    paths). It can also take the STAY action, in which case it does not receive
    the living reward.
    """
    def __init__(self, walls, reward, initial_state, terminal, noise):
        # Check dimensions
        assert walls.shape == reward.shape
        assert walls.shape == initial_state.shape

        # Setup state
        self.walls = walls  # used only for rendering
        transition = _create_transition(walls, noise)
        reward = reward.flatten()
        initial_state = initial_state.flatten()
        terminal = terminal.flatten()
        super().__init__(self, transition, reward, initial_state, terminal)

    def get_reward(self, state, action):
        """Get reward for state, action transition."""
        result = 0
        if state in self.rewards:
            result += self.rewards[state]
        if action != Direction.STAY:
            result += self.living_reward
        return result

    def render(self, mode='human'):
        """Returns a string representation of this grid world.

        The returned string has a line for every row, and each space is exactly
        one character. These are encoded in the same way as the grid input to
        the constructor -- walls are 'X', empty spaces are ' ', the start state
        is 'A', and rewards are their own values. However, rewards like 3.5 or
        -9 cannot be represented with a single character. Such rewards are
        encoded as 'R' (if positive) or 'N' (if negative).
        """
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        def get_char(x, y):
            if self.walls[y][x]:
                return 'X'
            elif (x, y) in self.rewards:
                reward = self.rewards[(x, y)]
                # Convert to an int if it would not lose information
                reward = int(reward) if int(reward) == reward else reward
                posneg_char = 'R' if reward >= 0 else 'N'
                reward_str = str(reward)
                return reward_str if len(reward_str) == 1 else posneg_char
            elif (x, y) == self.get_start_state():
                return 'A'
            else:
                return ' '

        def get_row_str(y):
            return ''.join([get_char(x, y) for x in range(self.width)])

        outfile.write('\n'.join([get_row_str(y) for y in range(self.height)]))

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