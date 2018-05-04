from gym import Env, spaces
from gym.envs import atari
import pandas as pd
import numpy as np
import time

OXYGEN_LOCATION = 102
DIVERS_LOCATION = 62

def identity_reward(ram, reward, image, action, state):
    return reward

def submarine_killed(prev_image, image):
    # We determine whether a submarine has been killed, based on whether 45 pixels of the submarine color disappeared in the last two frames
    SUB_COLOR = (170, 170, 170)

    prev_values, prev_counts = np.unique(prev_image.reshape([-1, 3]), return_counts=True, axis=0)
    previous_submarine_count = prev_counts[np.where(np.all(prev_values == SUB_COLOR, axis=1))[0]] // 45

    values, counts = np.unique(image.reshape([-1, 3]), return_counts=True, axis=0)
    current_submarine_count = counts[np.where(np.all(values == SUB_COLOR, axis=1))[0]] // 45

    # Method above returns an empty array if we found no submarines, so make sure to only unpack array if it has content
    previous_submarine_count = 0 if len(previous_submarine_count) < 1 else previous_submarine_count[0]
    current_submarine_count = 0 if len(current_submarine_count) < 1 else current_submarine_count[0]

    return (previous_submarine_count == (current_submarine_count + 1))

def fancy_reward_HoF(oxygen_fn=None, diver_reward=100, shark_reward=5, submarine_reward=10, action_fn=None, death_reward=(-20)):
    '''
    Higher-order-function that generates a reward function for the Seaquest population variant
    '''
    if (oxygen_fn == None):
        # oxygen_fn = lambda ox: 0
        oxygen_fn = lambda ox: -np.exp(5/ox)
    if (action_fn == None):
        # action_fn = lambda ac: 0
        action_fn = lambda ac: -1 if ac == 1 else 0
    def reward_function(ram, reward, image, action, lives, state):
        prev_oxygen = state['prev_ram'][OXYGEN_LOCATION]
        oxygen = ram[OXYGEN_LOCATION]

        prev_lives = state['prev_lives']

        prev_divers = state['prev_ram'][DIVERS_LOCATION]
        divers = ram[DIVERS_LOCATION]
        rescued_diver = 1 if (divers == prev_divers + 1) else 0

        new_reward = 0
        new_reward += oxygen_fn(oxygen)
        new_reward += action_fn(action)
        if (rescued_diver):
            new_reward += diver_reward

        if (reward > 0 and submarine_killed(state['prev_image'], image)):
            new_reward += submarine_reward

        surfaced_with_divers = oxygen > prev_oxygen and divers < prev_divers and lives == prev_lives
        if (reward > 0 and not submarine_killed(state['prev_image'], image) and not surfaced_with_divers):
            new_reward += shark_reward

        if (lives < prev_lives):
            new_reward += death_reward

        return new_reward
    return reward_function

AtariEnv = atari.AtariEnv

class SeaquestPopulationEnv(AtariEnv):
    def __init__(self, reward_fn=fancy_reward_HoF(), obs_type='ram'):
        """Creates a sequest environment that supports varying rewards. Wraps the
        standard gym Atari environment and adds support for custom reward functions

        Args:
            reward_fn (function (ram, reward, image, action, lives, (prev_ram, prev_image, prev_lives)) -> reward): Function that takes in a complete ram state and the original reward and maps it to a new reward
            obs_type: Whether to render pixels ('image') or return the state of the ram ('ram')
        """
        super().__init__(game='seaquest', obs_type=obs_type, frameskip=2)
        self._reward_fn = reward_fn
        self._image = super().render(mode='rgb_array')
        self._lives = 0
        self._ram = super()._get_ram()
        self._odd_frame = False

    def step(self, action):
        ob, reward, game_over, info = super().step(action)
        image = super().render(mode='rgb_array')
        ram = super()._get_ram()
        lives = info['ale.lives']
        reward = self._reward_fn(ram, reward, image, action, lives, {'prev_ram': self._ram, 'prev_image': self._image, 'prev_lives': self._lives})
        # We need to make sure we only record every second frame, so we capture dying submarines and sharks
        if self._odd_frame:
            self._odd_frame = False
        else:
            self._odd_frame = True
            self._ram = ram
            self._image = image
            self._lives = lives
        return ob, reward, lives==0, info
