# -*- coding: utf-8 -*-
"""Population version of ContinuousMountainCar-v0 from Gym."""

import math
import gym
import gym.spaces as spaces
from gym.utils import seeding
import numpy as np

# Variables you can vary, both between and within an environment:
# * Number of peaks. Think this should always be fixed for an instance.
# * Number of flags. Also fixed for an instance (modifies the state.)
# * Value of flags. Also fixed for an instance.
# * Position of flags. Could be fixed or variable.
# * Obstacles. Could be fixed or variable.

GOAL_COLORS = [
    (.8, .1, .1), # light red
    (.2, .5, .7), # light blue
    (.3, .7, .3), # light green
    (.6, .3, .6), # purple
]

class ContinuousMountainCarPopulationEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, num_peaks, goal_reward, vel_penalty, initial_noise,
                 goal_position=None):
        # As goal may appear on either side, I have chosen symmetric bounds [-1,1];
        # this is larger width than the original Gym environment, so I have
        # scaled up other constants by a factor of 10/9 to make them comparable.
        assert num_peaks >= 2
        assert len(goal_reward) >= 1 and len(goal_reward) <= 2
        self.num_peaks = num_peaks
        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position = 0.0
        self.max_position = float(num_peaks - 1)
        self.max_speed = 0.08
        self.power = 0.0017

        self.low_state = np.array([self.min_position, -self.max_speed])
        self.high_state = np.array([self.max_position, self.max_speed])

        self.action_space = spaces.Box(low=self.min_action, high=self.max_action, shape=(1,))
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state)

        self.goal_reward = np.array(goal_reward)
        self.static_goal_position = None
        if goal_position is not None:
            assert len(goal_position) == len(goal_reward)
            self.static_goal_position = goal_position
        self.vel_penalty = vel_penalty
        self.initial_noise = initial_noise

        self.viewer = None

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        position = self.state[0]
        velocity = self.state[1]
        force = min(max(action[0], -1.0), 1.0)

        old_position = position
        # Height is (1 + cos(2 * pi * x)) / 2
        # Gradient is -pi * sin(2 * pi * x)
        # Make gravity proportional to this.
        # (This is inconsistent with Newtonian mechanics; it would imply we
        # experience infinite acceleration on a vertical cliff.
        # Maintaining this for consistency with original MountainCar.
        gravity = -math.sin(2 * math.pi * position)
        velocity += force*self.power - 0.0028 * gravity
        if (velocity > self.max_speed): velocity = self.max_speed
        if (velocity < -self.max_speed): velocity = -self.max_speed
        position += velocity
        if (position > self.max_position): position = self.max_position
        if (position < self.min_position): position = self.min_position
        if (position == self.min_position and velocity<0): velocity = 0
        if (position == self.max_position and velocity>0): velocity = 0

        left_before = (old_position < self.goal_position)
        left_after = (position < self.goal_position)
        switched_side = left_before ^ left_after
        done = np.any(switched_side)
        reward = 0
        if done:
            assert np.sum(switched_side) <= 1
            reward += np.sum(self.goal_reward[switched_side])
            vel_cost = self.vel_penalty * abs(velocity) / self.max_speed
            reward -= 100.0 * vel_cost
        reward -= math.pow(action[0], 2) * 0.1

        if self.static_goal_position is not None:
            self.state = np.array([position, velocity])
        else:
            self.state = np.concatenate(([position, velocity],
                                         self.goal_position))
        return np.array(self.state), reward, done, {}

    def reset(self):
        # Random integer from 0,...,self.num_peaks - 2
        trough = self.np_random.randint(0, self.num_peaks - 1)
        noise = self.np_random.uniform(low=-1, high=1) * self.initial_noise
        pos = trough + 0.5 + noise

        self.state = np.array([pos, 0])
        if self.static_goal_position is not None:
            self.goal_position = self.static_goal_position
        else:
            self.goal_position = np.random.choice([self.min_position,
                                                   self.max_position],
                                                  size=len(self.goal_reward),
                                                  replace=False)
            self.state = np.concatenate((self.state, self.goal_position))

        return np.array(self.state)

    def _ys(self, xs):
        return (1 + np.cos(2 * math.pi * xs)) / 2 + 0.05

    def _xs(self, xs):
        return 0.05 + (xs - self.min_position)

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering

        screen_width = 600
        screen_height = 400

        world_width = self.max_position - self.min_position
        x_scale = screen_width / (world_width + 0.1)
        y_scale = screen_height * 0.8
        carwidth=40
        carheight=20

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)
            xs = np.linspace(self.min_position - 0.05, self.max_position + 0.05, 100)
            ys = self._ys(xs)
            xys = list(zip(self._xs(xs)*x_scale, ys*y_scale))

            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            clearance = 10

            l,r,t,b = -carwidth/2, carwidth/2, carheight, 0
            car = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            frontwheel = rendering.make_circle(carheight/2.5)
            frontwheel.set_color(.5, .5, .5)
            frontwheel.add_attr(rendering.Transform(translation=(carwidth/4,clearance)))
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            backwheel = rendering.make_circle(carheight/2.5)
            backwheel.add_attr(rendering.Transform(translation=(-carwidth/4,clearance)))
            backwheel.add_attr(self.cartrans)
            backwheel.set_color(.5, .5, .5)
            self.viewer.add_geom(backwheel)

        for color, pos in zip(GOAL_COLORS, self.goal_position):
            flagx = self._xs(pos) * x_scale
            flagy1 = self._ys(pos) * y_scale
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_onetime(flagpole)
            sign = -1 if (self.max_position - pos) < 0.5 else 1
            flag = rendering.FilledPolygon([(flagx, flagy2), (flagx, flagy2-10),
                                            (flagx+sign*25, flagy2-5)])
            flag.set_color(*color)
            self.viewer.add_onetime(flag)

        pos = self.state[0]
        self.cartrans.set_translation(self._xs(pos) * x_scale,
                                      self._ys(pos) * y_scale)

        # height is (1 + cos(2 * pi * x)) / 2
        # so gradient is -pi * sin(2 * pi * x)
        # take arctan of this to get the angle of the slope
        self.cartrans.set_rotation(np.arctan(np.sin(2 * np.pi * pos) * -np.pi))

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer: self.viewer.close()
