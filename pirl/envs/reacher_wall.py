import tempfile
import os.path

import numpy as np
from gym import utils
from gym.utils import seeding
from gym.envs.mujoco import mujoco_env

class ReacherWallEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, seed=0, start_variance=0.1,
                 wall_penalty=5, wall_state_access=False):
        self._start_variance = start_variance
        self._wall_state_access = wall_state_access
        self._wall_penalty = wall_penalty

        wall_rng = np.random.RandomState(seeding.create_seed(seed))
        self._wall_angle = np.pi * (2 * wall_rng.rand() - 1)
        x = np.cos(self._wall_angle)
        y = np.sin(self._wall_angle)

        model_path = os.path.join(os.path.dirname(__file__), 'reacher_wall.xml')
        with open(model_path, 'r') as model:
            model_xml = model.read()
            params = {
                'XS': x * 0.08,
                'YS': y * 0.08,
                'XE': x * 0.21,
                'YE': y * 0.21,
            }
            for k, v in params.items():
                model_xml = model_xml.replace(k, str(v))

        utils.EzPickle.__init__(self)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml') as f:
            f.write(model_xml)
            f.flush()
            mujoco_env.MujocoEnv.__init__(self, f.name, 2)

    def step(self, a):
        self.do_simulation(a, self.frame_skip)

        vec = self.get_body_com('fingertip') - self.get_body_com('target')
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + reward_ctrl

        if self.sim.data.ncon > 0:
            reward -= self._wall_penalty

        ob = self._get_obs()
        done = False
        info = dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)
        return ob, reward, done, info

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        # Randomly choose goal in circle radius 0.2, excluding the sector
        # 0.25 radians away from the wall.
        while True:
            self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
            within_armspan = np.linalg.norm(self.goal) < .2
            goal_angle = np.arctan2(self.goal[1], self.goal[0])
            outside_wall = np.abs(goal_angle - self._wall_angle) > 0.25
            if within_armspan and outside_wall:
                break

        # Randomly choose arm position, excluding sector 0.05 radians
        # around the wall.
        while True:
            sv = self._start_variance
            arm_pos_rnd = self.np_random.uniform(low=-sv, high=sv, size=2)
            arm_pos = self.init_qpos[:-2] + arm_pos_rnd
            arm_delta = arm_pos[0] - self._wall_angle

            arm_theta = np.cumsum(arm_pos)
            finger_xpos = np.sum([np.cos(arm_theta), np.sin(arm_theta)], axis=1)
            finger_angle = np.arctan2(finger_xpos[1], finger_xpos[0])
            finger_delta = finger_angle - self._wall_angle

            arm_outside = np.abs(arm_delta) > 0.05
            finger_outside = np.abs(finger_delta) > 0.05
            intersects = np.sign([arm_delta, finger_delta]).prod() == -1

            if arm_outside and finger_outside and not intersects:
                break
        arm_vel_rnd = self.np_random.uniform(low=-.005, high=.005, size=2)
        arm_vel = self.init_qvel[:-2] + arm_vel_rnd

        qpos = np.concatenate([arm_pos, self.goal])
        qvel = np.concatenate([arm_vel, np.zeros(2)])
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]
        if self._wall_state_access:
            theta = np.concatenate([theta, self._wall_angle])
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            theta,
            self.sim.data.qvel.flat[:2],
            self.get_body_com('fingertip') - self.get_body_com('target')
        ])
