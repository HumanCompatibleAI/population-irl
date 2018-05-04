import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.utils import seeding

class ReacherPopulationEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, seed=0, start_variance=0.1,
                 goal_state_pos='fixed', goal_state_access=True):
        '''
        Multi-task (population) version of Gym Reacher environment.

        Args:
            seed: for RNG determining goal position.
            start_variance: variance of the starting position for the arm.
            goal_state_pos: if 'fixed', goal position is static across reset().
            goal_state_access: is goal position included in the state?
        '''
        self._start_variance = start_variance
        self._goal_state_pos = goal_state_pos
        self._goal_state_access = goal_state_access
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'reacher.xml', 2)
        self._goal_rng = np.random.RandomState(seeding.create_seed(seed))
        if self._goal_state_pos == 'fixed':
            self._reset_goal()

    def _reset_goal(self):
        while True:
            self.goal = self._goal_rng.uniform(low=-.2, high=.2, size=2)
            if np.linalg.norm(self.goal) < 2:
                break

    def step(self, a):
        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        sv = self._start_variance
        qpos = self.np_random.uniform(low=-sv, high=sv, size=self.model.nq) + self.init_qpos

        if self._goal_state_pos != 'fixed':
            self._reset_goal()

        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.sim.data.qpos.flat[2:] if self._goal_state_access else [],
            self.sim.data.qvel.flat[:2],
            self.get_body_com("fingertip") - self.get_body_com("target")
        ])
