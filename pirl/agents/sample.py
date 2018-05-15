import gym
import numpy as np

from baselines.common.vec_env import VecEnvWrapper

from pirl.utils import vectorized


@vectorized(True)
def value(sample, envs, policy, discount, num_episodes=100, seed=0):
    '''Test policy saved in blog_dir on num_episodes in env.
        Return average reward.'''
    # TODO: does this belong in PPO or a more general class?
    trajectories = sample(envs, policy, num_episodes, seed)
    rewards = [r for (s, a, r) in trajectories]
    horizon = max([len(s) for (s, a, r) in trajectories])
    weights = np.cumprod([1] + [discount] * (horizon - 1))
    total_reward = [np.dot(r, weights[:len(r)]) for r in rewards]

    mean = np.mean(total_reward)
    se = np.std(total_reward, ddof=1) / np.sqrt(num_episodes)
    return mean, se


class SampleMonitor(gym.Wrapper):
    def __init__(self, env):
        self._trajectories = []
        self.observations = None
        self.actions = None
        self.rewards = None
        super(SampleMonitor, self).__init__(env)

    def step(self, action):
        self.actions.append(action)
        ob, r, done, info = self.env.step(action)
        self.observations.append(ob)
        self.rewards.append(r)
        return ob, r, done, info

    def reset(self, **kwargs):
        if self.observations is not None:
            traj = (self.observations[:-1], self.actions, self.rewards)
            traj = tuple(np.array(x) for x in traj)
            self._trajectories.append(traj)
        self.observations = []
        self.actions = []
        self.rewards = []
        ob = self.env.reset(**kwargs)
        self.observations.append(ob)
        return ob

    @property
    def trajectories(self):
        return self._trajectories

class SampleVecMonitor(VecEnvWrapper):
    def __init__(self, venv, trajectories=None):
        '''Takes a vector environment venv and an empty collection trajectories;
           trajectories are then stored in the collection. For most use cases,
           the default of trajectories as an empty list is sufficient; in some
           cases, other data structures e.g. a ring-buffer may be useful.'''
        if trajectories is None:
            trajectories = []
        assert len(trajectories) == 0
        self._trajectories = trajectories
        self.observations = None
        self.actions = None
        self.rewards = None
        super(SampleVecMonitor, self).__init__(venv)

    def step_async(self, actions):
        for i, a in enumerate(actions):
            self.actions[i].append(a)
        return self.venv.step_async(actions)

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        for i, (o, r, d) in enumerate(zip(obs, rews, dones)):
            self.rewards[i].append(r)
            if d:
                traj = (self.observations[i], self.actions[i], self.rewards[i])
                traj = tuple(np.array(x) for x in traj)
                self._trajectories.append(traj)
                self.observations[i] = [o]
                self.actions[i] = []
                self.rewards[i] = []
            else:
                self.observations[i].append(o)
        return obs, rews, dones, infos

    def reset(self):
        obs = self.venv.reset()
        num_envs = len(obs)
        self.observations = [[] for _i in range(num_envs)]
        self.actions = [[] for _i in range(num_envs)]
        self.rewards = [[] for _i in range(num_envs)]
        for i, o in enumerate(obs):
            self.observations[i].append(o)
        return obs

    @property
    def trajectories(self):
        return self._trajectories