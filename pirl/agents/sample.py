import gym
import numpy as np

def value(sample, env, policy, discount, num_episodes=100, seed=0):
    '''Test policy saved in blog_dir on num_episodes in env.
        Return average reward.'''
    # TODO: does this belong in PPO or a more general class?
    trajectories = sample(env, policy, num_episodes, seed)
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