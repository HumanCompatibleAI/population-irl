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
