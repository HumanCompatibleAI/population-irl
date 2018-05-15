import functools

import gym
from gym.utils import seeding
import numpy as np

from pirl.utils import discrete_sample, getattr_unwrapped, vectorized

def q_iteration(transition, reward, horizon, discount,
                policy=None, max_error=1e-3):
    """Performs value iteration on a finite-state MDP.

    Args:
        - T(array): nS*nA*nS transition matrix.
        - R(array): nS reward array.
        - H(int): maximum number of iterations.
        - policy(optional[array]): nS*nA policy matrix.
            Optional. If specified, computes value w.r.t. the policy, where
            cell (i, j) specifies probability of action j in state i.
            Otherwise, it computes the value of the optimal policy.
        - discount(float): in [0,1].
        - terminate_at(float): return when

    Returns (Q, info) where:
        - Q is the Q-value matrix (nS*nA matrix).
        - info is a dict providing information about convergence, in particular:
            - error, a bound on the error in the Q-value matrix.
            - num_iter, the number of iterations (at most max_iterations).
    """

    nS, nA, _ = transition.shape
    reward = reward.reshape(nS, 1)

    Q = np.zeros((nS, nA))
    V = np.zeros((nS, 1))
    delta = float('+inf')
    terminate_at = max_error * (1 - discount) / discount
    for i in range(horizon):
        if policy is None:
            policy_V = Q.max(1)
        else:
            policy_V = np.sum(policy * Q, axis=1)
        Q = reward + discount * (transition * policy_V).sum(2)
        new_V = Q.sum(1)
        delta = np.linalg.norm(new_V - V, float('inf'))
        if delta < terminate_at:
            break
        V = new_V

    info = {
        'error': delta * discount / (1 - discount),
        'num_iter': i,
    }
    return Q, info


def get_policy(Q):
    """
    Computes an optimal policy from a Q-matrix.
    Args:
        - Q(array): nS*nA Q-matrix.
    Returns:
        a policy matrix (nS*nA), assigning uniform probability across all
        optimal actions in a given state.
    """
    nS, nA = Q.shape
    pi = (Q >= Q.max(1).reshape(nS, 1))
    # If more than one optimal action, split probability between them
    pi = pi / pi.sum(1).reshape(nS, 1)
    return pi


def q_iteration_policy(T, R, H, discount):
    Q, info = q_iteration(T, R, H, discount)
    return get_policy(Q)


def env_wrapper(f):
    @vectorized(False)
    @functools.wraps(f)
    def helper(mdp, log_dir=None, reward=None, *args, **kwargs):
        # log_dir is not used but is needed to match function signature.
        T = getattr_unwrapped(mdp, 'transition')
        if reward is None:
            reward = getattr_unwrapped(mdp, 'reward')
        H = getattr_unwrapped(mdp, '_max_episode_steps')
        return f(T, reward, H, *args, **kwargs)
    return helper


@vectorized(False)
def value_in_mdp(mdp, policy, discount, seed):
    '''Exact value of a tabular policy in environment mdp with given discount.
       Returns (value, 0), where 0 represents the standard error.'''
    T = getattr_unwrapped(mdp, 'transition')
    R = getattr_unwrapped(mdp, 'reward')
    H = getattr_unwrapped(mdp, '_max_episode_steps')
    Q, info = q_iteration(T, R, H, discount, policy=policy)
    V = Q.sum(1)
    initial_states = getattr_unwrapped(mdp, 'initial_states')
    value = np.sum(V * initial_states)
    return value, 0


@vectorized(False)  # step is cheap so no point parallelizing
def sample(env, policy, num_episodes, seed):
    # seed to make results reproducible
    rng, _ = seeding.np_random(seed)

    def helper():
        '''Samples for one episode.'''
        states = []
        actions = []
        rewards = []

        state = env.reset()
        done = False
        while not done:
            states.append(state)
            action_dist = policy[state]
            action = discrete_sample(action_dist, rng)
            actions.append(action)
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
        return np.array(states), np.array(actions), np.array(rewards)

    return [helper() for i in range(num_episodes)]


class TabularRewardWrapper(gym.Wrapper):
    """Wrapper for a gym.Env replacing with a new reward matrix."""
    def __init__(self, env, new_reward):
        # Note: will fail on vectorized environments, but value_iteration
        # doesn't support these anyway.
        self.new_reward = new_reward
        super().__init__(env)

    def step(self, action):
        observation, old_reward, done, info = self.env.step(action)
        new_reward = self.new_reward[observation, action]
        return observation, new_reward, done, info

    @property
    def reward(self):
        return self.new_reward

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)