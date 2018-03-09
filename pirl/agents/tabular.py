import functools

import numpy as np

from pirl.utils import getattr_unwrapped

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
    @functools.wraps(f)
    def helper(env, reward=None, *args, **kwargs):
        T = getattr_unwrapped(env, 'transition')
        if reward is None:
            reward = getattr_unwrapped(env, 'reward')
        H = getattr_unwrapped(env, '_max_episode_steps')
        return f(T, reward, H, *args, **kwargs)
    return helper

def value_of_policy(env, policy, discount):
    T = getattr_unwrapped(env, 'transition')
    R = getattr_unwrapped(env, 'reward')
    H = getattr_unwrapped(env, '_max_episode_steps')
    Q, info = q_iteration(T, R, H, discount, policy=policy)
    V = Q.sum(1)
    initial_states = getattr_unwrapped(env, 'initial_states')
    return np.sum(V * initial_states)
