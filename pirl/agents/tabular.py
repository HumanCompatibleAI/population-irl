import functools

import numpy as np

from pirl.utils import getattr_unwrapped

def softmax(Q):
    nS, nA = Q.shape
    Q = Q - np.max(Q)
    probs = np.exp(Q)
    return probs / np.sum(probs, axis=1).reshape(nS, 1)

def value_iteration(T, R, soft=False, policy=None,
                    discount=0.99, max_error=1e-3, max_iterations=1000):
    """Performs value iteration on a finite-state MDP.

    Args:
        - T(array): nS*nA*nS transition matrix.
        - R(array): nS reward array.
        - soft(bool): soft-max or hard-max (only used if policy unspecified).
            If hard-max, resulting policy will be deterministic.
            If soft-max, stochastic.
        - policy(optional[array]): nS*nA policy matrix.
            Optional. If specified, computes value w.r.t. the policy, where
            cell (i, j) specifies probability of action j in state i.
            Otherwise, it computes the value of the optimal policy.
        - discount(float): in [0,1].
        - terminate_at(float): return when

    Returns (pi, Q, delta) where:
        - pi is a stochastic policy matrix (nS*nA).
        - Q is the Q-value matrix (nS*nA matrix).
        - delta is a bound on the error in the Q-value matrix.
          The user should check if this is less than terminate_at to be
          sure of convergence (in case it terminates before max_iterations).
    """

    nS, nA, _ = T.shape
    R = R.reshape(nS, 1)

    Q = np.zeros((nS, nA))
    V = np.zeros((nS, 1))
    delta = float('+inf')
    terminate_at = max_error * (1 - discount) / discount
    for i in range(max_iterations):
        if policy is None:
            if soft:
                policy_V = np.sum(softmax(Q) * Q, axis=1)
            else:
                policy_V = Q.max(1)
        else:
            policy_V = np.sum(policy * Q, axis=1)
        Q = R + discount * (T * policy_V).sum(2)
        new_V = Q.sum(1)
        delta = np.linalg.norm(new_V - V, float('inf'))
        if delta < terminate_at:
            break
        V = new_V

    if soft:
        pi = softmax(Q)
    else:
        pi = (Q >= Q.max(1).reshape(nS, 1))
        # Problem: if Q has multiple values in the same row attaining maxima,
        # we'll get duplicates. Pick one.
        pi = pi * np.arange(nS * nA).reshape(nS, nA)
        pi = (pi >= pi.max(1).reshape(nS, 1))
        pi = pi * 1
    return pi, Q, V, delta

def value_iteration_env(env, reward=None, **kwargs):
    T = getattr_unwrapped(env, 'transition')
    if reward is None:
        R = getattr_unwrapped(env, 'reward')
    else:
        R = reward
    return value_iteration(T, R, **kwargs)

def value_of_policy(env, policy, *args, **kwargs):
    _, _, V, _ = value_iteration_env(env, *args, policy=policy, **kwargs)
    initial_states = getattr_unwrapped(env, 'initial_states')
    return np.sum(V * initial_states)
