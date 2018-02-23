import numpy as np

from pirl.utils import getattr_unwrapped

def value_iteration(env, policy=None,
                    discount=0.99, max_error=1e-3, max_iterations=1000):
    """Performs value iteration on a finite-state MDP.

    Args:
        - tabular_mdp(TabularMdpEnv): with nS states and nA actions.
        - policy(optional[array]): a mapping from states to actions.
            Optional. If specified, computes value w.r.t. the policy.
            Otherwise, it computes the value of the optimal policy.
        - discount(float): in [0,1].
        - terminate_at(float): return when

    Returns (pi, Q, delta) where:
        - pi is a policy (nS length vector with actions between 0,...,nA-1)
        - Q is the Q-value matrix (nS*nA matrix).
        - delta is a bound on the error in the Q-value matrix.
          The user should check if this is less than terminate_at to be
          sure of convergence (in case it terminates before max_iterations).
    """
    T = getattr_unwrapped(env, 'transition')
    nS, nA, _ = T.shape
    R = getattr_unwrapped(env, 'reward').reshape(nS, 1)

    Q = np.zeros((nS, nA))
    V = np.zeros((nS, 1))
    delta = float('+inf')
    terminate_at = max_error * (1 - discount) / discount
    for i in range(max_iterations):
        if policy is None:
            policy_V = Q.max(1)
        else:
            policy_V = Q[np.arange(nS), policy]
        Q = R + discount * (T * policy_V).sum(2)
        new_V = Q.sum(1)
        delta = np.linalg.norm(new_V - V, float('inf'))
        if delta < terminate_at:
            break
        V = new_V

    pi = Q.argmax(1)
    return pi, Q, V, delta

#SOMEDAY: policy iteration, asynchronous value iteration
#SOMEDAY: PyTorch/TensorFlow-ize?