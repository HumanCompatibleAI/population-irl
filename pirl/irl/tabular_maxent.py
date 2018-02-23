"""
Implements a tabular version of maximum entropy inverse reinforcement learning
(Ziebart et al, 2008). There are two key differences from the version
described in the paper:
  - We do not make use of features, which are not needed in the tabular setting.
  - We use Adam rather than exponentiated gradient descent.
"""

import numpy as np

from pirl.utils import getattr_unwrapped

def visitation_counts(nS, trajectories, discount):
    """Compute empirical state-action feature counts from trajectories."""
    counts = np.zeros((nS, ))
    num_steps = 0
    for states, actions in trajectories:
        states = states[1:]
        length = len(states)
        # TODO: should this be discounted?
        #incr = np.cumprod([discount] * length)
        incr = np.array([1] * length)
        counts += np.bincount(states, weights=incr)
        num_steps += length
    return counts / num_steps


#TODO: horizon
def policy_counts(horizon, transition, initial_states, reward):
    """Corresponds to Algorithm 1 of Ziebart et al (2008)."""
    # Backwards pass
    nS = initial_states.shape[0]
    state_counts = np.ones(nS)  # TODO: terminal states only?
    for i in range(horizon):
        action_counts = np.einsum('ijk,i,k->ij', transition, np.exp(reward), state_counts)
        state_counts = np.sum(action_counts, axis=1)
    action_probs = action_counts / state_counts.reshape((nS, 1))

    # Forward pass
    counts = np.zeros((nS, horizon + 1))
    counts[:, 0] = initial_states
    for i in range(1, horizon + 1):
        x = np.einsum('ijk,k->ij', transition, counts[:, i-1])
        counts[:, i] = np.sum(x * action_probs, axis=1)
    return np.sum(counts, axis=1) / (horizon + 1)


def maxent_irl(mdp, trajectories, discount):
    """
    Args:
        - mdp(TabularMdpEnv): MDP trajectories were drawn from.
        - trajectories(list): observed trajectories.
            List containing one (states, actions) pair for each trajectory,
            where states and actions are lists containing all visited
            states/actions in that trajectory. The length of states should be
            one greater than that of actions (since we include the start and
            final state).
        - discount(float): between 0 and 1.
            Should match that of the agent generating the trajectories.

    Returns:
        list: estimated reward for each state in the MDP.
    """
    transition = getattr_unwrapped(mdp, 'transition')
    initial_states = getattr_unwrapped(mdp, 'initial_states')
    nS, _, _ = transition.shape
    reward = np.zeros(nS)
    horizon = max([len(states) for states, actions in trajectories])
    learning_rate = 1e-3

    demo_counts = visitation_counts(nS, trajectories, discount)
    # TODO: use actual optimization framework
    for i in range(10):
        expected_counts = policy_counts(horizon, transition,
                                        initial_states, reward)
        grad = expected_counts - demo_counts
        reward = reward - learning_rate * grad
    print(reward.reshape(4,4))
    print(grad.reshape(4,4))

    return reward