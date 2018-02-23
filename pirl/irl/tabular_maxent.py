"""
Implements a tabular version of maximum entropy inverse reinforcement learning
(Ziebart et al, 2008). There are two key differences from the version
described in the paper:
  - We do not make use of features, which are not needed in the tabular setting.
  - We use Adam rather than exponentiated gradient descent.
"""

import numpy as np
from scipy.special import logsumexp as sp_lse
import torch
from torch.autograd import Variable

from pirl.utils import getattr_unwrapped

#TODO: fully torchize?

def visitation_counts(nS, trajectories, discount):
    """Compute empirical state-action feature counts from trajectories."""
    counts = np.zeros((nS, ))
    discounted_steps = 0
    for states, actions in trajectories:
        incr = np.cumprod([1] + [discount] * (len(states) - 1))
        counts += np.bincount(states, weights=incr)
        discounted_steps += np.sum(incr)
    return counts / discounted_steps

def policy_counts(transition, initial_states, reward, horizon, discount):
    """Corresponds to Algorithm 1 of Ziebart et al (2008)."""
    nS = initial_states.shape[0]
    logsc = np.zeros(nS)  # TODO: terminal states only?
    logt = np.nan_to_num(np.log(transition))
    for i in range(horizon):
        x = logt + reward.reshape(nS, 1, 1) + logsc.reshape(1, 1, nS)
        logac = sp_lse(x, axis=2)
        logsc = sp_lse(logac, axis=1)
    action_counts = np.exp(logac - logsc.reshape(nS, 1))

    # Forward pass
    counts = np.zeros((nS, horizon + 1))
    counts[:, 0] = initial_states
    for i in range(1, horizon + 1):
        counts[:, i] = np.einsum('i,ij,ijk->k', counts[:, i-1], action_counts, transition) * discount
    if discount == 1:
        renorm = horizon + 1
    else:
        renorm = (1 - discount ** (horizon + 1)) / (1 - discount)
    return np.sum(counts, axis=1) / renorm


def maxent_irl(mdp, trajectories, discount, learning_rate=1e-2, num_iter=100):
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
        - learning_rate(float): for Adam optimizer.
        - num_iter(int): number of iterations of optimization process.

    Returns:
        list: estimated reward for each state in the MDP.
    """
    transition = getattr_unwrapped(mdp, 'transition')
    initial_states = getattr_unwrapped(mdp, 'initial_states')
    nS, _, _ = transition.shape
    horizon = max([len(states) for states, actions in trajectories])

    demo_counts = visitation_counts(nS, trajectories, discount)
    reward = Variable(torch.zeros(nS), requires_grad=True)
    optimizer = torch.optim.Adam([reward], lr=learning_rate)
    for i in range(num_iter):
        expected_counts = policy_counts(transition, initial_states,
                                        reward.data.numpy(), horizon, discount)
        optimizer.zero_grad()
        reward.grad = Variable(torch.Tensor(expected_counts - demo_counts))
        optimizer.step()

    return reward.data.numpy()