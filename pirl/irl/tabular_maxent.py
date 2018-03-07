"""
Implements a tabular version of maximum entropy inverse reinforcement learning
(Ziebart et al, 2008). There are two key differences from the version
described in the paper:
  - We do not make use of features, which are not needed in the tabular setting.
  - We use Adam rather than exponentiated gradient descent.
"""

import functools

import numpy as np
from scipy.special import logsumexp as sp_lse
import torch
from torch.autograd import Variable

from pirl.agents import tabular
from pirl.utils import getattr_unwrapped

#TODO: fully torchize?

def visitation_counts(nS, trajectories, discount):
    """Compute empirical state-action feature counts from trajectories."""
    counts = np.zeros((nS, ))
    discounted_steps = 0
    for states, actions in trajectories:
        incr = np.cumprod([1] + [discount] * (len(states) - 1))
        counts += np.bincount(states, weights=incr, minlength=nS)
        discounted_steps += np.sum(incr)
    return counts / discounted_steps

def policy_counts(transition, initial_states, reward, horizon, discount):
    """First-half corresponds to algorithm 9.1 of Ziebart (2010);
       second-half to algorithm 9.3."""
    nS = initial_states.shape[0]
    res = tabular.value_iteration(transition, reward,
                                  soft=True, max_iterations=horizon)
    action_counts, _, _, _ = res

    # Forward pass
    counts = np.zeros((nS, horizon + 1))
    counts[:, 0] = initial_states
    for i in range(1, horizon + 1):
        counts[:, i] = np.einsum('i,ij,ijk->k', counts[:, i-1],
                                 action_counts, transition) * discount
    if discount == 1:
        renorm = horizon + 1
    else:
        renorm = (1 - discount ** (horizon + 1)) / (1 - discount)
    return np.sum(counts, axis=1) / renorm

default_optimizer = functools.partial(torch.optim.SGD, lr=1e-1)
default_scheduler = functools.partial(torch.optim.lr_scheduler.ExponentialLR,
                                      gamma=0.995)


def maxent_irl(mdp, trajectories, discount,
               optimizer=None, scheduler=None, num_iter=500):
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
        - optimizer(callable): a callable returning a torch.optim object.
            The callable is called with an iterable of parameters to optimize.
        - scheduler(callable): a callable returning a torch.optim.lr_scheduler.
            The callable is called with a torch.optim optimizer object.
        - learning_rate(float): for Adam optimizer.
        - num_iter(int): number of iterations of optimization process.

    Returns (reward, info) where:
        reward(list): estimated reward for each state in the MDP.
        info(dict): log of extra info.
    """
    transition = getattr_unwrapped(mdp, 'transition')
    initial_states = getattr_unwrapped(mdp, 'initial_states')
    nS, _, _ = transition.shape
    horizon = max([len(states) for states, actions in trajectories])

    demo_counts = visitation_counts(nS, trajectories, discount)
    reward = Variable(torch.zeros(nS), requires_grad=True)
    if optimizer is None:
        optimizer = default_optimizer
    if scheduler is None:
        scheduler = default_scheduler
    optimizer = optimizer([reward])
    scheduler = scheduler(optimizer)
    ec_history = []
    grad_history = []
    for i in range(num_iter):
        expected_counts = policy_counts(transition, initial_states,
                                        reward.data.numpy(), horizon, discount)
        optimizer.zero_grad()
        reward.grad = Variable(torch.Tensor(expected_counts - demo_counts))
        optimizer.step()
        scheduler.step()

        ec_history.append(expected_counts)
        grad_history.append(reward.grad.data.numpy())

    info = {
        'demo_counts': demo_counts,
        'expected_counts': ec_history,
        'grads': grad_history
    }
    return reward.data.numpy(), info


def maxent_population_irl(mdps, trajectories, discount,
                          individual_reg=1e-2, common_scale=1, demean=True,
                          optimizer=None, scheduler=None, num_iter=500):
    """
    Args:
        - mdp(dict<TabularMdpEnv>): MDPs trajectories were drawn from.
            Dictionary containing MDPs trajectories, of the same name, were
            drawn from. MDPs must have the same state/action spaces, but may
            have different dynamics and reward functions.
        - trajectories(dict<list>): observed trajectories.
            Dictionary of lists containing one (states, actions) pair for each
            trajectory, where states and actions are lists containing all
            visited states/actions in that trajectory. The length of states
            should be one greater than that of actions (since we include the
            start and final state).
        - individual_reg(float): regularization factor for per-agent reward.
            Penalty factor applied to the l_2 norm of per-agent reward matrices.
        - common_scale(float): scaling factor for common gradient update.
        - demean(bool): demean the gradient.
        - discount(float): between 0 and 1.
            Should match that of the agent generating the trajectories.
        - optimizer(callable): a callable returning a torch.optim object.
            The callable is called with an iterable of parameters to optimize.
        - scheduler(callable): a callable returning a torch.optim.lr_scheduler.
            The callable is called with a torch.optim optimizer object.
        - num_iter(int): number of iterations of optimization process.

    Returns (reward, info) where:
        reward(dict<list>): estimated reward for each state in the MDP.
        info(dict): log of extra info.    """
    assert mdps.keys() == trajectories.keys()

    transitions = {}
    initial_states = {}
    rewards = {}
    transition_shape = None
    initial_shape = None
    for name, mdp in mdps.items():
        trans = getattr_unwrapped(mdp, 'transition')
        assert transition_shape is None or transition_shape == trans.shape
        transition_shape = trans.shape
        transitions[name] = trans

        initial = getattr_unwrapped(mdp, 'initial_states')
        assert initial_shape is None or initial_shape == initial.shape
        initial_shape = initial.shape
        initial_states[name] = initial

        nS, _, _ = trans.shape
        rewards[name] = Variable(torch.zeros(nS), requires_grad=True)
    rewards['common'] = Variable(torch.zeros(nS), requires_grad=True)

    horizons = {}
    demo_counts = {}
    for name, trajectory in trajectories.items():
        horizons[name] = max([len(states) for states, actions in trajectory])
        demo_counts[name] = visitation_counts(nS, trajectory, discount)

    if optimizer is None:
        optimizer = default_optimizer
    if scheduler is None:
        scheduler = default_scheduler
    optimizer = optimizer(rewards.values())
    scheduler = scheduler(optimizer)
    ec_history = {}
    grad_history = []
    reward_history = []
    for i in range(num_iter):
        optimizer.zero_grad()
        grads = {}
        for name in mdps.keys():
            effective_reward = rewards[name] + rewards['common']
            expected_counts = policy_counts(transitions[name],
                                            initial_states[name],
                                            effective_reward.data.numpy(),
                                            horizons[name],
                                            discount)
            grads[name] = expected_counts - demo_counts[name]
            ec_history.setdefault(name, []).append(expected_counts)
        common_grad = np.mean(list(grads.values()), axis=0)
        rewards['common'].grad = Variable(torch.Tensor(common_grad)) * common_scale

        if demean:
            grads = {k: g - common_grad for k, g in grads.items()}

        for name in mdps.keys():
            g = grads[name]
            g += individual_reg * rewards[name].data.numpy()
            rewards[name].grad = Variable(torch.Tensor(g))

        grad_history.append({k: v.grad for k, v in rewards.items()})
        reward_history.append(rewards)
        optimizer.step()
        scheduler.step()

    res = {k: (v + rewards['common']).data.numpy()
           for k, v in rewards.items() if k != 'common'}

    info = {
        'demo_counts': demo_counts,
        'expected_counts': ec_history,
        'grads': grad_history,
        'rewards': reward_history,
        'common_reward': rewards['common'].data.numpy(),
    }
    return res, info