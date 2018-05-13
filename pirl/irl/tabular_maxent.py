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

from pirl.utils import getattr_unwrapped, TrainingIterator

#TODO: fully torchize?

def empirical_counts(nS, trajectories, discount):
    """Compute empirical state-action feature counts from trajectories."""
    counts = np.zeros((nS, ))
    discounted_steps = 0
    for states, actions in trajectories:
        incr = np.cumprod([1] + [discount] * (len(states) - 1))
        counts += np.bincount(states, weights=incr, minlength=nS)
        discounted_steps += np.sum(incr)
    return counts / discounted_steps

def max_ent_policy(transition, reward, horizon, discount):
    """Backward pass of algorithm 1 of Ziebart (2008).
       This corresponds to maximum entropy.
       WARNING: You probably want to use max_causal_ent_policy instead.
       See discussion in section 6.2.2 of Ziebart's PhD thesis (2010)."""
    nS = transition.shape[0]
    logsc = np.zeros(nS)  # TODO: terminal states only?
    with np.warnings.catch_warnings():
        np.warnings.filterwarnings('ignore', 'divide by zero encountered in log')
        logt = np.nan_to_num(np.log(transition))
    reward = reward.reshape(nS, 1, 1)
    for i in range(horizon):
        # Ziebart (2008) never describes how to handle discounting. This is a
        # backward pass: so on the i'th iteration, we are computing the
        # frequency a state/action is visited at the (horizon-i-1)'th position.
        # So we should multiply reward by discount ** (horizon - i - 1).
        cur_discount = discount ** (horizon - i - 1)
        x = logt + (cur_discount * reward) + logsc.reshape(1, 1, nS)
        logac = sp_lse(x, axis=2)
        logsc = sp_lse(logac, axis=1)
    return np.exp(logac - logsc.reshape(nS, 1))

def max_causal_ent_policy(transition, reward, horizon, discount):
    """Soft Q-iteration, theorem 6.8 of Ziebart's PhD thesis (2010)."""
    nS, nA, _ = transition.shape
    V = np.zeros(nS)
    for i in range(horizon):
        Q = reward.reshape(nS, 1) + discount * (transition * V).sum(2)
        V = sp_lse(Q, axis=1)
    return np.exp(Q - V.reshape(nS, 1))

def expected_counts(policy, transition, initial_states, horizon, discount):
    """Forward pass of algorithm 1 of Ziebart (2008)."""
    nS = transition.shape[0]
    counts = np.zeros((nS, horizon + 1))
    counts[:, 0] = initial_states
    for i in range(1, horizon + 1):
        counts[:, i] = np.einsum('i,ij,ijk->k', counts[:, i-1],
                                 policy, transition) * discount
    if discount == 1:
        renorm = horizon + 1
    else:
        renorm = (1 - discount ** (horizon + 1)) / (1 - discount)
    return np.sum(counts, axis=1) / renorm

def policy_loss(policy, trajectories):
    loss = 0
    log_policy = np.log(policy)
    for states, actions in trajectories:
        loss += np.sum(log_policy[states, actions])
    return loss

default_optimizer = functools.partial(torch.optim.Adam, lr=1e-1)
default_scheduler = {
    max_ent_policy: functools.partial(
        torch.optim.lr_scheduler.ExponentialLR, gamma=1.0
    ),
    max_causal_ent_policy: functools.partial(
        torch.optim.lr_scheduler.ExponentialLR, gamma=0.999
    ),
}

def irl(env_fns, trajectories, discount, log_dir=None, demo_counts=None,
        horizon=None, planner=max_causal_ent_policy,
        regularize=None, common_reward=None, optimizer=None, scheduler=None,
        num_iter=5000, log_every=100, log_expensive_every=1000):
    """
    Args:
        - env_fns(list[() -> TabularMdpEnv]): MDP trajectories were drawn from.
        - trajectories(list): expert trajectories; exclusive with demo_counts.
            List containing one (states, actions) pair for each trajectory,
            where states and actions are lists containing all visited
            states/actions in that trajectory.
        - discount(float): between 0 and 1.
            Should match that of the agent generating the trajectories.
        - log_dir: ignored.
        - demo_counts(array): expert visitation frequency; exclusive with trajectories.
            The expected visitation frequency of the optimal policy.
            Must supply horizon with this argument.
        - horizon(int): optional, must be supplied if demo_counts used.
        - planner(callable): max_ent_policy or max_causal_ent_policy.
        - regularize(float): regularization constant; requires common_reward.
        - common_reward(list): regularize reward to be close to this.
            Same type as the return of this funcion.
            Argument is exclusive with demo_counts.
        - optimizer(callable): a callable returning a torch.optim object.
            The callable is called with an iterable of parameters to optimize.
        - scheduler(callable): a callable returning a torch.optim.lr_scheduler.
            The callable is called with a torch.optim optimizer object.
        - learning_rate(float): for Adam optimizer.
        - num_iter(int): number of iterations of optimization process.

    Returns (reward, policy) where:
        reward(list): estimated reward for each state in the MDP.
        policy(array): array of dimensions S * A, describing a stochastic policy.
    """
    assert sum([trajectories is None, demo_counts is None]) == 1
    assert (regularize is None) ^ (common_reward is None) == 0
    assert (regularize is None) or (demo_counts is None)

    mdp = env_fns[0]()
    transition = getattr_unwrapped(mdp, 'transition')
    initial_states = getattr_unwrapped(mdp, 'initial_states')
    if horizon is None:
        horizon = getattr_unwrapped(mdp, '_max_episode_steps')
    nS, _, _ = transition.shape

    num_trajs = None
    if trajectories is not None:
        demo_counts = empirical_counts(nS, trajectories, discount)
        num_trajs = len(trajectories)

    reward = Variable(torch.zeros(nS), requires_grad=True)
    if optimizer is None:
        optimizer = default_optimizer
    if scheduler is None:
        scheduler = default_scheduler[planner]
    optimizer = optimizer([reward])
    scheduler = scheduler(optimizer)

    it = TrainingIterator(num_iter, 'irl', heartbeat_iters=100)
    for i in it:
        pol = planner(transition, reward.data.numpy(), horizon, discount)
        ec = expected_counts(pol, transition, initial_states, horizon, discount)
        optimizer.zero_grad()

        grad = ec - demo_counts
        if regularize is not None:  # optionally, regularize
            delta = reward.data.numpy() - common_reward
            if num_trajs > 0:
                grad = grad + (regularize / num_trajs) * delta
            else:
                grad = delta
        reward.grad = Variable(torch.Tensor(grad))
        reward.grad = Variable(torch.Tensor(ec - demo_counts))
        optimizer.step()
        scheduler.step()

        if trajectories is not None and i % log_expensive_every == 0:
            # loss is expensive to compute
            loss = policy_loss(pol, trajectories)
            it.record('loss', loss)
        if i % log_every == 0:
            it.record('expected_counts', ec)
            it.record('grads', reward.grad.data.numpy())
            it.record('rewards', reward.data.numpy().copy())

    #TODO: log to disk (used to return it.vals, but this conflicts with new API)
    return reward.data.numpy(), pol


def metalearn(envs_fns, trajectories, discount, individual_reg=None, **kwargs):
    """
    Args:
        - envs_fns(dict<list<() -> TabularMdpEnv)>): MDPs trajectories were drawn from.
            Dictionary containing MDPs trajectories, of the same name, were
            drawn from. MDPs must have the same state/action spaces, but may
            have different dynamics and reward functions.
        - trajectories(dict<list>): observed trajectories.
            Dictionary of lists containing one (states, actions) pair for each
            trajectory, where states and actions are lists containing all
            visited states/actions in that trajectory.
        - discount(float): between 0 and 1.
            Should match that of the agent generating the trajectories.
        - individual_reg(float): regularization factor for per-agent reward.
            Penalty factor applied to the l_2 norm of per-agent reward matrices,
            divided by the number of trajectories.
        - kwargs: passed-through to irl().

    Returns mean_reward, a list containing the estimate reward for each state.
    """
    res = {k: irl(env_fns, trajectories[k], discount, **kwargs)
           for k, env_fns in envs_fns.items()}
    rewards = {k: r for k, (r, v) in res.items()}
    mean_reward = np.mean(list(rewards.values()), axis=0)
    return mean_reward

def finetune(mean_reward, env_fns, trajectories, discount, log_dir=None, individual_reg=1e-2, **kwargs):
    """First argument is result of metalearn; individual_reg is regularization
       factor; remaining arguments are passed-through to irl."""
    return irl(env_fns, trajectories, discount, log_dir,
               common_reward=mean_reward, regularize=individual_reg,
               **kwargs)