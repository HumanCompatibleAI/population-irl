from gym import Env, spaces
from gym.utils import seeding

import numpy as np

def discrete_sample(prob, rng):
    """Sample from discrete probability distribution, each row of prob
       specifies class probabilities."""
    return (np.cumsum(prob) > rng.rand()).argmax()

def _check_probability(x, axis, tol=1e-6):
    assert np.all(x >= 0)
    assert np.all((x.sum(axis) - 1).abs() < tol)

class TabularMdpEnv(Env):
    #TODO: Do I want to set reward_range?
    #TODO: am I ok with reward being a function of state?
    def __init__(self, transition, reward, initial_state, terminal):
        """Creates an environment for an MDP. The state and action spaces
           are consecutive integer sequences, with their size inferred from the
           dimensions of the transition matrix.

        Args:
            transition (S*A*S array-like): transition probability matrix
                transition[s, a, t] gives probability of moving to state t
                having taken action a in state s.
            reward (S array-like): reward per state.
            initial_state (S array-like): probability distribution over states.
            terminal (S array-like): boolean mask for if episode-ending.
        """
        super().__init__()

        self.transition = np.array(transition)
        self.reward = np.array(reward)
        self.initial_state = np.array(initial_state)
        self.terminal = np.array(terminal)

        # Check dimensions
        S, A, S2 = self.transition.shape
        assert S == S2
        assert reward.shape == (S, )
        assert initial_state.shape == (S, )
        assert terminal.shape == (S, )

        # Check probability distributions
        _check_probability(self.transition, 2)
        _check_probability(self.initial_state, 0)

        # State/action space
        self.observation_space = spaces.Discrete(S)
        self.action_space = spaces.Discrete(A)

        self.seed()
        self.reset()

    def seed(self, seed=None):
         self.rng, seed = seeding.np_random(seed)

         return [seed]

    def reset(self):
        self.state = discrete_sample(self.initial_state, self.rng)

    def step(self, action):
        p = self.transition[self.state, action, :]
        self.state = discrete_sample(p, self.rng)
        r = self.reward[self.state]
        finished = self.terminal[self.state]
        info = {"prob": p[self.state]}
        return (self.state, r, finished, info)

    @property
    def transition(self):
        return self.transition

    @property
    def reward(self):
        return self.reward

    @property
    def initial_state(self):
        return self.initial_state