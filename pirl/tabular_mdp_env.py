from gym import Env, spaces
from gym.utils import seeding

import numpy as np

def discrete_sample(prob, rng):
    """Sample from discrete probability distribution, each row of prob
       specifies class probabilities."""
    return (np.cumsum(prob) > rng.rand()).argmax()

def _check_probability(x, axis, tol=1e-6):
    assert np.all(x >= 0)
    assert np.all(abs(x.sum(axis) - 1) < tol)

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

        self._transition = np.array(transition)
        self._reward = np.array(reward)
        self._initial_state = np.array(initial_state)
        self._terminal = np.array(terminal)

        # Check dimensions
        S, A, S2 = self._transition.shape
        assert S == S2
        assert reward.shape == (S, )
        assert initial_state.shape == (S, )
        assert terminal.shape == (S, )

        # Check probability distributions
        _check_probability(self._transition, 2)
        _check_probability(self._initial_state, 0)

        # State/action space
        self.observation_space = spaces.Discrete(S)
        self.action_space = spaces.Discrete(A)

        self.seed()
        self.reset()

    def seed(self, seed=None):
         self.rng, seed = seeding.np_random(seed)

         return [seed]

    def reset(self):
        self._state = discrete_sample(self._initial_state, self.rng)

    def step(self, action):
        p = self._transition[self._state, action, :]
        self._state = discrete_sample(p, self.rng)
        r = self._reward[self._state]
        finished = self._terminal[self._state]
        info = {"prob": p[self._state]}
        return (self._state, r, finished, info)

    @property
    def state(self):
        return self._state

    @property
    def transition(self):
        return self._transition

    @property
    def reward(self):
        return self._reward

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def terminal(self):
        return self._terminal