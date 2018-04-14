import itertools
import pytest

import gym
import numpy as np

from pirl.experiments import experiments
from pirl.agents import tabular
from pirl.irl import tabular_maxent

def demean(x):
    return x - np.mean(x)

@pytest.mark.parametrize("env_name,planner,discount",
    itertools.product(
        ['pirl/GridWorld-Jungle-4x4-Liquid-v0'],
        [tabular_maxent.max_causal_ent_policy, tabular_maxent.max_ent_policy],
        [1.00, 0.99, 0.9],
    )
)
def test_irl(env_name, planner, discount):
    """Tests IRL algorithm on env_name. First, synthetic data is generated
       using planner with the true reward function. Then, Max(Causal)Ent IRL
       with the same planner is used to infer the reward function, using both:
         (a) true expected counts from the expert policy;
         (b) empirical estimates using the trajectories.
       The loss and inf/L2 norm error of the reward are then compared against
       pre-defined thresholds."""
    # Config parameters
    num_trajectories = 100000
    num_iter = 5000
    seed = 42
    thresholds = {
        tabular_maxent.max_causal_ent_policy: {
            'counts': (1e-5, 5e-4, 1e-3),
            'traj': (1e-4, 0.5, 0.5),
        },
        tabular_maxent.max_ent_policy: {
            'counts': (1e-5, 1e-2, 2e-2),
            # Reward becomes very sensitive to visitation frequency when it
            # is inferred to be a large negative number. This can lead to large
            # errors in reward (but loss should still be small).
            'traj': (1e-3, 100, 100),
        },
    }

    # Setup
    env = gym.make(env_name)
    env_planner = tabular.env_wrapper(planner)
    optimal_policy = env_planner(env, discount=discount)
    trajectories = experiments.synthetic_data.func(env_name, env, 'max_ent', num_trajectories,
                                                   seed, '/tmp/test-pirl/', 10, optimal_policy)
    optimal_loss = tabular_maxent.policy_loss(optimal_policy, trajectories)
    optimal_counts = tabular_maxent.expected_counts(optimal_policy,
                                                    env.unwrapped.transition,
                                                    env.unwrapped.initial_states,
                                                    env._max_episode_steps,
                                                    discount=discount)

    def check_reward(actual_reward, rel_tol, inf_tol, l2_tol):
        actual_policy = env_planner(env, reward=actual_reward,
                                    discount=discount)
        actual_loss = tabular_maxent.policy_loss(actual_policy, trajectories)

        error = demean(actual_reward) - demean(env.unwrapped.reward)
        inf_norm_error = np.linalg.norm(error, float('inf'))
        l2_norm_error = np.linalg.norm(error, 2)

        print('actual/optimal loss: {}/{}, inf error: {}, l2 error: {}'.format(
            actual_loss, optimal_loss, inf_norm_error, l2_norm_error
        ))
        assert actual_loss == pytest.approx(optimal_loss, rel=rel_tol)
        assert inf_norm_error < inf_tol
        assert l2_norm_error < l2_tol

    # Test IRL from counts
    count_reward, _ = tabular_maxent.irl(env, trajectories=None,
                                         discount=discount, planner=planner,
                                         demo_counts=optimal_counts,
                                         horizon=env._max_episode_steps,
                                         num_iter=num_iter)
    check_reward(count_reward, *thresholds[planner]['counts'])

    # Test IRL from trajectories
    traj_reward, _ = tabular_maxent.irl(env, trajectories,
                                        discount=discount, planner=planner,
                                        num_iter=num_iter)
    check_reward(traj_reward, *thresholds[planner]['traj'])

