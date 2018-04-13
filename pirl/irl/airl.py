import tensorflow as tf

from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv

from airl.envs.env_utils import CustomGymEnv
from airl.algos.irl_trpo import IRLTRPO
from airl.models.airl_state import *
from airl.utils.log_utils import rllab_logdir, load_latest_experts, load_latest_experts_multiple_runs
from airl.utils.hyper_sweep import run_sweep_parallel, run_sweep_serial

def _convert_trajectories(trajs):
    '''Convert trajectories from format used in PIRL to that expected in AIRL.

    Args:
        - trajs: trajectories in AIRL format. That is, a list of 2-tuples (obs, actions),
          where obs and actions are equal-length lists containing observations and actions.
    Returns: trajectories in AIRL format.
        A list of dictionaries, containing keys 'observations' and 'actions', with values that are equal-length
        numpy arrays.'''
    return [{'observations': np.array(obs), 'actions': np.array(actions)} for obs, actions in trajs]

def irl(env, trajectories, discount, log_dir, tf_config, fusion=False):
    experts = _convert_trajectories(trajectories)
    irl_model = AIRL(env=env, expert_trajs=experts, state_only=True, fusion=fusion, max_itrs=10)
    policy = GaussianMLPPolicy(name='policy', env_spec=env.spec, hidden_sizes=(32, 32))
    algo = IRLTRPO(
        env=env,
        policy=policy,
        irl_model=irl_model,
        n_itr=1000,
        batch_size=10000,
        max_path_length=500,
        discount=discount,
        store_paths=True,
        irl_model_wt=1.0,
        entropy_weight=0.1,
        zero_environment_reward=True,
        baseline=LinearFeatureBaseline(env_spec=env.spec),
    )
    with rllab_logdir(algo=algo, dirname=log_dir):
        with tf.Session(tf_config):
            algo.train()