import pickle

import cloudpickle
import gym
import numpy as np
import tensorflow as tf

from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv

from airl.algos.irl_trpo import IRLTRPO
from airl.models.airl_state import AIRL
from airl.utils.log_utils import rllab_logdir

def _convert_trajectories(trajs):
    '''Convert trajectories from format used in PIRL to that expected in AIRL.

    Args:
        - trajs: trajectories in AIRL format. That is, a list of 2-tuples (obs, actions),
          where obs and actions are equal-length lists containing observations and actions.
    Returns: trajectories in AIRL format.
        A list of dictionaries, containing keys 'observations' and 'actions', with values that are equal-length
        numpy arrays.'''
    return [{'observations': np.array(obs), 'actions': np.array(actions)} for obs, actions in trajs]


def irl(env, trajectories, discount, log_dir, tf_cfg, fusion=False,
        policy_cfg={}, irl_cfg={}):
    experts = _convert_trajectories(trajectories)
    env = GymEnv(env, record_video=False, record_log=False)
    env = TfEnv(env)
    make_irl_model = lambda : AIRL(env=env, expert_trajs=experts,
                                   state_only=True, fusion=fusion, max_itrs=10)
    train_graph = tf.Graph()
    with train_graph.as_default():
        policy_kwargs = {'hidden_sizes': (32, 32)}
        policy_kwargs.update(policy_cfg)
        policy = GaussianMLPPolicy(name='policy', env_spec=env.spec, **policy_kwargs)

        irl_kwargs = {
            'n_itr': 1000,
            'batch_size': 10000,
            'max_path_length': 500,
            'irl_model_wt': 1.0,
            'entropy_weight': 0.1,
        }
        irl_kwargs.update(irl_cfg)
        irl_model = make_irl_model()
        algo = IRLTRPO(
            env=env,
            policy=policy,
            irl_model=irl_model,
            discount=discount,
            store_paths=True,
            zero_environment_reward=True,
            baseline=LinearFeatureBaseline(env_spec=env.spec),
            **irl_kwargs
        )
        with rllab_logdir(algo=algo, dirname=log_dir):
            with tf.Session(config=tf_cfg):
                algo.train()

                reward_params = irl_model.get_params()
                policy_pkl = pickle.dumps(policy)

    reward = cloudpickle.dumps(make_irl_model), reward_params
    return reward, policy_pkl


def sample(env, policy_pkl, num_episodes, seed, tf_cfg):
    infer_graph = tf.Graph()
    with infer_graph.as_default():
        with tf.Session(config=tf_cfg):
            # Seed to make results reproducible
            seed = gym.utils.seeding.create_seed(seed)
            env.seed(seed)
            tf.set_random_seed(seed)

            policy = pickle.loads(policy_pkl)

            def helper():
                '''Samples from environment for an entire episode.'''
                observations = []
                actions = []
                rewards = []

                obs = env.reset()
                done = False
                while not done:
                    observations.append(obs[0])
                    a, _info = policy.get_action(obs)
                    actions.append(a)
                    obs, r, done, info = env.step(a)
                    rewards.append(r)

                return observations, actions, rewards

            return [helper() for i in range(num_episodes)]


class AIRLRewardWrapper(gym.Wrapper):
    """Wrapper for a gym.Env replacing with a new reward matrix."""
    def __init__(self, env, new_reward, tf_cfg):
        make_irl_model_pkl, reward_params = new_reward
        make_irl_model = cloudpickle.loads(make_irl_model_pkl)
        infer_graph = tf.Graph()
        with infer_graph.as_default():
            self.irl_model = make_irl_model()
            self.sess = tf.Session(config=tf_cfg)
            with self.sess.as_default():
                self.irl_model.set_params(reward_params)
        super().__init__(env)

    def step(self, a):
        obs, old_reward, done, info = self.env.step(a)
        feed_dict = {self.irl_model.act_t: np.array([a]), self.irl_model.obs_t: np.array([obs])}
        new_reward = self.sess.run(self.irl_model.reward, feed_dict=feed_dict)[0][0]
        return obs, new_reward, done, info
