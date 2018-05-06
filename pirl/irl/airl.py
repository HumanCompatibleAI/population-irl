import pickle

from baselines.common.vec_env import VecEnvWrapper
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
import cloudpickle
import gym
import numpy as np
import tensorflow as tf

from rllab.envs.base import Env, Step
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import convert_gym_space
from sandbox.rocky.tf.misc import tensor_utils

from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from airl.algos.irl_trpo import IRLTRPO
from airl.models.airl_state import AIRL
from airl.utils.log_utils import rllab_logdir

from pirl.agents.sample import SampleVecMonitor


class VecInfo(VecEnvWrapper):
    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        obs, rewards, dones, env_infos = self.venv.step_wait()
        env_infos = tensor_utils.stack_tensor_dict_list(env_infos)
        return obs, rewards, dones, env_infos

    def terminate(self):
        return self.close()


def _set_max_path_length(orig_env, max_path_length):
    env = orig_env
    while True:
        if isinstance(env, gym.Wrapper):
            if env.class_name() == 'TimeLimit':
                env._max_episode_steps = max_path_length
                return orig_env
            env = env.env
        else:
            return gym.wrappers.TimeLimit(orig_env, max_episode_steps=max_path_length)

def _make_vec_env(env_fns, parallel):
    if parallel and len(env_fns) > 1:
        return SubprocVecEnv(env_fns)
    else:
        return DummyVecEnv(env_fns)


class VecGymEnv(Env):
    def __init__(self, env_fns, parallel):
        self.env_fns = env_fns
        self.parallel = parallel
        env = env_fns[0]()
        self.env = env
        self._observation_space = convert_gym_space(env.observation_space)
        self._action_space = convert_gym_space(env.action_space)
        self._horizon = env._max_episode_steps

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def horizon(self):
        return self._horizon

    def reset(self):
        return self.env.reset()

    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        return Step(next_obs, reward, done, **info)

    def render(self):
        self.env.render()

    def terminate(self):
        self.env.close()

    @property
    def vectorized(self):
        return True

    def vec_env_executor(self, n_envs, max_path_length):
        assert n_envs <= len(self.env_fns)
        env_fns = [lambda: _set_max_path_length(fn(), max_path_length)
                   for fn in self.env_fns[:n_envs]]
        return VecInfo(_make_vec_env(env_fns, self.parallel))


def _convert_trajectories(trajs):
    '''Convert trajectories from format used in PIRL to that expected in AIRL.

    Args:
        - trajs: trajectories in AIRL format. That is, a list of 2-tuples (obs, actions),
          where obs and actions are equal-length lists containing observations and actions.
    Returns: trajectories in AIRL format.
        A list of dictionaries, containing keys 'observations' and 'actions', with values that are equal-length
        numpy arrays.'''
    return [{'observations': np.array(obs), 'actions': np.array(actions)}
            for obs, actions in trajs]


def irl(env_fns, trajectories, discount, log_dir, tf_cfg, fusion=False,
        parallel=True, policy_cfg={}, irl_cfg={}):
    experts = _convert_trajectories(trajectories)
    num_envs = len(env_fns)
    env = VecGymEnv(env_fns, parallel)
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
            sampler_args=dict(n_envs=num_envs),
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

    env.terminate()

    reward = cloudpickle.dumps(make_irl_model), reward_params
    return reward, policy_pkl


def sample(env_fns, policy_pkl, num_episodes, seed, tf_cfg, parallel=True):
    venv = _make_vec_env(env_fns, parallel)
    venv = SampleVecMonitor(venv)

    infer_graph = tf.Graph()
    with infer_graph.as_default():
        tf.set_random_seed(seed)  # seed to make results reproducible
        with tf.Session(config=tf_cfg):
            policy = pickle.loads(policy_pkl)

            completed = 0
            obs = venv.reset()
            while completed < num_episodes:
                a, _info = policy.get_actions(obs)
                obs, _r, dones, _info = venv.step(a)
                completed += np.sum(dones)

            return venv.trajectories

    venv.close()


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

    def step(self, action):
        obs, old_reward, done, info = self.env.step(action)
        feed_dict = {self.irl_model.act_t: np.array([action]),
                     self.irl_model.obs_t: np.array([obs])}
        new_reward = self.sess.run(self.irl_model.reward, feed_dict=feed_dict)
        return obs, new_reward[0][0], done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def close(self):
        self.sess.close()
        self.env.close()