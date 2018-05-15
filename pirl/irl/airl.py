import pickle

from baselines.common.vec_env import VecEnvWrapper
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
import gym
import numpy as np
import tensorflow as tf

from rllab.envs.base import Env, EnvSpec, Step
from sandbox.rocky.tf.envs.base import TfEnv, to_tf_space
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import convert_gym_space
from sandbox.rocky.tf.misc import tensor_utils

from rllab.core.serializable import Serializable
from rllab.spaces import Box
from sandbox.rocky.tf.distributions.diagonal_gaussian import DiagonalGaussian
from sandbox.rocky.tf.policies.base import StochasticPolicy
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy

from airl.algos.irl_trpo import IRLTRPO
from airl.models.airl_state import AIRL
from airl.utils.log_utils import rllab_logdir

from pirl.agents.sample import SampleVecMonitor
from pirl.utils import vectorized


class VecInfo(VecEnvWrapper):
    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        obs, rewards, dones, env_infos = self.venv.step_wait()
        env_infos = tensor_utils.stack_tensor_dict_list(env_infos)
        return obs, rewards, dones, env_infos

    def terminate(self):
        return self.close()


def _make_vec_env(env_fns, parallel):
    if parallel and len(env_fns) > 1:
        return SubprocVecEnv(env_fns)
    else:
        return DummyVecEnv(env_fns)


class VecGymEnv(Env):
    def __init__(self, venv):
        self.venv = venv
        self._observation_space = convert_gym_space(venv.observation_space)
        self._action_space = convert_gym_space(venv.action_space)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def terminate(self):
        pass

    @property
    def vectorized(self):
        return True

    def vec_env_executor(self, n_envs, max_path_length):
        # SOMEDAY: make these parameters have an effect?
        # We're powerless as the environments have already been created.
        # But I'm not too bothered by this, as we can tweak them elsewhere.
        return VecInfo(self.venv)


class GaussianPolicy(StochasticPolicy, Serializable):
    def __init__(self, env_spec, name=None, mean=0.0, log_std=1.0):
        with tf.variable_scope(name):
            assert isinstance(env_spec.action_space, Box)
            Serializable.quick_init(self, locals())

            self.action_dim = env_spec.action_space.flat_dim
            self._dist = DiagonalGaussian(self.action_dim)
            self.mean = mean * np.ones(self.action_dim)
            self.log_std = log_std * np.ones(self.action_dim)
            self.mean_tf = tf.constant(self.mean, dtype=tf.float32)
            self.log_std_tf = tf.constant(self.log_std, dtype=tf.float32)

            self.dummy_var = tf.get_variable(name='dummy', shape=self.action_dim)

            super(GaussianPolicy, self).__init__(env_spec=env_spec)

    @property
    def vectorized(self):
        return True

    def get_action(self, observation):
        rnd = np.random.normal(size=(self.action_dim, ))
        action = self.mean + np.exp(self.log_std) * rnd
        info = dict(mean=self.mean, log_std=self.log_std)
        return action, info

    def get_actions(self, observations):
        n = len(observations)
        shape = (n, self.action_dim)
        mean = np.broadcast_to(self.mean, shape)
        log_std = np.broadcast_to(self.log_std, shape)
        rnd = np.random.normal(size=shape)
        action = mean + np.exp(log_std) * rnd
        info = dict(mean=mean, log_std=log_std)
        return action, info

    @property
    def distribution(self):
        return self._dist

    def dist_info_sym(self, obs_var, state_info_vars):
        return dict(mean=self.mean_tf, log_std=self.log_std_tf)

    def dist_info(self, obs, state_infos):
        return dict(mean=self.mean, log_std=self.log_std)

    def get_params_internal(self, **tags):
        # Fake it as RLLab gets confused if we have no variables
        return [self.dummy_var]


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

@vectorized(True)
def irl(venv, trajectories, discount, log_dir, tf_cfg,
        model_cfg={}, policy_cfg=None, training_cfg={}):
    envs = VecGymEnv(venv)
    envs = TfEnv(envs)

    experts = _convert_trajectories(trajectories)
    model_kwargs = {'state_only': True, 'max_itrs': 10}
    model_kwargs.update(model_cfg)

    train_graph = tf.Graph()
    with train_graph.as_default():
        if policy_cfg is None:
            policy_cfg = {'policy': GaussianMLPPolicy, 'hidden_sizes': (32, 32)}
        policy_fn = policy_cfg.pop('policy')
        policy = policy_fn(name='policy', env_spec=envs.spec, **policy_cfg)

        training_kwargs = {
            'n_itr': 1000,
            'batch_size': 10000,
            'max_path_length': 500,
            'irl_model_wt': 1.0,
            'entropy_weight': 0.1,
        }
        training_kwargs.update(training_cfg)
        irl_model = AIRL(env_spec=envs.spec, expert_trajs=experts, **model_kwargs)
        algo = IRLTRPO(
            env=envs,
            policy=policy,
            irl_model=irl_model,
            discount=discount,
            sampler_args=dict(n_envs=venv.num_envs),
            store_paths=True,
            zero_environment_reward=True,
            baseline=LinearFeatureBaseline(env_spec=envs.spec),
            **training_kwargs
        )
        with rllab_logdir(algo=algo, dirname=log_dir):
            with tf.Session(config=tf_cfg):
                algo.train()

                reward_params = irl_model.get_params()
                policy_pkl = pickle.dumps(policy)

    reward = model_kwargs, reward_params
    return reward, policy_pkl


def sample(venv, policy_pkl, num_episodes, seed, tf_cfg):
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


def _setup_model(env, new_reward, tf_cfg):
    env_spec = EnvSpec(
        observation_space=to_tf_space(convert_gym_space(env.observation_space)),
        action_space=to_tf_space(convert_gym_space(env.action_space)))
    model_kwargs, reward_params = new_reward
    infer_graph = tf.Graph()
    with infer_graph.as_default():
        irl_model = AIRL(env_spec=env_spec, expert_trajs=None, **model_kwargs)
        sess = tf.Session(config=tf_cfg)
        with sess.as_default():
            irl_model.set_params(reward_params)
    return sess, irl_model


class AIRLRewardWrapper(gym.Wrapper):
    """Wrapper for a Env, using a reward network."""
    def __init__(self, env, new_reward, tf_cfg):
        self.sess, self.irl_model = _setup_model(env, new_reward, tf_cfg)
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


class AIRLVecRewardWrapper(VecEnvWrapper):
    """Wrapper for a VecEnv, using a reward network."""
    def __init__(self, venv, new_reward, tf_cfg):
        self.sess, self.irl_model = _setup_model(venv, new_reward, tf_cfg)
        super().__init__(venv)

    def step_async(self, actions):
        self.last_actions = actions
        self.venv.step_async(actions)

    def step_wait(self):
        obs, _old_rewards, dones, info = self.venv.step_wait()
        feed_dict = {self.irl_model.act_t: np.array(self.last_actions),
                     self.irl_model.obs_t: np.array(obs)}
        new_reward = self.sess.run(self.irl_model.reward, feed_dict=feed_dict)
        return obs, new_reward.flat, dones, info

    def reset(self):
        return self.venv.reset()

    def close(self):
        self.sess.close()
        self.venv.close()


def airl_reward_wrapper(env, new_reward, tf_cfg):
    cls = AIRLVecRewardWrapper if hasattr(env, 'num_envs') else AIRLRewardWrapper
    return cls(env, new_reward, tf_cfg)