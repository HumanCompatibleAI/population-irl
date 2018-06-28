import functools
import pickle
import random

from baselines.common.vec_env import VecEnvWrapper
import gym
import numpy as np
import os.path as osp
import tensorflow as tf

from rllab.envs.base import Env, EnvSpec
import rllab.misc.logger as rl_logger
from sandbox.rocky.tf.envs.base import TfEnv, to_tf_space
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import convert_gym_space

from rllab.core.serializable import Serializable
from rllab.spaces import Box
from sandbox.rocky.tf.distributions.diagonal_gaussian import DiagonalGaussian
from sandbox.rocky.tf.policies.base import StochasticPolicy
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy

from airl.algos.irl_trpo import IRLTRPO
from airl.models.airl_state import AIRL as AIRLStateOnly
from airl.models.imitation_learning import AIRLStateAction
from airl.utils.log_utils import rllab_logdir

from pirl.agents.sample import SampleVecMonitor

class VecInfo(VecEnvWrapper):
    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        obs, rewards, dones, env_infos = self.venv.step_wait()
        #SOMEDAY: handle env_infos with different keys
        #The problem is bench.Monitor adds an episode key only when an episode
        #ends. stack_tensor_dict_list assumes constant keys, so this breaks
        #when some but not all envirnoments are done.
        #env_infos is only used for some debugging code, so just removing this.
        #env_infos = tensor_utils.stack_tensor_dict_list(env_infos)
        return obs, rewards, dones, {}

    def terminate(self):
        # Normally we'd close environments, but pirl.experiments handles this
        pass


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
        # Normally we'd close environments, but pirl.experiments handles this.
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


def irl(venv, trajectories, discount, seed, log_dir, *, tf_cfg, model_cfg=None,
        policy_cfg=None, training_cfg={}):
    envs = VecGymEnv(venv)
    envs = TfEnv(envs)
    experts = _convert_trajectories(trajectories)

    train_graph = tf.Graph()
    with train_graph.as_default():
        tf.set_random_seed(seed)

        if model_cfg is None:
            model_cfg = {'model': AIRLStateOnly,
                         'state_only': True,
                         'max_itrs': 10}
        model_kwargs = dict(model_cfg)
        model_cls = model_kwargs.pop('model')
        irl_model = model_cls(env_spec=envs.spec, expert_trajs=experts,
                              **model_kwargs)

        if policy_cfg is None:
            policy_cfg = {'policy': GaussianMLPPolicy, 'hidden_sizes': (32, 32)}
        else:
            policy_cfg = dict(policy_cfg)
        policy_fn = policy_cfg.pop('policy')
        policy = policy_fn(name='policy', env_spec=envs.spec, **policy_cfg)

        training_kwargs = {
            'n_itr': 1000,
            'batch_size': 10000,
            'max_path_length': 500,
            'irl_model_wt': 1.0,
            'entropy_weight': 0.1,
            # paths substantially increase storage requirements
            'store_paths': False,
        }
        training_kwargs.update(training_cfg)
        algo = IRLTRPO(
            env=envs,
            policy=policy,
            irl_model=irl_model,
            discount=discount,
            sampler_args=dict(n_envs=venv.num_envs),
            zero_environment_reward=True,
            baseline=LinearFeatureBaseline(env_spec=envs.spec),
            **training_kwargs
        )

        with rllab_logdir(algo=algo, dirname=log_dir):
            with tf.Session(config=tf_cfg):
                algo.train()

                reward_params = irl_model.get_params()

                # Side-effect: forces policy to cache all parameters.
                # This ensures they are saved/restored during pickling.
                policy.get_params()
                # Must pickle policy rather than returning it directly,
                # since parameters in policy will not survive across tf sessions.
                policy_pkl = pickle.dumps(policy)

    reward = model_cfg, reward_params
    return reward, policy_pkl


def metalearn(venvs, trajectories, discount, seed, log_dir, *, tf_cfg, outer_itr=1000,
              lr=1e-2, model_cfg=None, policy_cfg=None, training_cfg={},
              policy_per_task=False):
    envs = {k: TfEnv(VecGymEnv(v)) for k, v in venvs.items()}
    env_spec = list(envs.values())[0].spec
    num_envs = list(venvs.values())[0].num_envs
    tasks = list(envs.keys())

    experts = {k: _convert_trajectories(v) for k, v in trajectories.items()}

    train_graph = tf.Graph()
    with train_graph.as_default():
        tf.set_random_seed(seed)

        if model_cfg is None:
            model_cfg = {'model': AIRLStateOnly,
                         'state_only': True,
                         'max_itrs': 10}
        model_kwargs = dict(model_cfg)
        model_cls = model_kwargs.pop('model')
        irl_model = model_cls(env_spec=env_spec, **model_kwargs)

        if policy_cfg is None:
            policy_cfg = {'policy': GaussianMLPPolicy, 'hidden_sizes': (32, 32)}
        else:
            policy_cfg = dict(policy_cfg)
        policy_fn = policy_cfg.pop('policy')
        policy_fn = functools.partial(policy_fn, env_spec=env_spec, **policy_cfg)
        if policy_per_task:
            policies = {k: policy_fn(name='policy_' + k) for k in envs.keys()}
        else:
            policy = policy_fn(name='policy')
            policies = {k: policy for k in envs.keys()}

        training_kwargs = {
            'n_itr': 10,
            'batch_size': 10000,
            'max_path_length': 500,
            'irl_model_wt': 1.0,
            'entropy_weight': 0.1,
            # paths substantially increase storage requirements
            'store_paths': False,
        }
        training_kwargs.update(training_cfg)

        #TODO: avoid duplication of everything? (all that's changing is environment)
        #TODO: actually this will likely do bad things due to storing paths etc
        algos = {k: IRLTRPO(
                env=env,
                policy=policies[k],
                irl_model=irl_model,
                discount=discount,
                sampler_args=dict(n_envs=num_envs),
                zero_environment_reward=True,
                baseline=LinearFeatureBaseline(env_spec=env_spec),
                **training_kwargs
            ) for k, env in envs.items()}
        with rllab_logdir(dirname=log_dir):
            with tf.Session(config=tf_cfg) as sess:
                sess.run(tf.global_variables_initializer())
                meta_reward_params = irl_model.get_params()
                for i in range(outer_itr):
                    task = random.choice(tasks)
                    with rl_logger.prefix('outer itr {} | task'.format(i, task)):
                        irl_model.set_demos(experts[task])
                        algos[task].init_irl_params = meta_reward_params
                        algos[task].train()

                        # {meta,task}_reward_params are lists of NumPy arrays
                        task_reward_params = irl_model.get_params()
                        assert len(task_reward_params) == len(meta_reward_params)
                        for i in range(len(task_reward_params)):
                            meta, task = meta_reward_params[i], task_reward_params[i]
                            # Reptile update: meta <- meta + lr * (task - meta)
                            #TODO: use Adam optimizer?
                            meta_reward_params[i] = (1 - lr) * meta + lr * task
                        irl_model.set_params(meta_reward_params)

    reward = model_kwargs, meta_reward_params

    return reward


def finetune(metainit, venv, trajectories, discount, seed, log_dir, *,
             tf_cfg, pol_itr=900, irl_itr=100,
             model_cfg=None, policy_cfg=None, training_cfg={}):
    envs = VecGymEnv(venv)
    envs = TfEnv(envs)
    experts = _convert_trajectories(trajectories)

    train_graph = tf.Graph()
    with train_graph.as_default():
        tf.set_random_seed(seed)

        if model_cfg is None:
            model_cfg = {'model': AIRLStateOnly,
                         'state_only': True,
                         'max_itrs': 10}
        model_kwargs = dict(model_cfg)
        model_cls = model_kwargs.pop('model')
        irl_model = model_cls(env_spec=envs.spec, expert_trajs=experts,
                              **model_kwargs)

        if policy_cfg is None:
            policy_cfg = {'policy': GaussianMLPPolicy, 'hidden_sizes': (32, 32)}
        else:
            policy_cfg = dict(policy_cfg)
        policy_fn = policy_cfg.pop('policy')
        policy = policy_fn(name='policy', env_spec=envs.spec, **policy_cfg)

        training_kwargs = {
            'batch_size': 10000,
            'max_path_length': 500,
            'irl_model_wt': 1.0,
            'entropy_weight': 0.1,
            # paths substantially increase storage requirements
            'store_paths': False,
        }
        training_kwargs.update(training_cfg)
        algo = IRLTRPO(
            env=envs,
            policy=policy,
            irl_model=irl_model,
            discount=discount,
            sampler_args=dict(n_envs=venv.num_envs),
            zero_environment_reward=True,
            baseline=LinearFeatureBaseline(env_spec=envs.spec),
            train_irl=False,
            n_itr=pol_itr,
            **training_kwargs
        )

        with tf.Session(config=tf_cfg):
            _kwargs, reward_params = metainit

            # First round: just optimize the policy, do not update IRL model
            with rllab_logdir(algo=algo, dirname=osp.join(log_dir, 'pol')):
                with rl_logger.prefix('finetune policy |'):
                    algo.init_irl_params = reward_params
                    algo.train()
                    pol_params = policy.get_param_values()

            # Second round: we have a good policy (generator), update IRL
            with rllab_logdir(algo=algo, dirname=osp.join(log_dir, 'all')):
                with rl_logger.prefix('finetune all |'):
                    algo.train_irl = True
                    algo.init_pol_params = pol_params
                    algo.n_itr = irl_itr
                    algo.train()

            reward_params = irl_model.get_params()

            # Side-effect: forces policy to cache all parameters.
            # This ensures they are saved/restored during pickling.
            policy.get_params()
            # Must pickle policy rather than returning it directly,
            # since parameters in policy will not survive across tf sessions.
            policy_pkl = pickle.dumps(policy)

    reward = model_cfg, reward_params
    return reward, policy_pkl


def sample(venv, policy_pkl, num_episodes, seed, tf_cfg):
    venv = SampleVecMonitor(venv)

    infer_graph = tf.Graph()
    with infer_graph.as_default():
        tf.set_random_seed(seed)
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
    model_cfg, reward_params = new_reward
    infer_graph = tf.Graph()
    with infer_graph.as_default():
        model_kwargs = dict(model_cfg)
        model_cls = model_kwargs.pop('model')
        irl_model = model_cls(env_spec=env_spec, expert_trajs=None, **model_kwargs)
        if model_cls == AIRLStateOnly:
            reward_var = irl_model.reward
        elif model_cls == AIRLStateAction:
            reward_var = irl_model.energy
        else:
            assert False, "Unsupported model type"
        sess = tf.Session(config=tf_cfg)
        with sess.as_default():
            irl_model.set_params(reward_params)
    return sess, irl_model, reward_var


class AIRLRewardWrapper(gym.Wrapper):
    """Wrapper for a Env, using a reward network."""
    def __init__(self, env, new_reward, tf_cfg):
        self.sess, self.irl_model, self.reward_var = _setup_model(env, new_reward, tf_cfg)
        super().__init__(env)

    def step(self, action):
        obs, old_reward, done, info = self.env.step(action)
        feed_dict = {self.irl_model.act_t: np.array([action]),
                     self.irl_model.obs_t: np.array([obs])}
        new_reward = self.sess.run(self.reward_var, feed_dict=feed_dict)
        return obs, new_reward[0][0], done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def close(self):
        self.sess.close()
        self.env.close()


class AIRLVecRewardWrapper(VecEnvWrapper):
    """Wrapper for a VecEnv, using a reward network."""
    def __init__(self, venv, new_reward, tf_cfg):
        self.sess, self.irl_model, self.reward_var = _setup_model(venv, new_reward, tf_cfg)
        super().__init__(venv)

    def step_async(self, actions):
        self.last_actions = actions
        self.venv.step_async(actions)

    def step_wait(self):
        obs, _old_rewards, dones, info = self.venv.step_wait()
        feed_dict = {self.irl_model.act_t: np.array(self.last_actions),
                     self.irl_model.obs_t: np.array(obs)}
        new_reward = self.sess.run(self.reward_var, feed_dict=feed_dict)
        return obs, new_reward.flat, dones, info

    def reset(self):
        return self.venv.reset()

    def close(self):
        self.sess.close()
        self.venv.close()


def airl_reward_wrapper(env, new_reward, tf_cfg):
    cls = AIRLVecRewardWrapper if hasattr(env, 'num_envs') else AIRLRewardWrapper
    return cls(env, new_reward, tf_cfg)
