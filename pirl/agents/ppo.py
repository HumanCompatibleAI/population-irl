'''Wrapper around OpenAI Baselines PPO2 (GPU optimized).
   Based on baselines/ppo2/run_mujoco.py.'''

import functools
import logging
import joblib
import os
import os.path as osp
import pickle

import cloudpickle
from gym.utils import seeding
import numpy as np
import tensorflow as tf

from baselines.ppo2 import ppo2
from baselines import bench, logger as blogger
from baselines.common.vec_env import VecEnvWrapper
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.ppo2.policies import MlpPolicy

from pirl.agents.sample import SampleVecMonitor

logger = logging.getLogger('pirl.agents.ppo')

def make_config(tf_config):
    ncpu = 1
    config = tf.ConfigProto()
    config.CopyFrom(tf_config)
    config.allow_soft_placement=True
    config.intra_op_parallelism_threads = ncpu
    config.inter_op_parallelism_threads = ncpu
    return config


class DummyVecNormalize(VecEnvWrapper):
    """
    Vectorized environment base class
    """
    def __init__(self, venv):
        VecEnvWrapper.__init__(self, venv)

    def step_wait(self):
        return self.venv.step_wait()

    def reset(self):
        return self.venv.reset()


class ConstantStatistics(object):
    def __init__(self, running_mean):
        self.mean = running_mean.mean
        self.var = running_mean.var
        self.count = running_mean.count

    def update(self, x):
        pass

    def update_from_moments(self, _batch_mean, _batch_var, _batch_count):
        pass


def _save_stats(env_wrapper):
    venv = env_wrapper.venv
    env_wrapper.venv = None
    serialized = pickle.dumps(env_wrapper)
    env_wrapper.venv = venv
    return serialized


def _restore_stats(serialized, venv):
    env_wrapper = pickle.loads(serialized)
    env_wrapper.venv = venv
    env_wrapper.num_envs = venv.num_envs
    if hasattr(env_wrapper, 'ret'):
        env_wrapper.ret = np.zeros(env_wrapper.num_envs)
    return env_wrapper


def _make_const(norm):
    '''Monkey patch classes such as VecNormalize that use a
       RunningMeanStd (or compatible class) to keep track of statistics.'''
    for k, v in norm.__dict__.items():
        if hasattr(v, 'update_from_moments'):
            setattr(norm, k, ConstantStatistics(v))


def train_continuous(env_fns, discount, log_dir, tf_config, num_timesteps,
                     parallel=True, norm=True):
    '''Policy with hyperparameters optimized for continuous control environments
       (e.g. MuJoCo). Returns log_dir, where the trained policy is saved.'''
    blogger.configure(dir=log_dir)
    checkpoint_dir = osp.join(blogger.get_dir(), 'checkpoints')
    os.makedirs(checkpoint_dir)

    def bench_helper(env_fn, i):
        return bench.Monitor(env_fn(), osp.join(log_dir, 'env{}'.format(i)))
    env_fns = [functools.partial(bench_helper, env_fn, i)
               for i, env_fn in enumerate(env_fns)]
    num_envs = len(env_fns)
    make_envs = SubprocVecEnv if (parallel and num_envs > 1) else DummyVecEnv
    make_vec_normalize = VecNormalize if norm else DummyVecNormalize
    envs = make_envs(env_fns)

    train_graph = tf.Graph()

    with train_graph.as_default():
        with tf.Session(config=make_config(tf_config)):
            norm_envs = make_vec_normalize(envs)

            policy = MlpPolicy
            nsteps = 2048 // num_envs
            learner = ppo2.learn(
                policy=policy, env=norm_envs, nsteps=nsteps, nminibatches=32,
                lam=0.95, gamma=discount, noptepochs=10, log_interval=1,
                ent_coef=0.0,
                lr=3e-4,
                cliprange=0.2,
                total_timesteps=num_timesteps,
                save_interval=4)
            best_mean_reward = None
            best_checkpoint = None
            for update, mean_reward, make_model, params in learner:
                # joblib cannot pickle closures, so use cloudpickle first
                make_model_pkl = cloudpickle.dumps(make_model)
                model = make_model_pkl, params
                policy = model, _save_stats(norm_envs)

                checkpoint_fname = osp.join(checkpoint_dir, '{:05}'.format(update))
                joblib.dump(policy, checkpoint_fname)
                if best_mean_reward is None or mean_reward > best_mean_reward:
                    best_checkpoint = checkpoint_fname
                    blogger.log("Updating model, mean reward {} -> {}".format(
                                best_mean_reward, mean_reward))
                    best_mean_reward = mean_reward

    return joblib.load(best_checkpoint)


def sample(env_fns, policy, num_episodes, seed, tf_config, parallel=True, const_norm=False):
    smodel, snorm_env = policy
    num_envs = len(env_fns)
    make_envs = SubprocVecEnv if (parallel and num_envs > 1) else DummyVecEnv
    envs = make_envs(env_fns)
    envs_monitor = SampleVecMonitor(envs)

    infer_graph = tf.Graph()
    with infer_graph.as_default():
        # Seed to make results reproducible
        seed = seeding.create_seed(seed)
        tf.set_random_seed(seed)  # TODO: is this seed really needed (can set globally)?
        with tf.Session(config=make_config(tf_config)):
            # Load model
            make_model_pkl, params = smodel
            make_model = cloudpickle.loads(make_model_pkl)
            model = make_model(num_envs)
            model.restore_params(params)

            # Initialize environment
            norm_envs = _restore_stats(snorm_env, envs_monitor)
            if const_norm:
                norm_envs = _make_const(norm_envs)

            obs = norm_envs.reset()
            states = model.initial_state
            dones = np.zeros(len(env_fns), dtype='bool')
            completed = 0
            while completed < num_episodes:
                a, v, states, neglogp = model.step(obs, states, dones)
                obs, r, dones, info = norm_envs.step(a)
                completed += np.sum(dones)

            envs.close()
            return envs_monitor.trajectories