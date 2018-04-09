'''Wrapper around OpenAI Baselines PPO2 (GPU optimized).
   Based on baselines/ppo2/run_mujoco.py.'''

import os
import os.path as osp

import cloudpickle
from gym.utils import seeding
import tensorflow as tf
import numpy as np
import logging

from baselines.ppo2 import ppo2
from baselines import bench, logger as blogger
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.ppo2.policies import MlpPolicy

logger = logging.getLogger('pirl.agents.ppo')

def make_config(tf_config):
    ncpu = 1
    config = tf.ConfigProto()
    config.CopyFrom(tf_config)
    config.allow_soft_placement=True
    config.intra_op_parallelism_threads = ncpu
    config.inter_op_parallelism_threads = ncpu
    return config


def load_model(log_dir):
    '''Load model from checkpoint. (Yuck!)'''
    path = osp.join(log_dir, 'make_model.pkl')
    with open(path, 'rb') as fh:
        make_model = cloudpickle.load(fh)
    model = make_model()
    checkpoint_dir = osp.join(log_dir, 'checkpoints')
    checkpoint_files = os.listdir(checkpoint_dir)
    # filename starts with strictly increasing 5-digit update ID
    latest_checkpoint = max(checkpoint_files)
    checkpoint_path = osp.join(checkpoint_dir, latest_checkpoint)
    logger.debug('Loading model from %s', checkpoint_path)
    model.load(checkpoint_path)

    return model


def train_continuous(env, discount, tf_config, log_dir, num_timesteps):
    '''Policy with hyperparameters optimized for continuous control environments
       (e.g. MuJoCo). Returns log_dir, where the trained policy is saved.'''
    #TODO: should we set a seed from within here, or is it ok to rely on caller?

    blogger.configure(dir=log_dir)
    #TODO: Do I want to use this for all algorithms?
    env = bench.Monitor(env, blogger.get_dir())
    with tf.Session(config=make_config(tf_config)):
        env = DummyVecEnv([lambda: env])
        env = VecNormalize(env)

        policy = MlpPolicy
        ppo2.learn(policy=policy, env=env, nsteps=2048, nminibatches=32,
            lam=0.95, gamma=discount, noptepochs=10, log_interval=1,
            ent_coef=0.0,
            lr=3e-4,
            cliprange=0.2,
            total_timesteps=num_timesteps,
            save_interval=4)

    return log_dir


def value(env, log_dir, discount, tf_config, num_episodes=10, seed=0):
    '''Test policy saved in log_dir on num_episodes in env.
        Return average reward.'''
    # TODO: does this belong in PPO or a more general class?
    trajectories = sample(env, log_dir, tf_config, num_episodes, seed)
    rewards = [r for (s, a, r) in trajectories]
    horizon = max([len(s) for (s, a, r) in trajectories])
    weights = np.cumprod([1] + [discount] * (horizon - 1))
    total_reward = [np.dot(r, weights[:len(r)]) for r in rewards]

    mean = np.mean(total_reward)
    se = np.std(total_reward, ddof=1) / np.sqrt(num_episodes)
    return mean, se


def sample(env, log_dir, tf_config, num_episodes, seed):
    with tf.Session(config=make_config(tf_config)):
        # Seed to make results reproducible
        seed = seeding.create_seed(seed)
        env.seed(seed)
        tf.set_random_seed(seed)

        # Load model and initialize environment
        model = load_model(log_dir)
        env = DummyVecEnv([lambda: env])

        def helper():
            '''Samples from environment for an entire episode.'''
            # TODO: do we need to reset model between episodes?
            # (Think not, no memory?)
            observations = []
            actions = []
            rewards = []

            obs = env.reset()
            done = False
            while not done:
                observations.append(obs[0])
                #TODO: this won't work for LSTMs etc, make more general
                a, v, sprev, neglogp = model.step(obs)
                actions.append(a[0])
                obs, r, done, info = env.step(a)
                rewards.append(r[0])

            return observations, actions, rewards

        return [helper() for i in range(num_episodes)]