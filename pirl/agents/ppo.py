'''Wrapper around OpenAI Baselines PPO2 (GPU optimized).
   Based on baselines/ppo2/run_mujoco.py.'''

import os.path as osp

import cloudpickle
from gym.utils import seeding
import tensorflow as tf
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


def train_continuous(env, discount, log_dir, tf_config, num_timesteps):
    '''Policy with hyperparameters optimized for continuous control environments
       (e.g. MuJoCo). Returns log_dir, where the trained policy is saved.'''
    #TODO: should we set a seed from within here, or is it ok to rely on caller?

    blog_dir = osp.join(log_dir, 'ppo')
    blogger.configure(dir=blog_dir)
    #TODO: Do I want to use this for all algorithms?
    env = bench.Monitor(env, blogger.get_dir())
    train_graph = tf.Graph()
    with train_graph.as_default():
        with tf.Session(config=make_config(tf_config)):
            env = DummyVecEnv([lambda: env])
            env = VecNormalize(env)

            policy = MlpPolicy
            mean_reward, make_model, params = ppo2.learn(
                policy=policy, env=env, nsteps=2048, nminibatches=32,
                lam=0.95, gamma=discount, noptepochs=10, log_interval=1,
                ent_coef=0.0,
                lr=3e-4,
                cliprange=0.2,
                total_timesteps=num_timesteps,
                save_interval=4)

    make_model_pkl = cloudpickle.dumps(make_model)  # joblib cannot pickle lambdas
    return make_model_pkl, params


def sample(env, policy, num_episodes, seed, tf_config):
    infer_graph = tf.Graph()
    with infer_graph.as_default():
        with tf.Session(config=make_config(tf_config)):
            # Seed to make results reproducible
            seed = seeding.create_seed(seed)
            env.seed(seed)
            tf.set_random_seed(seed)

            # Load model and initialize environment
            make_model_pkl, params = policy
            make_model = cloudpickle.loads(make_model_pkl)
            model = make_model()
            model.restore_params(params)
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