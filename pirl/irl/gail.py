'''Interface for GAIL from OpenAI Baselines.'''

import os.path as osp

import numpy as np
import tensorflow as tf

from baselines.gail import mlp_policy
from baselines.gail.dataset.mujoco_dset import Dset
from baselines.gail import behavior_clone, trpo_mpi
from baselines.gail.adversary import TransitionClassifier

from pirl.agents.sample import SampleMonitor

def _make_dset(trajectories, randomize=True):
    '''Return a Dset object containing observations and actions extracted
       from trajectories. GAIL does not care about episode bounds, so
       we concatenate together all trajectories, and optionally randomly
       sample state-action pairs.'''
    obs = np.concatenate([x[0] for x in trajectories])
    acs = np.concatenate([x[1] for x in trajectories])
    return Dset(obs, acs, randomize)

def _policy_factory(policy_cfg):
    policy_kwargs = {
        'hid_size': 100,
        'num_hid_layers': 2,
    }
    if policy_cfg is not None:
        policy_kwargs.update(policy_cfg)
    def policy_fn(name, ob_space, ac_space, reuse=False):
        return mlp_policy.MlpPolicy(name=name, reuse=reuse,
                                    ob_space=ob_space, ac_space=ac_space,
                                    **policy_kwargs)
    return policy_fn

def irl(env, trajectories, discount, seed, log_dir, *,
        tf_cfg, policy_cfg=None, gan_cfg=None, train_cfg=None):
    #TODO: discount seems to be unused?
    dataset = _make_dset(trajectories)

    train_graph = tf.Graph()
    with train_graph.as_default():
        tf.set_random_seed(seed)

        policy_fn = _policy_factory(policy_cfg)

        gan_kwargs = {'hidden_size': 32}
        if gan_cfg is not None:
            gan_kwargs.update(gan_cfg)
        reward_giver = TransitionClassifier(env, **gan_kwargs)

        train_kwargs = {
            'pretrained': False,
            'BC_max_iter': 10000,
            'g_step': 3, # number of steps to train policy in each epoch
            'd_step': 1, # number of steps to train discriminator in each epoch
            'entcoeff': 0, # entropy coefficiency of policy
            'max_timesteps': 5e6, # number of timesteps per episode
            'timesteps_per_batch': 1024,
            'max_kl': 0.01,
            'cg_iters': 10,
            'cg_damping': 0.1,
            'lam': 0.97,
            'vf_iters': 5,
            'vf_stepsize': 1e-3,
        }
        if train_cfg is not None:
            train_kwargs.update(train_cfg)

        pretrained_weight = None
        bc_max_iter = train_kwargs.pop('BC_max_iter')
        if train_kwargs['pretrained']:
            # Pretrain with behavior cloning
            pretrained_weight = behavior_clone.learn(env, policy_fn, dataset,
                                         max_iters=bc_max_iter)
        # TODO: use MPI?
        ckpt_dir = osp.join(log_dir, 'checkpoints')

        with tf.Session(config=tf_cfg) as sess:
            trpo_mpi.learn(env, policy_fn, reward_giver, dataset, rank=0,
                           pretrained_weight=pretrained_weight,
                           ckpt_dir=ckpt_dir, log_dir=log_dir,
                           gamma=discount, save_per_iter=100,
                           task_name='gail', **train_kwargs)

            policy_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'pi')
            policy_serialised = sess.run(policy_vars)

    return None, policy_serialised

def sample(env, policy_saved, num_episodes, seed, *, tf_cfg, policy_cfg=None):
    env = SampleMonitor(env)

    infer_graph = tf.Graph()
    with infer_graph.as_default():
        policy_fn = _policy_factory(policy_cfg)
        policy = policy_fn('pi', env.observation_space, env.action_space,
                           reuse=False)

        with tf.Session(config=tf_cfg) as sess:
            # Deserialize policy
            policy_vars = policy.get_variables()
            restores = []
            for p, loaded_p in zip(policy_vars, policy_saved):
                restores.append(p.assign(loaded_p))
            sess.run(restores)

            # Policy rollout
            completed = 0
            ob = env.reset()
            while completed < num_episodes:
                # First argument to act determines if stochastic (sample)
                # or deterministic (mode of policy)
                a, vpred = policy.act(True, ob)
                ob, _r, done, _info = env.step(a)
                if done:
                    completed += 1
                    ob = env.reset()

            return env.trajectories