'''Based on OpenAI Baselines PPO2 (GPU optimized).'''

from collections import deque
import logging
import joblib
import math
import os
import os.path as osp
import pickle
import time

import cloudpickle
from gym.utils import seeding
import numpy as np
import tensorflow as tf

from baselines import logger as blogger
from baselines.common import explained_variance
from baselines.common.runners import AbstractEnvRunner
from baselines.common.vec_env import VecEnvWrapper
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.ppo2.policies import MlpPolicy

from pirl.agents.sample import SampleVecMonitor
from pirl.utils import set_cuda_visible_devices

logger = logging.getLogger('pirl.agents.ppo')

# Lightly adapted from baselines/ppo2/ppo2.py
# Main changes: in Model, add {get,restore}_params.
# In Runner:
# * Report original and reward after normalization
# * Yield intermediate parameters during optimization

class Model(object):
    def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                nsteps, ent_coef, vf_coef, max_grad_norm):
        sess = tf.get_default_session()

        act_model = policy(sess, ob_space, ac_space, nbatch_act, 1, reuse=False)
        train_model = policy(sess, ob_space, ac_space, nbatch_train, nsteps, reuse=True)

        A = train_model.pdtype.sample_placeholder([None])
        ADV = tf.placeholder(tf.float32, [None])
        R = tf.placeholder(tf.float32, [None])
        OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        OLDVPRED = tf.placeholder(tf.float32, [None])
        LR = tf.placeholder(tf.float32, [])
        CLIPRANGE = tf.placeholder(tf.float32, [])

        neglogpac = train_model.pd.neglogp(A)
        entropy = tf.reduce_mean(train_model.pd.entropy())

        vpred = train_model.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        vf_losses1 = tf.square(vpred - R)
        vf_losses2 = tf.square(vpredclipped - R)
        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
        pg_losses = -ADV * ratio
        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef
        with tf.variable_scope('model'):
            params = tf.trainable_variables()
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        _train = trainer.apply_gradients(grads)

        def train(lr, cliprange, obs, returns, masks, actions, values, neglogpacs, states=None):
            advs = returns - values
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            td_map = {train_model.X:obs, A:actions, ADV:advs, R:returns, LR:lr,
                    CLIPRANGE:cliprange, OLDNEGLOGPAC:neglogpacs, OLDVPRED:values}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            return sess.run(
                [pg_loss, vf_loss, entropy, approxkl, clipfrac, _train],
                td_map
            )[:-1]
        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']

        def get_params():
            return sess.run(params)

        def save(save_path):
            ps = get_params()
            joblib.dump(ps, save_path)

        def restore_params(loaded_params):
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            sess.run(restores)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restore_params(loaded_params)
            # If you want to load weights, also save/load observation scaling inside VecNormalize

        self.train = train
        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value
        self.initial_state = act_model.initial_state
        self.get_params = get_params
        self.save = save
        self.restore_params = restore_params
        self.load = load
        tf.global_variables_initializer().run(session=sess) #pylint: disable=E1101


class Runner(AbstractEnvRunner):
    def __init__(self, *, env, model, nsteps, gamma, lam):
        super().__init__(env=env, model=model, nsteps=nsteps)
        self.lam = lam
        self.gamma = gamma

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        mb_states = self.states
        epinfos = []
        for _ in range(self.nsteps):
            actions, values, self.states, neglogpacs = self.model.step(self.obs, self.states, self.dones)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, self.states, self.dones)
        #discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values
        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
            mb_states, epinfos)
# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

def constfn(val):
    def f(_):
        return val
    return f

def learn(*, policy, env, epinfobuf, trajectorybuf,
             nsteps, total_timesteps, ent_coef, lr,
             vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,
             log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
             save_interval=0, load_path=None):

    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)
    total_timesteps = int(total_timesteps)

    # WARNING: Make sure to extract this info before wrapping env in
    # SampleVecMonitor. Otherwise, we will have to pickle SampleVecMonitor.
    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches

    make_model = lambda nenvs: Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                    nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm)
    model = make_model(nenvs)
    if load_path is not None:
        model.load(load_path)
    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)

    tfirststart = time.time()

    nupdates = total_timesteps//nbatch
    for update in range(1, nupdates+1):
        assert nbatch % nminibatches == 0
        nbatch_train = nbatch // nminibatches
        tstart = time.time()
        frac = 1.0 - (update - 1.0) / nupdates
        lrnow = lr(frac)
        cliprangenow = cliprange(frac)
        obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run() #pylint: disable=E0632
        epinfobuf.extend(epinfos)
        mblossvals = []
        if states is None: # nonrecurrent version
            inds = np.arange(nbatch)
            for _ in range(noptepochs):
                np.random.shuffle(inds)
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices))
        else: # recurrent version
            assert nenvs % nminibatches == 0
            envsperbatch = nenvs // nminibatches
            envinds = np.arange(nenvs)
            flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
            envsperbatch = nbatch_train // nsteps
            for _ in range(noptepochs):
                np.random.shuffle(envinds)
                for start in range(0, nenvs, envsperbatch):
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    mbflatinds = flatinds[mbenvinds].ravel()
                    slices = (arr[mbflatinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mbstates = states[mbenvinds]
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices, mbstates))

        lossvals = np.mean(mblossvals, axis=0)
        tnow = time.time()
        fps = int(nbatch / (tnow - tstart))
        mean_reward = safemean([sum(trajectory[2]) for trajectory in trajectorybuf])
        if update % log_interval == 0 or update == 1:
            ev = explained_variance(values, returns)
            blogger.logkv('serial_timesteps', update*nsteps)
            blogger.logkv('nupdates', update)
            blogger.logkv('total_timesteps', update*nbatch)
            blogger.logkv('fps', fps)
            blogger.logkv('explained_variance', float(ev))
            # orig_eprewmean comes from bench.Monitor, immediately after
            # environment is created. eprewmean comes from the reward PPO sees.
            # These may differ when e.g. normalization is applied, or if we are
            # reoptimizing a learnt reward function using a reward wrapper.
            blogger.logkv('eprewmean', mean_reward)
            blogger.logkv('eprewmean_orig', safemean([epinfo['r'] for epinfo in epinfobuf]))
            blogger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
            blogger.logkv('time_elapsed', tnow - tfirststart)
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                blogger.logkv(lossname, lossval)
            blogger.dumpkvs()
        if save_interval and (update % save_interval == 0 or update == 1) and blogger.get_dir():
            yield update, mean_reward, make_model, model.get_params()


def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)

# My code

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


def train_continuous(venv, discount, seed, log_dir, tf_config,
                     num_timesteps, norm=True):
    '''Policy with hyperparameters optimized for continuous control environments
       (e.g. MuJoCo). Returns log_dir, where the trained policy is saved.'''
    blogger.configure(dir=log_dir)
    checkpoint_dir = osp.join(blogger.get_dir(), 'checkpoints')
    os.makedirs(checkpoint_dir)

    # PPO relies on two monitors to report episode reward:
    # - bench.Monitor, that is applied as soon as the environment created,
    #   for the original reward. This is used in reporting only.
    # - SampleVecMonitor, applied here immediately before (optional) normalization.
    #   In most cases, the reward here is the same as for bench.Monitor.
    #   But they will be different if we are applying a reward wrapper (when
    #   reoptimizing from IRL). This is the reward used to choose the best model.
    epinfobuf = deque(maxlen=100)
    trajectorybuf = deque(maxlen=100)
    venv = SampleVecMonitor(venv, trajectorybuf)
    make_vec_normalize = VecNormalize if norm else DummyVecNormalize
    norm_venv = make_vec_normalize(venv)

    train_graph = tf.Graph()
    with train_graph.as_default():
        tf.set_random_seed(seed)
        with tf.Session(config=make_config(tf_config)):
            policy = MlpPolicy
            nsteps = 2048 // venv.num_envs
            learner = learn(
                policy=policy, env=norm_venv,
                epinfobuf=epinfobuf, trajectorybuf=trajectorybuf,
                nsteps=nsteps, nminibatches=32,
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
                policy = model, _save_stats(norm_venv)

                checkpoint_fname = osp.join(checkpoint_dir, '{:05}'.format(update))
                joblib.dump(policy, checkpoint_fname)

                valid = math.isfinite(mean_reward)
                improvement = (best_mean_reward is None
                               or mean_reward > best_mean_reward)
                if valid and improvement:
                    best_checkpoint = checkpoint_fname
                    blogger.log("Updating model, mean reward {} -> {}".format(
                                best_mean_reward, mean_reward))
                    best_mean_reward = mean_reward

    return joblib.load(best_checkpoint)


#TODO: remove None defaults (workaround Ray issue #998)
def sample(envs=None, policy=None, num_episodes=None,
           seed=None, tf_config=None, const_norm=False):
    smodel, snorm_env = policy
    envs_monitor = SampleVecMonitor(envs)

    infer_graph = tf.Graph()
    with infer_graph.as_default():
        # Seed to make results reproducible
        tf.set_random_seed(seed)
        with tf.Session(config=make_config(tf_config)):
            # Load model
            make_model_pkl, params = smodel
            make_model = cloudpickle.loads(make_model_pkl)
            model = make_model(envs.num_envs)
            model.restore_params(params)

            # Initialize environment
            norm_envs = _restore_stats(snorm_env, envs_monitor)
            if const_norm:
                norm_envs = _make_const(norm_envs)

            obs = norm_envs.reset()
            states = model.initial_state
            dones = np.zeros(envs.num_envs, dtype='bool')
            completed = 0
            while completed < num_episodes:
                a, v, states, neglogp = model.step(obs, states, dones)
                obs, r, dones, info = norm_envs.step(a)
                completed += np.sum(dones)

            envs.close()
            return envs_monitor.trajectories