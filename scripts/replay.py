import argparse
import functools
import joblib
import pickle

import gym
from gym.wrappers import Monitor
import tensorflow as tf

from pirl import config, experiments
from pirl.irl import airl


class InteractiveMonitor(gym.Wrapper):
    def __init__(self, env):
        super(InteractiveMonitor, self).__init__(env)
        self.episode = 0
        self.reward = 0

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.env.render()
        self.reward += reward
        print('R = {:.5f}'.format(reward))
        if done:
            print('DONE', self.episode, self.reward)
            self.episode += 1
            self.reward = 0
        return observation, reward, done, info

    def reset(self):
        print('START ', self.episode)
        return self.env.reset()


def airl_checkpoint(envs, fname, num_episodes, seed):
    tf_cfg = tf.ConfigProto(device_count={'GPU': 0})
    with tf.Session(config=tf_cfg) as sess:
        print('Loading checkpoint from ', fname)
        checkpoint = joblib.load(fname)  # depickling needs a default session
        policy = checkpoint['policy']
        policy_pkl = pickle.dumps(policy)
    airl.sample(envs, policy_pkl, num_episodes, seed, tf_cfg=config.TENSORFLOW)


def sample_decorator(f):
    @functools.wraps(f)
    def wrapper(envs, fname, num_episodes, seed):
        print('Loading policy from ', fname)
        policy = joblib.load(fname)
        return f(envs, policy, num_episodes, seed)
    return wrapper


ALGO_SEARCH = {
    'rl': config.RL_ALGORITHMS,
    'sirl': config.SINGLE_IRL_ALGORITHMS,
    'pirl': config.POPULATION_IRL_ALGORITHMS,
}
CUSTOM_SEARCH = {'airl_checkpoint': (airl_checkpoint, True)}

def find_algo(algo_name):
    for k, d in ALGO_SEARCH.items():
        if algo_name in d:
            print('Found {} in {}'.format(algo_name, k))
            algo = d[algo_name]
            return sample_decorator(algo.sample), algo.vectorized
    return CUSTOM_SEARCH[algo_name]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('algo', type=find_algo, help='algorithm name')
    parser.add_argument('policy', type=str, help='path to policy checkpoint')
    parser.add_argument('env', type=str, help='name of Gym environment.')
    parser.add_argument('--out-dir', dest='out_dir', metavar='DIR', type=str,
                        default='/tmp/play-ppo')
    parser.add_argument('--num-episodes', dest='num_episodes', metavar='N',
                        type=int, default=1)
    parser.add_argument('--seed', dest='seed', metavar='N', type=int, default=0)
    args = parser.parse_args()

    print('Sampling {} in {}, saving videos to {}'.format(
           args.num_episodes, args.env, args.out_dir))
    def wrapper(env):
        env = Monitor(env, directory=args.out_dir,
                      force=True, video_callable=lambda x: True)
        return InteractiveMonitor(env)

    sample, vectorized = args.algo
    with experiments.make_envs(args.env, vectorized, parallel=1,
                               base_seed=args.seed, pre_wrapper=wrapper,
                               log_prefix=args.out_dir) as envs:
        sample(envs, args.policy, args.num_episodes, args.seed)


if __name__ == '__main__':
    main()