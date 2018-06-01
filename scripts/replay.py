import argparse
import gym
from gym.wrappers import Monitor
import joblib

from pirl import config, experiments


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('algo', type=str, help='algorithm name')
    parser.add_argument('policy', type=str, help='path to policy checkpoint')
    parser.add_argument('env', type=str, help='name of Gym environment.')
    parser.add_argument('--out-dir', dest='out_dir', metavar='DIR', type=str, default='/tmp/play-ppo')
    parser.add_argument('--num-episodes', dest='num_episodes', metavar='N', type=int, default=1)
    parser.add_argument('--seed', dest='seed', metavar='N', type=int, default=0)
    args = parser.parse_args()

    algo_search = {'rl': config.RL_ALGORITHMS,
                   'sirl': config.SINGLE_IRL_ALGORITHMS,
                   'pirl': config.POPULATION_IRL_ALGORITHMS}
    for k, d in algo_search.items():
        if args.algo in d:
            print('Found {} in {}'.format(args.algo, k))
            algo = d[args.algo]

    print('Loading checkpoint from ', args.policy)
    policy = joblib.load(args.policy)

    print('Sampling {} in {}, saving videos to {}'.format(
           args.num_episodes, args.env, args.out_dir))
    def wrapper(env):
        env = Monitor(env, directory=args.out_dir,
                      force=True, video_callable=lambda x: True)
        return InteractiveMonitor(env)

    with experiments.make_envs(args.env, algo.vectorized, parallel=1,
                               base_seed=args.seed, pre_wrapper=wrapper,
                               log_prefix=args.out_dir) as envs:
        algo.sample(envs, policy, args.num_episodes, args.seed)


if __name__ == '__main__':
    main()