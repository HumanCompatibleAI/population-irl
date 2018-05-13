import argparse
import gym
from gym.wrappers import Monitor
import joblib

from pirl.agents import ppo
from pirl.experiments import config

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
    parser.add_argument('checkpoint', type=str, help='path to PPO checkpoint')
    parser.add_argument('env', type=str, help='name of Gym environment.')
    parser.add_argument('--out-dir', dest='out_dir', metavar='DIR', type=str, default='/tmp/play-ppo')
    parser.add_argument('--num-episodes', dest='num_episodes', metavar='N', type=int, default=1)
    parser.add_argument('--seed', dest='seed', metavar='N', type=int, default=0)
    args = parser.parse_args()

    print('Loading checkpoint from ', args.checkpoint)
    policy = joblib.load(args.checkpoint)

    def make_env():
        env = gym.make(args.env)
        env.seed(args.seed)
        env = Monitor(env, directory=args.out_dir,
                          force=True, video_callable=lambda x: True)
        env = InteractiveMonitor(env)
        return env
    print('Sampling {} in {}, saving videos to {}'.format(
           args.num_episodes, args.env, args.out_dir))
    ppo.sample([make_env], policy, args.num_episodes,
               args.seed, tf_config=config.TENSORFLOW)

if __name__ == '__main__':
    main()