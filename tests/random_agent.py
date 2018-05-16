import argparse
import time

import gym
import gym.wrappers as wrappers
import gym.logger as logger

from pirl import envs  # needed for side effect of registering environments

# Loosely adapted from example in OpenAI Gym

class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--video-every', dest='video_freq', metavar='N', type=int, default=0)
    parser.add_argument('--render', dest='fps', metavar='FPS', type=int, default=0)
    parser.add_argument('--num-episodes', dest='num_episodes', metavar='N', type=int, default=100)
    parser.add_argument('--out-dir', dest='out_dir', metavar='DIR', type=str, default='/tmp/random-agent-results')
    parser.add_argument('env_id', nargs='?', default='CartPole-v0', help='Select the environment to run')
    args = parser.parse_args()

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.WARN)

    env = gym.make(args.env_id)

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    env = wrappers.Monitor(env, directory=args.out_dir, force=True,
                           video_callable=lambda x: args.video_freq > 0 and x % args.video_freq == 0)
    env.seed(1)
    agent = RandomAgent(env.action_space)

    reward = 0
    last_render = time.time()
    for i in range(args.num_episodes):
        print('*** Episode {} ***'.format(i))
        ob = env.reset()
        done = False
        while not done:
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            if args.fps > 0:
                now = time.time()
                sleep_for = (last_render + 1 / args.fps) - now
                if sleep_for > 0:
                    time.sleep(sleep_for)
                env.render()
                last_render = now

    # Close the env and write monitor result info to disk
    env.close()
