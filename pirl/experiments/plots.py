import itertools
import math
import os

import gym
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def _gridworld_heatmap(reward, shape, ax):
    reward = reward.reshape(shape)
    sns.heatmap(reward, annot=True, fmt='.0f', ax=ax)

def gridworld_heatmap(reward, shape, num_cols=3):
    num_general_trajectories = reward.keys()
    envs = list(list(reward.values())[0].values())[0].keys()
    num_plots = sum([len(d) for d in reward.values()])
    num_rows = math.ceil(num_plots / num_cols)
    for env_name in envs:
        fig, axs = plt.subplots(num_rows, num_cols, squeeze=False)
        fig.suptitle(env_name)
        axs = list(itertools.chain(*axs))  # flatten
        i = 0

        # Ground truth reward
        env = gym.make(env_name)
        gt = env.unwrapped.reward
        _gridworld_heatmap(gt, shape, ax=axs[i])
        axs[i].set_title('Ground Truth')

        for n, reward_by_m in reward.items():
            for m, r in reward_by_m.items():
                r = r[env_name]
                i += 1
                r = r - np.mean(r) + np.mean(gt)
                _gridworld_heatmap(r, shape, ax=axs[i])
                axs[i].set_title('{}/{}'.format(m, n))

        yield env_name, fig
        plt.close()

def save_figs(figs, prefix):
    os.makedirs(prefix, exist_ok=True)
    for name, fig in figs:
        path = os.path.join(prefix, name.replace('/', '_') + '.pdf')
        fig.savefig(path)