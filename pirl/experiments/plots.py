import collections
import itertools
import math
import os

import gym
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))
plt.style.use(os.path.join(THIS_DIR, 'default.mplstyle'))

def extract_value(data):
    value = data['value']
    ground_truth = data['ground_truth']

    def extract(idx):
        res = collections.OrderedDict()
        for irl_name, value_by_irl in value.items():
            d = collections.OrderedDict()
            for n, value_by_n in value_by_irl.items():
                for m, value_by_m in value_by_n.items():
                    k = '{}/{}'.format(m, n)
                    d[k + '_mu'] = pd.Series(collections.OrderedDict(
                        [(env, value_by_env[idx][0])
                         for env, value_by_env in value_by_m.items()]))
                    d[k + '_se'] = pd.Series(collections.OrderedDict(
                        [(env, value_by_env[idx][1])
                         for env, value_by_env in value_by_m.items()]))
            df = pd.DataFrame(d)
            res[irl_name] = df
        res['ground_truth'] = pd.DataFrame(ground_truth, index=df.columns).T

        return pd.Panel(res)

    res = {'optimal': extract(0), 'planner': extract(1)}

    return res


def _gridworld_heatmap(reward, shape, walls=None, **kwargs):
    reward = reward.reshape(shape)
    kwargs.setdefault('fmt', '.0f')
    kwargs.setdefault('annot', True)
    kwargs.setdefault('annot_kws', {'fontsize': 'smaller'})
    sns.heatmap(reward, mask=walls, **kwargs)


def gridworld_heatmap(reward, shape, num_cols=3, figsize=(11.6, 8.6)):
    envs = list(list(reward.values())[0].values())[0].keys()
    num_plots = sum([len(d) for d in reward.values()]) + 1
    num_rows = math.ceil(num_plots / num_cols)
    for env_name in envs:
        fig, axs = plt.subplots(num_rows,
                                num_cols,
                                squeeze=False,
                                figsize=figsize,
                                sharex=True,
                                sharey=True)
        fig.suptitle(env_name)
        axs = list(itertools.chain(*axs))  # flatten
        i = 0

        # Ground truth reward
        env = gym.make(env_name)
        gt = env.unwrapped.reward
        try:
            walls = env.unwrapped.walls
        except AttributeError:
            walls = None
        _gridworld_heatmap(gt, shape, walls, ax=axs[i])
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