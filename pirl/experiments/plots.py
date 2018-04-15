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

def nested_dicts_to_df(ds, idxs, transform):
    if len(idxs) == 2:
        ds = transform(ds)
        df = pd.DataFrame(ds)
        df.columns.name = idxs[0]
        df.index.name = idxs[1]
    else:
        ds = {k: nested_dicts_to_df(v, idxs[1:], transform)
              for k, v in ds.items()}
        ds = {k: v.stack() for k, v in ds.items()}
        df = pd.DataFrame(ds)
        df.columns.name = idxs[0]
    return df

def extract_value(data):
    def unpack_mean_sd_tuple(d):
        return {k: {'mean': v[0], 'se': v[1]} for k, v in d.items()}
    idxs = ['irl', 'n', 'm', 'env', 'eval', 'type']
    values = nested_dicts_to_df(data['values'], idxs, unpack_mean_sd_tuple)
    sorted_idx = ['env', 'n', 'm', 'eval', 'type']
    values = values.reorder_levels(sorted_idx)

    ground_truth = pd.DataFrame(unpack_mean_sd_tuple(data['ground_truth']))
    def get_gt(k):
        env, _, _, _, kind = k
        return ground_truth.loc[kind, env]
    values['expert'] = list(map(get_gt, values.index))

    return values


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