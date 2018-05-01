import itertools
import logging
import math
import os
import os.path as osp

import gym
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import pandas as pd
import numpy as np
import seaborn as sns

logger = logging.getLogger('pirl.experiments.plots')

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
    kwargs.setdefault('cmap', 'YlGnBu')
    sns.heatmap(reward, mask=walls, **kwargs)


def _gridworld_heatmaps(reward, shape, env_name, get_axis,
                        prefix=None, share_scale=True, **kwargs):
    env = gym.make(env_name)
    try:
        walls = env.unwrapped.walls
    except AttributeError:
        walls = None

    gt = env.unwrapped.reward
    ax = get_axis('gt')
    _gridworld_heatmap(gt, shape, walls, ax=ax, **kwargs)
    ax.set_title('Ground Truth')
    yield ax

    vmin = None
    vmax = None
    if share_scale:
        vmin = min([v.min() for v in reward.values()])
        vmax = min([v.min() for v in reward.values()])

    i = 0
    for n, reward_by_m in reward.items():
        for m, r in reward_by_m.items():
            r = r[env_name]
            r = r - np.mean(r) + np.mean(gt)
            ax = get_axis(i)
            _gridworld_heatmap(r, shape, vmin=vmin, vmax=vmax, ax=ax)
            title = '{}/{}'.format(m, n)
            if prefix is not None:
                title = '{} ({})'.format(prefix, title)
            ax.set_title(title)
            yield ax
            i += 1


def gridworld_heatmap(reward, shape, num_cols=3, figsize=(11.6, 8.6),
                      prefix=None, share_scale=False):
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
        axs = list(itertools.chain(*axs))  # flatten
        def get_ax(n):
            if n == 'gt':
                return axs[0]
            else:
                return axs[n + 1]
        fig.suptitle(env_name)
        it = _gridworld_heatmaps(reward, shape, env_name, get_ax,
                                 prefix=prefix, share_scale=share_scale)
        list(it)

        yield env_name, fig
        plt.close()


def gridworld_heatmap_movie(out_dir, reward, shape,
                            prefix=None, share_scale=False, fps=1, dpi=300):
    envs = list(list(reward.values())[0].values())[0].keys()
    get_ax = lambda n: fig.gca() 
    os.makedirs(out_dir, exist_ok=True)
    for env_name in envs:
        logger.debug('Generating movie for %s', env_name)
        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title='Reward Heatmap', artist='matplotlib')
        writer = FFMpegWriter(fps=fps, metadata=metadata)
    
        fig = plt.figure()
        fname = osp.join(out_dir, env_name.replace('/', '_') + '.mp4')
        with writer.saving(fig, fname, dpi):
            it = _gridworld_heatmaps(reward, shape, env_name, get_ax,
                                     prefix=prefix, share_scale=share_scale)
            for i, _v in enumerate(it):
                writer.grab_frame()
                fig.clf()
                logger.debug('%s: written frame %d', fname, i)
        plt.close(fig)


def save_figs(figs, prefix):
    os.makedirs(prefix, exist_ok=True)
    for name, fig in figs:
        path = os.path.join(prefix, name.replace('/', '_') + '.pdf')
        fig.savefig(path)
