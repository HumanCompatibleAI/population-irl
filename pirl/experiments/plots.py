import itertools
import logging
import math
import os
import os.path as osp

import gym
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as manimation
import pandas as pd
import numpy as np
import scipy.stats
import seaborn as sns

logger = logging.getLogger('pirl.experiments.plots')

THIS_DIR = osp.join(os.path.dirname(os.path.realpath(__file__)))

def style(name):
    return osp.join(THIS_DIR, '{}.mplstyle'.format(name))

plt.style.use(style('default'))

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

    idx =  ['env', 'eval', 'type']
    ground_truth = nested_dicts_to_df(data['ground_truth'], idx, unpack_mean_sd_tuple)
    ground_truth = ground_truth.stack().unstack('eval')
    ground_truth = ground_truth.reorder_levels(['env', 'type'])

    def get_gt(k):
        env, _, _, _, kind = k
        return ground_truth.loc[(env, kind), :]
    values_gt = pd.DataFrame(list(map(get_gt, values.index)), index=values.index)
    return pd.concat([values, values_gt], axis=1)


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


def gridworld_ground_truth(envs, shape):
    data = {}
    rmin = 1e10
    rmax = -1e10
    for nickname, env_name in envs.items():
        env = gym.make(env_name)
        reward = env.unwrapped.reward
        walls = env.unwrapped.walls
        env.close()

        data[nickname] = (reward, walls)
        rmin = min(rmin, np.min(reward))
        rmax = max(rmax, np.max(reward))

    num_envs = len(envs)
    width, height = mpl.rcParams['figure.figsize']
    height = width / num_envs
    figsize = (width, height)

    width_ratios = [10] * num_envs + [1]
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, num_envs + 1, width_ratios=width_ratios, wspace=0.2)

    first_ax = plt.subplot(gs[0])
    for i, (nickname, (reward, walls)) in enumerate(data.items()):
        if i == 0:
            ax = first_ax
        else:
            ax = plt.subplot(gs[i], sharex=first_ax, sharey=first_ax)
            plt.setp(ax.get_yticklabels(), visible=False)
        if i == num_envs - 1:
            kwargs = {'cbar': True, 'cbar_ax': plt.subplot(gs[num_envs])}
        else:
            kwargs = {'cbar': False}
        _gridworld_heatmap(reward, shape, walls, vmin=rmin, vmax=rmax,
                           ax=ax, **kwargs)
        ax.set_title(nickname)

    return fig


def value_bar_chart(values, alpha=0.05, relative=None,
                    error=False, ax=None, **kwargs):
    '''Takes DataFrame with columns corresponding to algorithms, and
       a MultiIndex with levels [n, type] where n is the number of trajectories
       and type is either 'mean' or 'se'. It outputs a stacked bar graph.'''
    if ax is None:
        ax = plt.gca()
    y = values.xs('mean', level='type')
    se = values.xs('se', level='type')
    z = scipy.stats.norm.ppf(1 - (alpha / 2))
    err = se * z

    if relative is not None:
        if error:
            y = -y.sub(y[relative], 0)
        else:
            rel = y[relative]

        y = y.drop(relative, axis=1)
    y.plot.bar(yerr=err, ax=ax, **kwargs)

    ax.set_xlabel('Trajectories')
    ax.set_ylabel('Expected Value')
    if relative is not None:
        if error:
            ax.set_ylabel('Error')
        else:
            color = 'C{}'.format(len(y.columns))
            ax.axhline(rel.iloc[0], xmin=0, xmax=1,
                       linestyle=':', linewidth=0.5,
                       color=color, label=relative)


def value_bar_chart_by_env(values, envs=None, relative=None, **kwargs):
    legend_height = 0.1
    legend_pad = 0.28
    fig_top = 1 - (legend_height + legend_pad)

    if envs is None:
        envs = values.index.levels[0]
    num_envs = len(envs)
    width, height = mpl.rcParams['figure.figsize']
    height = width / num_envs
    figsize = (width, height)
    fig, axs = plt.subplots(1, num_envs, figsize=figsize,
                            sharex=True, sharey=True,
                            gridspec_kw={'top': fig_top})

    for env, ax in zip(envs, axs):
        value_bar_chart(values.xs(env, level='env'), ax=ax,
                        legend=False, relative=relative, **kwargs)
        ax.set_title(env)

    handles, labels = ax.get_legend_handles_labels()
    if labels[0] == relative:
        # make relative label always go at the end
        labels = labels[1:] + [labels[0]]
        handles = handles[1:] + [handles[0]]
    leftmost = axs[0].xaxis.get_minpos()
    rightmost = axs[-1].get_position().xmax
    max_width = rightmost - leftmost
    x0 = leftmost + 0.05 * max_width
    x1 = rightmost - 0.05 * max_width
    num_algos = len(values.columns)
    fig.legend(handles, labels,
               loc='lower left', bbox_to_anchor=(x0, 0.9, x1 - x0, legend_height),
               mode='expand', ncol=num_algos, borderaxespad=0.)

    return fig

def value_latex_table(values, dps=2, relative=None, envs=None):
    df = values.xs('mean', level='type')
    if relative is not None:
        df = df.sub(df[relative], axis=0)
        df = df.drop(relative, axis=1)

    df = df.round(dps)
    df.columns.name = 'algo'
    df = df.unstack('m')

    def bold_group(group):
        best_idx = group.idxmax()
        best_idx_per_row = group.unstack().apply(pd.Series.idxmax)

        group = group.apply(str)
        for m, env in best_idx_per_row.iteritems():
            idx = env, m
            if idx != best_idx:
                group.loc[idx] = r'\textit{' + group.loc[idx] + '}'
        group[best_idx] = r'\textbf{' + group[best_idx] + '}'
        return group
    df = df.transform(bold_group, axis=1)
    df = df.stack('m').unstack('env')
    df.columns = df.columns.reorder_levels(['env', 'algo'])
    if envs is None:
        envs = df.columns.levels[0]
    df = pd.concat([df.loc[:, [x]] for x in envs], axis=1)
    return df.to_latex(escape=False)


def save_figs(figs, prefix):
    os.makedirs(prefix, exist_ok=True)
    for name, fig in figs:
        path = os.path.join(prefix, name.replace('/', '_') + '.pdf')
        fig.savefig(path)
