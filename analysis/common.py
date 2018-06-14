import collections
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
from matplotlib.colors import ListedColormap
import pandas as pd
import numpy as np
import scipy.stats
import seaborn as sns

from pirl.envs import jungle_topology

logger = logging.getLogger('analysis.common')
THIS_DIR = osp.join(os.path.dirname(os.path.realpath(__file__)))

def style(name):
    return osp.join(THIS_DIR, '{}.mplstyle'.format(name))

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

    idx = ['seed', 'eval', 'env', 'type']
    ground_truth = nested_dicts_to_df(data['ground_truth'], idx, unpack_mean_sd_tuple)
    ground_truth = ground_truth.stack().unstack('eval')
    ground_truth = ground_truth.reorder_levels(['env', 'seed', 'type'])
    ground_truth.columns.name = None

    idxs = ['seed', 'eval', 'irl', 'env', 'n', 'm', 'type']
    values = nested_dicts_to_df(data['values'], idxs, unpack_mean_sd_tuple)
    values = values.stack().unstack('irl')
    values.columns.name = 'irl'

    sorted_idx = ['env', 'n', 'm', 'eval', 'seed', 'type']
    if not values.empty:
        values = values.reorder_levels(sorted_idx)
        idx = values.index
    else:
        idx = [(env, 0, 0, 'gt', seed, kind)
               for env, seed, kind in tuple(ground_truth.index)]
        idx = pd.MultiIndex.from_tuples(idx, names=sorted_idx)

    def get_gt(k):
        env, _, _, _, seed, kind = k
        return ground_truth.loc[(env, seed, kind), :]
    values_gt = pd.DataFrame(list(map(get_gt, idx)), index=idx)
    values = pd.concat([values, values_gt], axis=1)

    return values

def load_value(experiment_dir, algo_pattern='(.*)', env_pattern='(.*)', algos=['.*'], dps=2):
    fname = osp.join(experiment_dir, 'results.pkl')
    data = pd.read_pickle(fname)

    value = extract_value(data)
    value.columns = value.columns.str.extract(algo_pattern, expand=False)
    envs = value.index.levels[0].str.extract(env_pattern, expand=False)
    value.index = value.index.set_levels(envs, level=0)

    matches = []
    mask = pd.Series(False, index=value.columns)
    for p in algos:
        m = value.columns.str.match(p)
        matches += list(value.columns[m & (~mask)])
        mask |= m
    value = value.loc[:, matches]

    value.columns = value.columns.str.split('_').str.join(' ')  # so lines wrap
    value = value.round(dps)
    return value

def _extract_means_ses(values, with_seed=True):
    nil_slices = (slice(None),) * (len(values.index.levels) - 1)
    means = values.loc[nil_slices + ('mean',), :].copy()
    means.index = means.index.droplevel('type')
    ses = values.loc[nil_slices + ('se',), :].copy()
    ses.index = ses.index.droplevel('type')
    return means, ses


def _combine_means_ses(means, ses):
    means['type'] = 'mean'
    means = means.set_index('type', append=True)
    ses['type'] = 'se'
    ses = ses.set_index('type', append=True)
    return pd.concat([means, ses])

def aggregate_value(values, n=100):
    '''Aggregate mean and standard error across seeds. We assume the same number
       of samples n are used to calculate the mean and s.e. of each seed.'''
    means, ses = _extract_means_ses(values)

    # The mean is just the mean across seeds
    mean = means.stack().unstack('seed').mean(axis=1).unstack(-1)

    # Reconstruct mean-of-squares
    squares = (ses * ses * n) + (means * means)
    mean_square = squares.stack().unstack('seed').mean(axis=1).unstack(-1)

    # Back out standard error
    var = mean_square - (mean * mean)
    se = np.sqrt(var) / np.sqrt(n)

    return _combine_means_ses(mean, se)

def plot_ci(values, dp=3):
    mean, se = _extract_means_ses(values)
    fstr = '{:.' + str(dp) + 'f}'
    return mean.applymap(lambda x: (fstr + ' +/- ').format(x)) + se.applymap(lambda x: fstr.format(1.96 * x))

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


def gridworld_cartoon(shape, **kwargs):
    aspect_ratio = 1
    width, height = mpl.rcParams['figure.figsize']
    height = width / aspect_ratio

    fig = plt.figure(figsize=(width, height))
    gs = gridspec.GridSpec(3, 5)
    cartoon_ax = fig.add_subplot(gs[0:2, 1:4])
    legend_ax = fig.add_subplot(gs[2, :])
    #fig, axs = plt.subplots(2,1, gridspec_kw=dict(height_ratios=[2,1]), figsize=(width, height))
    #cartoon_ax, legend_ax = axs
    kind_to_colors = {
        'X': '#000000', # 'wall' (unnreachable)
        'A': '#9c755f',  # start state/'dirt' (-1 reward)
        ' ': '#9c755f',  # default cell/'dirt' (-1 reward)
        'R': '#59a14f', # grass (0 reward)
        'L': '#ff0000', # lava (-10 reward)
        'S': '#C0C0C0', # silver (+1 reward)
        'W': '#ffd700', # gold (+1 reward)
    }
    kind_to_colors_list = list(kind_to_colors.items())
    kind_to_idx = {k: i for i, (k, color) in enumerate(kind_to_colors_list)}
    colors = [color for (k, color) in kind_to_colors_list]
    cmap = ListedColormap(colors)

    topology = jungle_topology['{}x{}'.format(shape[0], shape[1])]
    idx = np.vectorize(kind_to_idx.get)(topology)

    kwargs.setdefault('cbar', False)
    kwargs.setdefault('cmap', cmap)
    sns.heatmap(idx, ax=cartoon_ax, **kwargs)

    from matplotlib.lines import Line2D
    legend = collections.OrderedDict([
        ('A', '-1\n-1\n-1'),
        ('R', '0\n0\n0'),
        ('L', '-10\n-10\n-10'),
        ('S', '0\n1\n1'),
        ('W', '1\n0\n1'),
    ])
    patches = [Line2D([0], [0], linewidth=0, markersize=12, marker='s', color=kind_to_colors[kind])
               for kind in legend.keys()]
    labels = list(legend.values())
    legend_ax.legend(patches, labels, fontsize=8, markerfirst=True,
                     loc='lower left', bbox_to_anchor=(0.0, 0.0, 1.0, 0.1),
                     mode='expand', ncol=len(legend) + 1, borderaxespad=0.)
    legend_ax.axis('off')

    return fig

def value_bar_chart(values, alpha=0.05, relative=None,
                    error=False, ax=None, **kwargs):
    '''Takes two DataFrames with columns corresponding to algorithms, and
       the index to the number of trajectories. It outputs a stacked bar graph.'''
    mean, se = _extract_means_ses(values)
    if ax is None:
        ax = plt.gca()
    z = scipy.stats.norm.ppf(1 - (alpha / 2))
    err = se * z

    if relative is not None:
        if error:
            mean = -mean.sub(mean[relative], 0)
        else:
            rel = mean[relative]

        mean = mean.drop(relative, axis=1)
        err = err.drop(relative, axis=1)

        if not error:
            # Clip error bars to not be above maximum possible
            err = err.values  # err: M*N
            bottom_err = err
            max_err = rel.values.reshape(-1, 1) - mean.values
            top_err = np.maximum(0, np.minimum(max_err, err))
            err = np.array([bottom_err, top_err])
            err = err.transpose([2, 0, 1])  # err: N*2*M
    mean.plot.bar(yerr=err, ax=ax, error_kw=dict(lw=0.5), **kwargs)

    ax.set_xlabel('Trajectories')
    ax.set_ylabel('Expected Value')
    if relative is not None:
        if error:
            ax.set_ylabel('Error')
        else:
            color = 'C{}'.format(len(mean.columns))
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
    x0 = leftmost + 0.025 * max_width
    x1 = rightmost - 0.05 * max_width
    num_algos = len(values.columns)
    fig.legend(handles, labels,
               loc='lower left', bbox_to_anchor=(x0, 0.9, x1 - x0, legend_height),
               mode='expand', ncol=num_algos, borderaxespad=0.)

    return fig

def value_latex_table(mean, dps=2, relative=None, envs=None):
    if relative is not None:
        mean = mean.sub(mean[relative], axis=0)
        mean = mean.drop(relative, axis=1)

    mean = mean.round(dps)
    mean.columns.name = 'algo'
    mean = mean.unstack('m')

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
    mean = mean.transform(bold_group, axis=1)
    mean = mean.stack('m').unstack('env')
    mean.columns = mean.columns.reorder_levels(['env', 'algo'])
    if envs is None:
        envs = mean.columns.levels[0]
    mean = pd.concat([mean.loc[:, [x]] for x in envs], axis=1)
    return mean.to_latex(escape=False)


def save_figs(figs, prefix):
    os.makedirs(prefix, exist_ok=True)
    for name, fig in figs:
        path = os.path.join(prefix, name.replace('/', '_') + '.pdf')
        fig.savefig(path)
