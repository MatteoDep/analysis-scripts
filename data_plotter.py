# -*- coding: utf-8 -*-
"""plot_data.

Plot data together.
"""


import os
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import Normalize
from matplotlib import ticker
import warnings
import analysis as a
from analysis import ur


def main():
    global names, modes, labels, res_names, titles, x_win, y_win
    global markersize, colors, show, save, correct_offset

    for i, names_ in enumerate(names):
        n_sp = len(modes)
        fig, axs = plt.subplots(n_sp, 1, figsize=(7, n_sp*4))
        if n_sp == 1:
            axs = [axs]
        for name, label, color in zip(names_, labels[i], colors[i]):
            dh.load(name)
            for j, m in enumerate(modes):
                axs[j] = dh.plot(axs[j], mode=m, label=label, color=color,
                                 markersize=markersize, x_win=x_win, y_win=y_win)
        if titles is not None:
            fig.suptitle(titles[i])
        if not np.array([label is None for label in labels[i]]).all():
            plt.legend()
        if show:
            plt.show()
        if save:
            fig.savefig(res_paths[i], dpi=100)


# OTHER FUNCIONS

def plot_iv_with_inset(dh, res_dir):
    field_win = [-0.05, 0.05] * ur['V/um']
    bias_win = field_win * dh.get_length()
    fig, ax1 = plt.subplots(figsize=(12, 9))
    dh.plot(ax1)
    ax2 = inset_axes(ax1, width='35%', height='35%', loc=4, borderpad=4)
    dh.plot(ax2, bias_win=bias_win)
    fig.suptitle(f"{dh.chip} {dh.prop['pair']} at {dh.prop['temperature']}")
    res_image = os.path.join(
        res_dir,
        "{1}_{2}_iv_{0.magnitude}{0.units}.png".format(
            dh.prop['temperature'],
            dh.chip,
            dh.prop['pair']
        )
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=UserWarning)
        fig.savefig(res_image, dpi=100)
    plt.close()


def get_cbar_and_cols(fig, values, log=False, ticks=None, **kwargs):
    values_, _, units = a.separate_measurement(values)
    if log:
        values_ = np.log(values_)
    norm = Normalize(**kwargs)
    sm = plt.cm.ScalarMappable(norm=norm, cmap='viridis')
    cols = plt.cm.viridis(norm(values_))
    cbar = fig.colorbar(sm)
    if ticks is None:
        ticks = values_
    else:
        ticks, _, units = a.separate_measurement(ticks)
    if log:
        # cbar.ax.set_yscale('log')
        cbar.ax.yaxis.set_major_locator(ticker.FixedLocator((np.log(ticks))))
    else:
        cbar.ax.yaxis.set_major_locator(ticker.FixedLocator((ticks)))
    cbar.ax.yaxis.set_major_formatter(ticker.FixedFormatter(([str(f*units) for f in ticks])))
    return cbar, cols


def include_origin(ax, axis='xy'):
    """Fix limits to include origin."""
    for a in axis:
        lim = getattr(ax, f"get_{a}lim")()
        d = np.diff(lim)[0] / 20
        lim = [min(lim[0], -d), max(lim[1], d)]
        getattr(ax, f"set_{a}lim")(lim)
    return ax


if __name__ == "__main__":
    # general
    show = True
    save = True
    correct_offset = True

    # paths
    exp = 'raw'
    chip = 'SPC3'
    data_dir = os.path.join('data', chip)
    res_dir = os.path.join('results', exp, chip)
    os.makedirs(res_dir, exist_ok=True)

    dh = a.DataHandler()
    dh.load_chip(chip)

    # defaults
    res_names = None
    titles = None
    labels = "{name}"
    markersize = 5
    x_win = None
    y_win = None
    colors = None
    modes = 'i/v'

    # presets per experiment

    nums = np.arange(30, 33)
    names = [[f"{chip}_{i}"] for i in nums]
    modes = ['i/t', 'vg/t']
    labels = None
    titles = [f"{chip} {dh.props[names_[0]]['pair']}" for names_ in names]
    res_names = [f"{chip}_{dh.props[names_[0]]['pair']}" for names_ in names]

    # check all parameters and arguments
    if isinstance(labels, str) or labels is None:
        labels = [[labels for name in names_] for names_ in names]
    if colors is None:
        colors = [[None for name in names_] for names_ in names]
    if res_names is None:
        res_paths = [os.path.join(res_dir, '-'.join(names_) + '.png') for names_ in names]
    else:
        res_paths = [os.path.join(res_dir, name + '.png') for name in res_names]
    if not isinstance(modes, list):
        modes = [modes]

    main()
