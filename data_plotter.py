# -*- coding: utf-8 -*-
"""plot_data.

Plot data together.
"""


import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import ticker
import warnings
import analysis as a
from analysis import ur


def plot_iv(dh, names, res_dir, zoom='auto', appendix=''):
    for i, name in enumerate(names):
        print(f"\rPlotting iv {i+1} of {len(names)}.", end='', flush=True)
        dh.load(name)
        fig, ax = plt.subplots(figsize=(12, 9))
        dh.plot(ax)
        if zoom == 'auto':
            zoom_ = dh.prop['temperature'] < 55*ur.K
        else:
            zoom_ = zoom
        if zoom_:
            mask = dh.get_mask([-0.05, 0.05] * ur['V/um'])
            ax_in = ax.inset_axes([0.65, 0.08, 0.3, 0.3])
            dh.plot(ax_in, mask=mask, set_xy_label=False)
            ax_in.set_xmargin(0)
            ax_in.set_ymargin(0)
            ax.indicate_inset_zoom(ax_in)
        fig.suptitle(f"{dh.chip} {dh.prop['pair']} at {dh.prop['temperature']}")
        res_image = os.path.join(
            res_dir, f"{dh.chip}_{dh.prop['pair']}_iv_{a.fmt(dh.prop['temperature'], sep='')}{appendix}.png"
        )
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UserWarning)
            fig.savefig(res_image, dpi=100)
        plt.close()
    print()


def get_cbar_and_cols(fig, values, log=False, ticks=None, **kwargs):
    values_, _, _ = a.separate_measurement(values)
    if log:
        values_ = np.log(values_)
    norm = Normalize(**kwargs)
    sm = plt.cm.ScalarMappable(norm=norm, cmap='viridis')
    cols = plt.cm.viridis(norm(values_))
    cbar = fig.colorbar(sm)
    if ticks is None:
        ticks = values_
    else:
        ticks, _, _ = a.separate_measurement(ticks)
    if log:
        # cbar.ax.set_yscale('log')
        cbar.ax.yaxis.set_major_locator(ticker.FixedLocator((np.log(ticks))))
    else:
        cbar.ax.yaxis.set_major_locator(ticker.FixedLocator((ticks)))
    cbar.ax.yaxis.set_major_formatter(ticker.FixedFormatter(([a.fmt(f) for f in ticks])))
    return cbar, cols


def include_origin(ax, axis='xy'):
    """Fix limits to include origin."""
    for k in axis:
        lim = getattr(ax, f"get_{k}lim")()
        d = np.diff(lim)[0] / 20
        lim = [min(lim[0], -d), max(lim[1], d)]
        getattr(ax, f"set_{k}lim")(lim)
    return ax


if __name__ == "__main__":
    # general
    show = True
    save = True
    correct_offset = True

    # paths
    exp = 'gate'
    chip = 'SPC2'
    data_dir = os.path.join('data', chip)
    res_dir = os.path.join('results', exp, chip)
    os.makedirs(res_dir, exist_ok=True)

    dh = a.DataHandler()
    dh.set_default_params(only_return=False)
    dh.load_chip(chip)

    # defaults
    res_names = None
    titles = None
    labels = "{name}"
    colors = None
    modes = 'i/v'

    # presets per experiment

    nums = [496]
    names = [[f"{chip}_{i}"] for i in nums]
    modes = ['i/t', 'vg/t']
    labels = None
    titles = [f"Gate Trace ({chip} {dh.props[names_[0]]['pair']})" for names_ in names]
    res_names = [f"{chip}_{dh.props[names_[0]]['pair']}_gate_trace" for names_ in names]

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

    for i, names_ in enumerate(names):
        n_sp = len(modes)
        fig, axs = plt.subplots(n_sp, 1, figsize=(7, n_sp*4))
        if n_sp == 1:
            axs = [axs]
        for name, label, color in zip(names_, labels[i], colors[i]):
            dh.load(name)
            for j, m in enumerate(modes):
                axs[j] = dh.plot(axs[j], mode=m, label=label, color=color)
        if titles is not None:
            fig.suptitle(titles[i])
        if not np.array([label is None for label in labels[i]]).all():
            plt.legend()
        fig.tight_layout()
        if show:
            plt.show()
        if save:
            fig.savefig(res_paths[i], dpi=100)
