# -*- coding: utf-8 -*-
"""plot_data.

Plot data together.
"""


import os
import numpy as np
from matplotlib import pyplot as plt
import analysis as a


def plot_data():
    global names, mode, labels, res_names, titles, x_win, y_win
    global markersize, colors, show, save, correct_offset

    for i, names_ in enumerate(names):
        fig, ax = plt.subplots()
        for name, label, color in zip(names_, labels[i], colors[i]):
            dh.load(name)
            ax = dh.plot(ax, label=label, color=color, markersize=markersize, x_win=x_win, y_win=y_win)
        if titles is not None:
            ax.set_title(titles[i])
        if not np.array([label is None for label in labels[i]]).all():
            plt.legend()
        if show:
            plt.show()
        if save:
            fig.savefig(res_paths[i], dpi=300)


if __name__ == "__main__":
    # general
    show = True
    save = False
    correct_offset = True

    # paths
    exp = 'light_effect'
    chip = 'SPC2'
    data_dir = os.path.join('data', chip)
    res_dir = os.path.join('results', exp, 'SIC1x')
    os.makedirs(res_dir, exist_ok=True)

    dh = a.DataHandler(data_dir)

    # defaults
    res_names = None
    titles = None
    labels = "{name}"
    markersize = 5
    x_win = None
    y_win = None
    colors = None
    mode = 'i/v'

    # presets per experiment

    nums = np.arange(45, 55)

    names = [[f"{chip}_{i}"] for i in nums]
    mode = 'i/v'

    # check all parameters and arguments
    if isinstance(labels, str):
        labels = [[labels for name in names_] for names_ in names]
    if colors is None:
        colors = [[None for name in names_] for names_ in names]
    if res_names is None:
        res_paths = [os.path.join(res_dir, '-'.join(names_) + '.png') for names_ in names]
    else:
        res_paths = [os.path.join(res_dir, name + '.png') for name in res_names]

    plot_data()
