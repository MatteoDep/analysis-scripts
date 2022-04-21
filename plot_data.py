# -*- coding: utf-8 -*-
"""plot_together.

Plot data together.
"""


import os
import numpy as np
from matplotlib import pyplot as plt
import analysis as a


def main():
    global names, labels, res_names, titles, x_lim, y_lim, markersize, colors
    global show, save, correct_offset, plot_against_time
    # check all parameters and arguments
    if labels is None:
        labels = [[None for name in names_] for names_ in names]
    elif isinstance(labels, str):
        keys = labels.split('-')
        fstring = ", ".join(["{0[" + key + "]}" for key in keys if key != 'name'])
        if 'name' in keys:
            labels = [[f"{name} " + fstring.format(props[name]) for name in names_] for names_ in names]
        else:
            labels = [[fstring.format(props[name]) for name in names_] for names_ in names]
    if colors is None:
        colors = [[None for name in names_] for names_ in names]
    if res_names is None:
        res_paths = [os.path.join(res_dir, '-'.join(names_) + '.png') for names_ in names]
    else:
        res_paths = [os.path.join(res_dir, name + '.png') for name in res_names]

    if plot_against_time:
        x_key = 'time'
        x_label = 'Time [s]'
    else:
        x_key = 'voltage'
        x_label = 'Voltage [V]'
    y_key = 'current'
    y_label = 'Current [A]'

    for i, names_ in enumerate(names):
        fig, ax = plt.subplots()
        for name, label, color in zip(names_, labels[i], colors[i]):
            path = os.path.join(data_dir, name + '.xlsx')
            data = a.load_data(path, order=props[name]['order'])
            x = data[x_key]
            y = data[y_key]
            if correct_offset:
                if plot_against_time:
                    y -= np.mean(y[:10])
                else:
                    y -= np.mean(y)
            ax.scatter(x, y, label=label, c=color, s=markersize, edgecolors=None)
        if titles is not None:
            ax.set_title(titles[i])
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if x_lim is not None:
            if x_lim == 'auto':
                a.include_origin(ax, axis='x')
            else:
                ax.set_xlim(x_lim[i])
        if y_lim is not None:
            if y_lim == 'auto':
                a.include_origin(ax, axis='y')
            else:
                ax.set_ylim(y_lim[i])
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
    data_dir = os.path.join('data', 'SIC1x')
    res_dir = os.path.join('results', exp, 'SIC1x')
    os.makedirs(res_dir, exist_ok=True)
    prop_path = os.path.join(data_dir, 'properties.csv')
    props = a.load_properties(prop_path)

    # defaults
    res_names = None
    titles = None
    labels = "name"
    markersize = 5
    x_lim = 'auto'
    y_lim = 'auto'
    colors = None
    plot_against_time = False

    # presets per experiment

    names = [
        ['SIC1x_786', 'SIC1x_787'],
    ]
    plot_against_time = True

    # x_lim = None
    # names = [
    #     ['SIC1x_711',  'SIC1x_712'],
    #     ['SIC1x_713',  'SIC1x_714'],
    #     ['SIC1x_715',  'SIC1x_716'],
    #     ['SIC1x_717',  'SIC1x_718'],
    #     ['SIC1x_719',  'SIC1x_720'],
    #     ['SIC1x_721',  'SIC1x_722'],
    #     ['SIC1x_723',  'SIC1x_724'],
    #     ['SIC1x_725',  'SIC1x_726'],
    #     ['SIC1x_727',  'SIC1x_728'],
    #     ['SIC1x_729',  'SIC1x_730'],
    #     ['SIC1x_731',  'SIC1x_732'],
    #     ['SIC1x_733',  'SIC1x_734'],
    #     ['SIC1x_735',  'SIC1x_736'],
    #     ['SIC1x_737',  'SIC1x_738'],
    #     ['SIC1x_739',  'SIC1x_740'],
    #     ['SIC1x_741',  'SIC1x_742'],
    # ]
    # res_names = [
    #     'iv_hb_10K',
    #     'iv_lb_10K',
    #     'iv_hb_15K',
    #     'iv_lb_15K',
    #     'iv_hb_20K',
    #     'iv_lb_20K',
    #     'iv_hb_30K',
    #     'iv_lb_30K',
    #     'iv_hb_40K',
    #     'iv_lb_40K',
    #     'iv_hb_50K',
    #     'iv_lb_50K',
    #     'iv_hb_62K',
    #     'iv_lb_62K',
    #     'iv_hb_82K',
    #     'iv_lb_82K',
    # ]

    main()
