# -*- coding: utf-8 -*-
"""plot_together.

Plot data together.
"""


import os
import numpy as np
from matplotlib import pyplot as plt
import analysis as a


if __name__ == "__main__":
    # general
    custom = False
    correct_offset = True

    # paths
    exp = 'light_effect'
    data_dir = os.path.join('data', 'SIC1x')
    res_dir = os.path.join('results', exp, 'SIC1x')
    os.makedirs(res_dir, exist_ok=True)
    prop_path = os.path.join(data_dir, 'properties.json')
    props = a.load_properties(prop_path)

    # init variables to None
    res_names = None
    titles = None
    labels = None
    markersize = 5
    x_lim = None
    y_lim = None

    # # presets per experiment
    # plot_against_time = True
    # markersize = 1
    # names = [
    #     ['SIC1x_772', 'SIC1x_773', 'SIC1x_774', 'SIC1x_775', 'SIC1x_776'],
    #     ['SIC1x_788', 'SIC1x_789', 'SIC1x_790', 'SIC1x_791', 'SIC1x_792'],
    #     ['SIC1x_835', 'SIC1x_836', 'SIC1x_837', 'SIC1x_838', 'SIC1x_839'],
    # ]
    # titles = [
    #     'Constant Power - termal paste',
    #     'Constant Power - glass',
    #     'Constant Power - nothing',
    # ]
    # res_names = [
    #     'thermal_paste',
    #     'glass',
    #     'nothing',
    # ]
    # labels = [
    #     ['blue', 'red', 'green', 'white', 'dark']
    #     for _ in names
    # ]
    # x_lim = [
    #     [0, 150],
    #     [0, 100],
    #     [0, 150],
    # ]

    plot_against_time = False
    x_lim = None
    names = [
        ['SIC1x_711',  'SIC1x_712'],
        ['SIC1x_713',  'SIC1x_714'],
        ['SIC1x_715',  'SIC1x_716'],
        ['SIC1x_717',  'SIC1x_718'],
        ['SIC1x_719',  'SIC1x_720'],
        ['SIC1x_721',  'SIC1x_722'],
        ['SIC1x_723',  'SIC1x_724'],
        ['SIC1x_725',  'SIC1x_726'],
        ['SIC1x_727',  'SIC1x_728'],
        ['SIC1x_729',  'SIC1x_730'],
        ['SIC1x_731',  'SIC1x_732'],
        ['SIC1x_733',  'SIC1x_734'],
        ['SIC1x_735',  'SIC1x_736'],
        ['SIC1x_737',  'SIC1x_738'],
        ['SIC1x_739',  'SIC1x_740'],
        ['SIC1x_741',  'SIC1x_742'],
    ]
    res_names = [
        'iv_hb_10K',
        'iv_lb_10K',
        'iv_hb_15K',
        'iv_lb_15K',
        'iv_hb_20K',
        'iv_lb_20K',
        'iv_hb_30K',
        'iv_lb_30K',
        'iv_hb_40K',
        'iv_lb_40K',
        'iv_hb_50K',
        'iv_lb_50K',
        'iv_hb_62K',
        'iv_lb_62K',
        'iv_hb_82K',
        'iv_lb_82K',
    ]

    # check all parameters and arguments
    if labels is None:
        labels = names
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

    # for names, orders, labels, x_lim in zip(names_list, orders_list, labels_list, x_lim_list):
    for i, names_ in enumerate(names):
        fig, ax = plt.subplots()
        for name, label in zip(names_, labels[i]):
            path = os.path.join(data_dir, name + '.xlsx')
            data = a.load_data(path, order=props[name]['order'])
            x = data[x_key]
            y = data[y_key]
            if correct_offset:
                if plot_against_time:
                    y -= np.mean(y[:10])
                else:
                    y -= np.mean(y)
            ax.scatter(x, y, label=label, s=markersize, edgecolors=None)
        if titles is not None:
            ax.set_title(titles[i])
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if x_lim is not None:
            ax.set_xlim(x_lim[i])
        if y_lim is not None:
            ax.set_ylim(y_lim[i])
        plt.legend()
        fig.savefig(res_paths[i], dpi=300)
