# -*- coding: utf-8 -*-
"""fix_data.

Created on 16.05.2022
Author: Matteo De Pellegrin
Email: matteo.dep97@gmail.com
"""


import os
import time
import numpy as np
import pandas as pd
import shutil
import warnings
from matplotlib import pyplot as plt


def load(path_name):
    if os.path.isfile(path_name + '.csv'):
        df = pd.read_csv(path_name + '.csv')
    elif os.path.isfile(path_name + '.xlsx'):
        old_df = pd.read_excel(path_name + '.xlsx')
        df = pd.DataFrame({
            'x1': old_df.loc[0],
            'y': old_df.loc[1],
            'x2': np.zeros(old_df.shape[1]),
            'temp': old_df.loc[3],
            't': old_df.loc[2],
            'temp1': old_df.loc[3],
            'temp2': old_df.loc[3],
        })
    else:
        warnings.warn(f"Could not load name '{name}'. File does not exist.")
        return None
    return df


def plot(ax, mode, df, units):
    cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
    ykeys, xkey = mode.split('/')
    ykeys = ykeys.split('-')
    u1 = None
    u0 = units[ykeys[0]]
    hs = []
    for i, ykey in enumerate(ykeys):
        if units[ykey] == u0:
            h_ = ax.scatter(df[xkey], df[ykey], s=5, edgecolors=None, c=cols[i], label=f'{ykey} [{units[ykey]}]')
        else:
            u1 = units[ykey]
            ax1 = ax.twinx()
            h_ = ax1.scatter(df[xkey], df[ykey], s=5, edgecolors=None, c=cols[i], label=f'{ykey} [{units[ykey]}]')
        hs.append(h_)
    ax.set_xlabel(f'{xkey} [{units[xkey]}]')
    ax.tick_params(axis='y')
    ax.legend(handles=hs)
    ax.set_ylabel(f'[{u0}]')
    if u1 is not None:
        ax1.tick_params(axis='y')
        ax1.set_ylabel(f'[{u1}]')
        ax1.set_ylim([1.2 * y for y in ax1.get_ylim()])


def save(df, path_name):
    df.to_csv(path_name + '.csv', index=False)

    name = os.path.basename(path_name)
    io_keys = ['x1', 'x2', 'y']
    vmax = {k: np.max(df[k]) for k in io_keys + ['t']}
    temp = np.mean(df['temp'])
    units = {k: 'A' if 0 < vmax[k] < 1e-2 else 'V' for k in io_keys}
    units['t'] = 's'
    main_input = 'x2' if vmax['x1'] == 0 else 'x1'

    info = "Temperature: {}K\n".format(temp) + \
        "Max input 1: {}{}\n".format(vmax['x1'], units['x1']) + \
        "Max input 2: {}{}\n".format(vmax['x2'], units['x2']) + \
        "Max output: {}{}\n".format(vmax['y'], units['y']) + \
        "Measurement took {}s ({})".format(vmax['t'], time.strftime('%Hh %Mm %Ss', time.gmtime(round(vmax['t']))))
    with open(path_name + '.txt', 'w+') as output_file:
        print(f'{info}\nSaved in {name}.csv with column order: input, output, gate, sample temperature' +
              ', time, arm temperature, cold head temperature', file=output_file)

    modes = [f'y/{main_input}', 'x1-x2-y/t']
    n_sp = len(modes)
    fig, axs = plt.subplots(n_sp, 1, figsize=(7, n_sp*4))
    fig.suptitle(f'{name} ({temp:.2f}K)')
    if n_sp == 1:
        axs = [axs]
    for ax, mode in zip(axs, modes):
        plot(ax, mode, df, units)
    fig.tight_layout()
    fig.savefig(path_name + '.png', dpi=100)
    plt.close()


def modify_gain(df, key, fact):
    df.loc[key] *= fact
    return df


if __name__ == "__main__":
    chip = "SIC1x"
    data_dir = os.path.join('data', chip)
    bkp_data_dir = os.path.join('data', 'bkp', chip)
    os.makedirs(bkp_data_dir, exist_ok=True)

    nums = np.arange(699, 846)
    names = [f"{chip}_{i}" for i in nums]

    for i, name in enumerate(names):
        print(f"\rProcessing {i+1} of {len(names)}.", end='', flush=True)
        path_name = os.path.join(data_dir, name)
        bkp_name = os.path.join(bkp_data_dir, name)
        for ext in ['.xlsx', '.csv', '.txt', '.png']:
            if os.path.isfile(path_name + ext):
                if not os.path.exists(bkp_name + ext):
                    shutil.copyfile(path_name + ext, bkp_name + ext)
        df = load(path_name)
        if df is not None:
            save(df, path_name)
    print(f"\rProcessed {len(names)} of {len(names)}.")
