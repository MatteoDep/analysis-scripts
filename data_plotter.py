# -*- coding: utf-8 -*-
"""plot_data.

Plot data together.
"""


import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import ticker
import analysis as a
from analysis import ur, fmt


RES_DIR = os.path.join('results', 'data')
os.makedirs(RES_DIR, exist_ok=True)


def plot_fit(dh, data_dict):
    delta_field = 0.001*ur['V/um']
    for chip in data_dict:
        dh.load_chip(chip)
        nums = data_dict[chip]['nums']
        names = np.array([f"{chip}_{i}" for i in nums])
        for name in names:
            dh.load(name)
            inj = dh.prop['injection'].lower()
            mask = dh.get_mask([-delta_field, delta_field])
            bias = dh.get_bias()
            current = a.q_to_compact(dh.get_current(correct_offset=False))
            if inj == '2p':
                x, y = bias, current
            else:
                x, y = current, bias
            x_, _, ux = a.separate_measurement(x[mask])
            y_, _, uy = a.separate_measurement(y[mask])
            coeffs, model = a.fit_linear((x_, None, ux), (y_, None, uy))
            fig, ax = plt.subplots()
            inj_label = inj.replace('p', '-probe')
            fig.suptitle(f'{inj_label} Measurement')
            mode = 'i/v' if inj == '2p' else 'v/i'
            ax = dh.plot(ax, mode=mode, label='data')
            ax.plot(x_, y_, 'o', c='C2', label='fitted data')
            x_model = np.array([np.amin(x.m), np.amax(x.m)])
            ax.plot(x_model, model(x_model), c='r', label='fit')
            ax_in = ax.inset_axes([0.65, 0.08, 0.3, 0.3])
            ax_in.plot(x_, y_, 'o', c='C2')
            ax_in.plot(x_, model(x_), c='r')
            ax_in.set_xmargin(0)
            ax_in.set_ymargin(0)
            ax_in.set_title('fit region')
            ax.indicate_inset_zoom(ax_in)
            ax.legend()
            res_image = os.path.join(RES_DIR, f"{chip}_{dh.prop['pair']}_{inj}_resistance.pdf")
            plt.savefig(res_image)
            plt.close()


def plot_ivs(dh, data_dict):
    for chip in data_dict:
        dh.load_chip(chip)
        nums = data_dict[chip]['nums']
        names = np.array([f"{chip}_{i}" for i in nums])
        names_dict = {}
        lengths = []
        for name in names:
            dh.load(name)
            length = dh.get_length()
            key = fmt(length)
            if key not in names_dict:
                names_dict[key] = []
                lengths.append(length.m)
            names_dict[key].append(name)
        idx = np.argsort(lengths)
        n_sf = len(names_dict)
        fig = plt.figure(constrained_layout=True, figsize=(18, n_sf*9))
        fig.suptitle(f'Example I-Vs from chip {chip}')
        subfigs = fig.subfigures(nrows=n_sf, ncols=1)
        if n_sf == 1:
            subfigs = [subfigs]
            pairs = []
        for i, length in enumerate(names_dict):
            subfigs[idx[i]].suptitle(f'Segment of length {length}')
            axs = subfigs[idx[i]].subplots(1, 2)
            for name in names_dict[length]:
                dh.load(name)
                inj = dh.prop['injection'].lower()
                if inj == '2p':
                    dh.plot(axs[0], mode='i/v')
                    axs[0].set_title('2-probe')
                else:
                    dh.plot(axs[1], mode='v/i')
                    axs[1].set_title('4-probe')
            pairs.append(dh.prop['pair'])
        res_image = os.path.join(
            RES_DIR, f"{dh.chip}_{'_'.join(pairs)}_iv_2-4p.pdf"
        )
        fig.savefig(res_image)
        plt.close()


def plot_iv(dh, data_dict, highbias=False):
    for chip in data_dict:
        dh.load_chip(chip)
        nums = data_dict[chip]['nums']
        names = np.array([f"{chip}_{i}" for i in nums])
        for name in names:
            dh.load(name)
            fig, ax = plt.subplots(figsize=(12, 9))
            fig.suptitle(f"Example I-V at {dh.prop['temperature']}")
            dh.plot(ax,)
            if not highbias and dh.prop['temperature'] < 55*ur.K:
                mask = dh.get_mask([-0.05, 0.05] * ur['V/um'])
                ax_in = ax.inset_axes([0.65, 0.08, 0.3, 0.3])
                dh.plot(ax_in, mask=mask, set_xy_label=False)
                ax_in.set_xmargin(0)
                ax_in.set_ymargin(0)
                ax.indicate_inset_zoom(ax_in)
            res_image = os.path.join(
                RES_DIR, f"{dh.chip}_{dh.prop['pair']}_iv_{fmt(dh.prop['temperature'], sep='')}{'_highbias' if highbias else ''}.pdf"
            )
            fig.savefig(res_image)
            plt.close()


def plot_gate_trace(dh, title=None, res_names=None):
    modes = ['i/t', 'vg/t']
    for chip in data_dict:
        dh.load_chip(chip)
        nums = data_dict[chip]['nums']
        names = np.array([f"{chip}_{i}" for i in nums])
        for name in names:
            dh.load(name)
            n_sp = len(modes)
            fig, axs = plt.subplots(n_sp, 1, figsize=(9, 10))
            if n_sp == 1:
                axs = [axs]
            for j, m in enumerate(modes):
                axs[j] = dh.plot(axs[j], mode=m)
            fig.suptitle(f"Example Gate Trace on chip {chip}")
            fig.tight_layout()
            res_image = os.path.join(RES_DIR, f"{chip}_{dh.prop['pair']}_gate_trace.pdf")
            fig.savefig(res_image)


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
    cbar.ax.yaxis.set_major_formatter(ticker.FixedFormatter(([fmt(f) for f in ticks])))
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
    dh = a.DataHandler()

    data_dict = {
        'SPC3': {
            'nums': [30],
        },
        'SPC2': {
            'nums': [492],
        },
    }
    plot_gate_trace(dh, data_dict)

    data_dict = {
        'SPC2': {
            'nums': [5, 19],
        },
        'SPC3': {
            'nums': [7, 15],
        },
        'SQC1': {
            'nums': [9, 10, 14, 23, 24, 28],
        },
    }
    plot_ivs(dh, data_dict)

    data_dict = {
        'SPC2': {
            'nums': [282],
        },
    }
    plot_iv(dh, data_dict, highbias=True)
    data_dict = {
        'SPC2': {
            'nums': [283, 297, 305, 332],
        },
    }
    plot_iv(dh, data_dict)

    data_dict = {
        'SPC2': {
            'nums': [11, 26],
        },
    }
    plot_fit(dh, data_dict)
