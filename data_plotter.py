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
            coeffs, model = a.fit_linear((x_, None, ux), (y_, None, uy), already_separated=True)
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
            res_image = os.path.join(RES_DIR, f"{chip}_{dh.prop['pair']}_{inj}_resistance.png")
            plt.savefig(res_image)
            plt.close()


def plot_iv(dh, names, zoom='auto', appendix=''):
    for chip in data_dict:
        dh.load_chip(chip)
        nums = data_dict[chip]['nums']
        names = np.array([f"{chip}_{i}" for i in nums])
        for name in names:
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
                RES_DIR, f"{dh.chip}_{dh.prop['pair']}_iv_{a.fmt(dh.prop['temperature'], sep='')}{appendix}.png"
            )
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=UserWarning)
                fig.savefig(res_image, dpi=100)
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
            fig.suptitle(f"Gate Trace ({chip} {dh.prop['pair']})")
            fig.tight_layout()
            res_image = os.path.join(RES_DIR, f"{chip}_{dh.prop['pair']}_gate_trace.png")
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
    dh = a.DataHandler()

    data_dict = {
        'SPC3': {
            'nums': [30],
        },
        'SPC2': {
            'nums': range(484, 498),
        },
    }
    plot_gate_trace(dh, data_dict)

    data_dict = {
        'SPC2': {
            'nums': [282],
        },
    }
    plot_iv(dh, data_dict, zoom=False, appendix='_highbias')
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
