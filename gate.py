# -*- coding: utf-8 -*-
"""gate.

Analize gate dependence.
"""


import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker

from analysis import ur, fmt, ulbl
import analysis as a
import data_plotter as dp


EXPERIMENT = os.path.splitext(os.path.basename(__file__))[0]
RES_DIR = os.path.join('results', EXPERIMENT)
os.makedirs(RES_DIR, exist_ok=True)


def get_gate_dependence(dh, names, fields, delta_field):
    gate = [] * ur.V
    conductance = [[] * ur.nS for f in fields]
    for i, name in enumerate(names):
        dh.load(name)
        gate_ = dh.get_gate(method='average')
        gate = np.append(gate, gate_)
        mask_denoise = dh.get_mask(current_win=[0.5*ur.pA, 1])
        full_conductance = dh.get_conductance(mask=mask_denoise)
        for j, field in enumerate(fields):
            field_win = [field - delta_field, field + delta_field]
            mask = dh.get_mask(field_win=field_win)
            if field == 0 * ur['V/um']:
                cond = np.sum(a.isnan(full_conductance[mask])) < 0.9 * np.sum(mask)
                if cond:
                    conductance_i = dh.get_conductance(method='fit', mask=mask)
                else:
                    conductance_i = np.nan
            else:
                cond = (np.sum(a.isnan(full_conductance[mask])) < 0.1 * np.sum(mask)) * (field_win[1] <= dh.prop['bias'] / dh.get_length())
                if cond:
                    conductance_i = a.average(full_conductance[mask])
                else:
                    conductance_i = np.nan
            conductance[j] = np.append(conductance[j], conductance_i)
    return gate, conductance


def main(data_dict):
    """Main analysis routine.
    names: list
    """
    fields = np.arange(0.05, 2, 0.05) * ur['V/um']
    delta_field = 0.005 * ur['V/um']
    dh = a.DataHandler()
    for chip in data_dict:
        dh.load_chip(chip)
        for pair in data_dict[chip]:
            nums = data_dict[chip][pair]['nums']
            names = np.array([f"{chip}_{i}" for i in nums])

            dh.load(names)
            print(f"Chip {chip}, Pair {pair} of length {fmt(dh.get_length())}.")

            names_dict = {}
            for name in names:
                key = fmt(dh.props[name]['temperature'], sep='')
                if key not in names_dict:
                    names_dict[key] = []
                names_dict[key].append(name)

            for temp_key in names_dict:
                print(f'Processing temperature {temp_key}')
                names = names_dict[temp_key]
                dh.load(names[0])
                max_field = np.amin(a.qlist_to_qarray([dh.props[name]['bias'] for name in names])) / dh.get_length()
                fact = max(a.separate_measurement((fields[4] + delta_field) // max_field)[0] + 1, 1)
                fields_ = fields / fact
                delta_field_ = delta_field / fact
                fields_ = fields_[fields_ + delta_field_ < max_field]

                gate, conductance = get_gate_dependence(dh, names, fields_, delta_field_)

                ranges = []
                directions = []
                last_index = 0
                while last_index < len(gate):
                    min_index = np.argmin(gate[last_index:]) + last_index
                    max_index = np.argmax(gate[last_index:]) + last_index
                    ranges.append(sorted([min_index, max_index]))
                    ranges[-1][1] += 1
                    directions.append('up' if ranges[-1][0] == min_index else 'down')
                    last_index = ranges[-1][1]

                # plot gate dependence
                ls = {'up': '--', 'down': ':'}
                ms = {'up': 'o', 'down': 'd'}
                x = gate
                x_, dx, ux = a.separate_measurement(x)
                fig, ax = plt.subplots(figsize=(7, 5))
                fig.suptitle("Gate Effect on Conductance")
                cbar, cols = dp.get_cbar_and_cols(fig, fields_, vmin=0)
                yminabs = None
                ymaxabs = None
                xspace = 0.1 * (np.amax(x_) - np.amin(x_))
                textcoord_index = np.argmax(x_)
                xtext = x_[textcoord_index] + xspace
                for i, field in enumerate(fields_):
                    y = conductance[i]
                    y_, dy, uy = a.separate_measurement(y)
                    maskout = np.isnan(y_)
                    if not maskout.all():
                        ymin = np.nanmin(y_)
                        ymax = np.nanmax(y_)
                        if yminabs is None or ymin < yminabs:
                            yminabs = ymin
                        if ymaxabs is None or ymax > ymaxabs:
                            ymaxabs = ymax
                        for d, (j0, j1) in zip(directions, ranges):
                            ax.errorbar(x_[j0:j1], y_[j0:j1],
                                        xerr=dx[j0:j1] if dx is not None else None,
                                        yerr=dy[j0:j1] if dx is not None else None,
                                        ls=ls[d], marker=ms[d], zorder=i, c=cols[i])
                        textcoord_index = np.argmax(np.where(maskout, -np.inf, x_))
                        ytext = y_[textcoord_index]
                        ax.text(xtext, ytext, r'$\mathbf{' + f'{ymax/ymin:.2f}' + r'}$')
                ax.set_yscale('log')
                matinv = ax.transLimits
                mat = matinv.inverted()
                xtext, ytext = mat.transform(matinv.transform([xtext, np.log10(ytext)]) + np.array([-0.3, 0.1]))
                ax.text(xtext, 10**ytext, r'$\mathbf{G_{max}/G_{min}:}$')
                ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
                if ymaxabs / yminabs < 5:
                    ax.yaxis.set_minor_formatter(ticker.ScalarFormatter())
                (xmin, ymin), (xmax, ymax) = mat.transform([[0, 0], [1.25, 1+0.15*min(len(directions), 2)]])
                ax.set_xlim([xmin, xmax])
                ax.set_ylim([10**ymin, 10**ymax])
                ax.legend(handles=[
                    plt.Line2D([0], [0], color='k', linestyle=ls[d], marker=ms[d], label=f"sweep {d}")
                    for d in ['up', 'down'] if d in directions
                ], loc='upper left')
                ax.set_xlabel(r"$V_G$" + ulbl(ux))
                ax.set_ylabel(r"$G$" + ulbl(uy))
                cbar.ax.set_ylabel(r"$\mathcal{E}_{{bias}}$" + ulbl(fields_.u))
                res_image = os.path.join(RES_DIR, f"{chip}_{pair}_{temp_key}_gate_dep.png")
                fig.savefig(res_image, bbox_inches='tight')
                plt.close()


if __name__ == "__main__":
    data_dict = {
        'SPC2': {
            'P16-P17': {
                'nums': np.concatenate([
                    np.arange(405, 419),
                    np.arange(419, 432),
                    np.arange(433, 483),
                ])
            },
            'P13-P14': {
                'nums': np.concatenate([
                    np.arange(509, 551),
                    np.arange(635, 677),
                ])
            },
            'P2-P4': {
                'nums': np.concatenate([
                    # np.arange(551, 572),
                    # np.arange(593, 614),
                    np.arange(719, 803),
                ])
            },
            'P1-P4': {
                'nums': np.concatenate([
                    # np.arange(572, 593),
                    # np.arange(614, 635),
                    np.arange(677, 719),
                    np.arange(804, 846),
                ])
            },
        },
        'SLBC2': {
            'P2-P3': {
                'nums': np.concatenate([
                    np.arange(6, 24),
                    np.arange(30, 48),
                    np.arange(52, 70),
                    np.arange(73, 91),
                    np.arange(93, 111),
                    np.arange(114, 132),
                    np.arange(134, 152),
                    np.arange(154, 162),
                    np.arange(163, 172),
                    np.arange(174, 182),
                    np.arange(183, 192),
                ])
            },
        },
    }

    main(data_dict)
