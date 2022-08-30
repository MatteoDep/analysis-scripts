# -*- coding: utf-8 -*-
"""length_characterization.

Characterize resistance at room temperature for different segments.
"""


import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker
import analysis as a
from analysis import ur, fmt, ulbl


EXPERIMENT = os.path.splitext(os.path.basename(__file__))[0]
RES_DIR = os.path.join('results', EXPERIMENT)
os.makedirs(RES_DIR, exist_ok=True)


def get_length_dependence(dh, names, field, delta_field):
    injs = ['2p', '4p']
    length = {inj: [] * ur.um for inj in injs}
    resistance = {inj: [] * ur.Mohm for inj in injs}
    pair = {'2p': [], '4p': []}
    for i, name in enumerate(names):
        dh.load(name)
        inj = dh.prop['injection'].lower()
        length[inj] = np.append(length[inj], dh.get_length())
        pair[inj] = np.append(pair[inj], dh.prop['pair'])
        mask = dh.get_mask([-delta_field, delta_field])
        resistance_i = dh.get_resistance(method='fit', mask=mask)
        resistance[inj] = np.append(resistance[inj], resistance_i)
    return length, resistance, pair


def plot_fit(dh, data_dict):
    delta_field = 0.001*ur['V/um']

    for chip in data_dict:
        dh.load_chip(chip)
        nums = data_dict[chip]['nums']
        names = np.array([f"{chip}_{i}" for i in nums])
        for i, name in enumerate(names):
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
            x_model = np.array([np.amin(x.m), np.amax(x.m)])
            ax.plot(x_model, model(x_model), c='r', label='fit')
            ax_in = ax.inset_axes([0.65, 0.08, 0.3, 0.3])
            ax_in.plot(x_, y_, 'o')
            ax_in.plot(x_, model(x_), c='r')
            ax_in.set_xmargin(0)
            ax_in.set_ymargin(0)
            ax_in.set_title('fit region')
            ax.indicate_inset_zoom(ax_in)
            ax.legend()
            res_image = os.path.join(RES_DIR, f"{chip}_{dh.prop['pair']}_{inj}_resistance.png")
            plt.savefig(res_image)
            plt.close()


def main(dh, data_dict, pair_on_axis=False):
    """Compute resistance over length characterization."""
    field = 0*ur['V/um']
    delta_field = 0.01*ur['V/um']

    for chip in data_dict:
        dh.load_chip(chip)
        nums = data_dict[chip]['nums']
        names = np.array([f"{chip}_{i}" for i in nums])
        dh.load(names)
        print("Chip {}.".format(chip))

        length, resistance, pair = get_length_dependence(dh, names, field, delta_field)

        injs = ['2p', '4p']
        if len(length['4p']) > 0:
            coeffs, model = a.fit_linear(length['4p'], resistance['4p'])
            resistivity = coeffs[0] * a.NUM_FIBERS * np.pi * a.FIBER_RADIUS**2
            conductivity = (1 / resistivity).to('S / cm')
            print(f"conductivity: {fmt(conductivity)}")
        else:
            injs.remove('4p')
        if len(length['2p']) == 0:
            injs.remove('2p')

        for inj in injs:
            figsize = (0.15 * np.amax(a.separate_measurement(length[inj])[0]), 9) if pair_on_axis else None
            fig, ax = plt.subplots(figsize=figsize)
            inj_label = inj.replace('p', '-probe')
            fig.suptitle(f"{inj_label} Resistance")
            x = length[inj]
            x_, dx, ux = a.separate_measurement(x)
            pair_label = [f"{p}\n{x[i]}" for i, p in enumerate(pair[inj])]
            y = resistance[inj]
            y_, dy, uy = a.separate_measurement(y)
            ax.errorbar(x_, y_, xerr=dx, yerr=dy, fmt='o', label='data')
            if inj == '4p':
                x_model = np.array([0, np.amax(x_)*1.1])
                y_model = model(x_model)
                ax.plot(x_model, y_model, label=fr'fit ($\sigma = {fmt(conductivity, latex=True)})$')
                ax.legend()
            if pair_on_axis:
                ax.xaxis.set_major_locator(ticker.FixedLocator((x_)))
                ax.xaxis.set_major_formatter(ticker.FixedFormatter((pair_label)))
                plt.setp(ax.get_xticklabels(), rotation=60, horizontalalignment='right', fontsize='x-small')
                res_image = os.path.join(RES_DIR, f"{chip}_{inj}_resistance_pairs.png")
            else:
                ax.set_xlabel("Length" + ulbl(ux))
                res_image = os.path.join(RES_DIR, f"{chip}_{inj}_resistance.png")
            ax.set_ylabel("$R$" + ulbl(uy))
            plt.savefig(res_image)
            plt.close()

        # make compatible
        if '2p' in injs and '4p' in injs:
            common_length = [] * ur.um
            contact_resistance_rel = []
            common_pair_label = []
            for i, p in enumerate(pair['2p']):
                for j, p_ in enumerate(pair['4p']):
                    if p_ == p:
                        common_length = np.append(common_length, length['2p'][i])
                        common_pair_label = np.append(common_pair_label, f"{p}\n{common_length[-1]}")
                        contact_resistance_rel = np.append(
                            contact_resistance_rel,
                            (resistance['2p'][i] - resistance['4p'][j]) / resistance['2p'][i]
                        )
            figsize = (0.15 * np.amax(a.separate_measurement(common_length)[0]), 9) if pair_on_axis else None
            fig, ax = plt.subplots(figsize=figsize)
            fig.suptitle("Contact Resistance")
            x = common_length
            x_, dx, ux = a.separate_measurement(x)
            y = contact_resistance_rel
            y_, dy, uy = a.separate_measurement(y)
            ax.errorbar(x_, y_, xerr=dx, yerr=dy, fmt='o')
            if pair_on_axis:
                ax.xaxis.set_major_locator(ticker.FixedLocator((x_)))
                ax.xaxis.set_major_formatter(ticker.FixedFormatter((pair_label)))
                plt.setp(ax.get_xticklabels(), rotation=60, horizontalalignment='right', fontsize='x-small')
                res_image = os.path.join(RES_DIR, f"{chip}_contact_resistance_pairs.png")
            else:
                ax.set_xlabel("Length" + ulbl(ux))
                res_image = os.path.join(RES_DIR, f"{chip}_contact_resistance.png")
            ax.set_ylabel(r"$R_{cont}/R_{2P}$")
            plt.savefig(res_image)
            plt.close()


if __name__ == "__main__":
    data_dict = {
        'SPC2': {
            'nums': np.concatenate([
                np.arange(1, 23),
                np.arange(25, 31),
            ])
        },
        'SPC3': {
            'nums': np.concatenate([
                [1],
                np.arange(3, 10),
                np.arange(11, 13),
                np.arange(14, 18),
            ])
        },
        'SQC1': {
            'nums': np.concatenate([
                np.arange(7, 12),
                np.arange(13, 37),
            ])
        }
    }

    dh = a.DataHandler()

    main(dh, data_dict, pair_on_axis=False)

    data_dict = {
        'SPC2': {
            'nums': [11, 18]
        },
    }
    plot_fit(dh, data_dict)
