# -*- coding: utf-8 -*-
"""length_characterization.

Characterize resistance at room temperature for different segments.
"""


import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker
import analysis as a
from analysis import ur


EXPERIMENT = os.path.splitext(os.path.basename(__file__))[0]


def get_length_dependence(names):
    global dh, res_dir, plot_ivs
    time_win = [0.25, 0.75]

    injs = ['2p', '4p']
    length = {inj: [] * ur.um for inj in injs}
    resistance = {inj: [] * ur['Mohm'] for inj in injs}
    pair = {'2p': [], '4p': []}
    for i, name in enumerate(names):
        print(f"\rProcessing {i+1} of {len(names)}.", end='', flush=True)
        dh.load(name)
        inj = dh.prop['injection'].lower()
        length[inj] = np.append(length[inj], dh.get_length())
        pair[inj] = np.append(pair[inj], dh.prop['pair'])
        if dh.prop['temperature'] > 100 * ur.K:
            bias = 0 * ur.V
            delta_bias = dh.prop['bias'] / 5
            method = 'fit'
        else:
            bias = 0.2 * ur['V/um'] * dh.get_length()
            delta_bias = 0.01 * ur['V/um'] * dh.get_length()
            method = 'average'
        bias_win = [bias - delta_bias, bias + delta_bias]
        resistance0 = dh.get_resistance(method=method, time_win=time_win, bias_win=bias_win)
        resistance[inj] = np.append(resistance[inj], resistance0)
    print()
    return length, resistance, pair


def main(data_dict, noise_level=0.5*ur.pA):
    """Compute resistance over length characterization."""
    global dh, res_dir
    injs = ['2p', '4p']

    dh = a.DataHandler()
    for chip in data_dict:
        dh.load_chip(chip)
        res_dir = os.path.join('results', EXPERIMENT, chip)
        os.makedirs(res_dir, exist_ok=True)

        nums = data_dict[chip]['nums']
        names = np.array([f"{chip}_{i}" for i in nums])

        print("Chip {}.".format(chip))

        names_dict = {}
        for name in names:
            key = str(dh.props[name]['temperature']).replace(' ', '')
            if key not in names_dict:
                names_dict[key] = []
            names_dict[key].append(name)

        for temp_key in names_dict:
            print(f'Temperature {temp_key}:')
            names = names_dict[temp_key]

            length, resistance, pair = get_length_dependence(names)

            if len(length['4p']) > 0:
                coeffs, reslen_model = a.fit_linear(length['4p'], resistance['4p'])
                resistivity = coeffs[0] * a.NUM_FIBERS * np.pi * a.FIBER_RADIUS**2
                conductivity = (1 / resistivity).to('S / cm')
                print(f"conductivity: {conductivity}")
            else:
                injs.remove('4p')
            if len(length['2p']) == 0:
                injs.remove('2p')

            for inj in injs:
                fig, ax = plt.subplots(figsize=(20, 10))
                inj_label = inj.replace('p', '-probe')
                ax.set_title(f"{inj_label} resistance ({temp_key})")
                x = length[inj]
                x_, dx, ux = a.separate_measurement(x)
                pair_label = [f"{p}\n{x[i]}" for i, p in enumerate(pair[inj])]
                y = resistance[inj]
                y_, dy, uy = a.separate_measurement(y)
                ax.errorbar(x_, y_, xerr=dx, yerr=dy, fmt='o', label=inj_label)
                if inj == '4p':
                    y1 = reslen_model(x_)
                    ax.plot(x_, y1, '-', label=f'{inj} fit')
                ax.xaxis.set_major_locator(ticker.FixedLocator((x_)))
                ax.xaxis.set_major_formatter(ticker.FixedFormatter((pair_label)))
                plt.setp(ax.get_xticklabels(), rotation=60, horizontalalignment='right', fontsize='x-small')
                ax.set_ylabel(f"$R$ [${uy:~L}$]")
                plt.tight_layout()
                res_image = os.path.join(res_dir, f"{temp_key}_{inj}_resistance.png")
                plt.savefig(res_image, dpi=100)

            # make compatible
            if '2p' in injs and '4p' in injs:
                common_length = [] * ur.um
                contact_resistance = [] * ur['Mohm']
                common_pair_label = []
                for i, p in enumerate(pair['2p']):
                    for j, p_ in enumerate(pair['4p']):
                        if p_ == p:
                            common_length = np.append(common_length, length['2p'][i])
                            common_pair_label = np.append(common_pair_label, f"{p}\n{common_length[-1]}")
                            contact_resistance = np.append(
                                contact_resistance, resistance['2p'][i] - resistance['4p'][j]
                            )

                fig, ax = plt.subplots(figsize=(20, 10))
                ax.set_title(f"Contact Resistance ({temp_key})")
                x = common_length
                x_, dx, ux = a.separate_measurement(x)
                y = contact_resistance
                y_, dy, uy = a.separate_measurement(y)
                ax.errorbar(x_, y_, xerr=dx, yerr=dy, fmt='o')
                ax.xaxis.set_major_locator(ticker.FixedLocator((x_)))
                ax.xaxis.set_major_formatter(ticker.FixedFormatter((common_pair_label)))
                plt.setp(ax.get_xticklabels(), rotation=60, horizontalalignment='right', fontsize='x-small')
                ax.set_ylabel(f"$R_{{cont}}$ [${uy:~L}$]")
                plt.tight_layout()
                res_image = os.path.join(res_dir, f"{temp_key}_contact_resistance.png")
                plt.savefig(res_image, dpi=100)


if __name__ == "__main__":
    data_dict = {
        # 'SPC2': {
        #     'nums': np.concatenate([
        #         np.arange(1, 23),
        #         np.arange(24, 31),
        #         np.arange(111, 124),
        #     ])
        # },
        'SPC3': {
            'nums': np.concatenate([
                np.arange(1, 10),
                np.arange(11, 13),
                np.arange(14, 18),
            ])
        },
    }

    dh = a.DataHandler()

    main(data_dict)
