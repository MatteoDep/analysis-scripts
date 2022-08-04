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


def get_length_dependence(names, field, delta_field):
    global dh, res_dir, plot_ivs
    time_win = [0.25, 0.75]
    method = 'fit' if field == 0*ur['V/um'] else 'average'

    injs = ['2p', '4p']
    length = {inj: [] * ur.um for inj in injs}
    resistance = {inj: [] * ur['Mohm'] for inj in injs}
    pair = {'2p': [], '4p': []}
    for i, name in enumerate(names):
        dh.load(name)
        inj = dh.prop['injection'].lower()
        length[inj] = np.append(length[inj], dh.get_length())
        pair[inj] = np.append(pair[inj], dh.prop['pair'])
        field_win = [field - delta_field, field + delta_field]
        resistance0 = dh.get_resistance(method=method, time_win=time_win, field_win=field_win)
        resistance[inj] = np.append(resistance[inj], resistance0)
    return length, resistance, pair


def main(data_dict, noise_level=0.5*ur.pA, pair_on_axis=False):
    """Compute resistance over length characterization."""
    global dh, res_dir

    dh = a.DataHandler()
    for chip in data_dict:
        dh.load_chip(chip)
        res_dir = os.path.join('results', EXPERIMENT, chip)
        os.makedirs(res_dir, exist_ok=True)

        nums = data_dict[chip]['nums']
        names = np.array([f"{chip}_{i}" for i in nums])

        print("Chip {}.".format(chip))
        for i, name in enumerate(names):
            print(f"\rLoading {i+1} of {len(names)}.", end='', flush=True)
            dh.load(name)
        print()

        names_dict = {}
        for name in names:
            key = fmt(dh.props[name]['temperature'], sep='')
            if key not in names_dict:
                names_dict[key] = []
            names_dict[key].append(name)

        for temp_key in names_dict:
            print(f'Temperature {temp_key}:')
            names = names_dict[temp_key]

            if ur.Quantity(temp_key) < 50*ur.K:
                field = 0.2*ur['V/um']
            else:
                field = 0*ur['V/um']
            delta_field = 0.01*ur['V/um']
            length, resistance, pair = get_length_dependence(names, field, delta_field)

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
                fig, ax = plt.subplots(figsize=(20, 10))
                inj_label = inj.replace('p', '-probe')
                ax.set_title(f"{inj_label} resistance ({temp_key})")
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
                else:
                    ax.set_xlabel("Length" + ulbl(ux))
                ax.set_ylabel("$R$" + ulbl(uy))
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
                if pair_on_axis:
                    ax.xaxis.set_major_locator(ticker.FixedLocator((x_)))
                    ax.xaxis.set_major_formatter(ticker.FixedFormatter((pair_label)))
                    plt.setp(ax.get_xticklabels(), rotation=60, horizontalalignment='right', fontsize='x-small')
                else:
                    ax.set_xlabel("Length" + ulbl(ux))
                ax.set_ylabel("$R_{cont}$" + ulbl(uy))
                res_image = os.path.join(res_dir, f"{temp_key}_contact_resistance.png")
                plt.savefig(res_image, dpi=100)


if __name__ == "__main__":
    data_dict = {
        'SPC2': {
            'nums': np.concatenate([
                np.arange(1, 23),
                np.arange(24, 31),
                np.arange(111, 124),
            ])
        },
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
