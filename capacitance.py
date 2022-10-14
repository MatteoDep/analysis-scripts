# -*- coding: utf-8 -*-
"""temp_characterization.

Analize capacitive effect.
"""


import os
import numpy as np
from matplotlib import pyplot as plt

from analysis import ur, fmt
import analysis as a


EXPERIMENT = os.path.splitext(os.path.basename(__file__))[0]
RES_DIR = os.path.join('results', EXPERIMENT)
os.makedirs(RES_DIR, exist_ok=True)


def capacitance_study(dh, data_dict):

    for chip in data_dict:
        dh.load_chip(chip)
        for pair in data_dict[chip]:
            nums = data_dict[chip][pair]['nums']
            names = np.array([f"{chip}_{i}" for i in nums])

            dh.load(names[0])
            length = dh.get_length()
            temperature = dh.prop['temperature']
            print(f"Chip {chip}, Pair {pair} of length {fmt(length)}, at temperature {fmt(temperature)}.")

            bias_win = data_dict[chip][pair]['bias_win']
            time_wins = [[0.75, 1], [0.25, 0.75]]
            conductance = [] * ur.nS
            frequency = [] * ur.millihertz
            capacitance = [] * ur.pF
            for name in names:
                data = dh.load(name)
                conductance0 = [] * ur.nS
                offset = [] * ur.nA

                # prepare data
                mask_cond = dh.get_mask(bias_win=bias_win, only_return=False)
                total_time = data['time'][-1]
                for time_win in time_wins:
                    mask_time = dh.get_mask(time_win=time_win, only_return=False)
                    mask = mask_time * mask_cond
                    coeffs, model = a.fit_linear(data['bias'][mask], data['current'][mask])
                    conductance0 = np.append(conductance0, coeffs[0])
                    offset = np.append(offset, coeffs[1])
                frequency = np.append(frequency, 1 / total_time)
                capacitance0 = 0.5 * np.abs(np.diff(offset)) * (0.25 * total_time / np.max(data['bias']))
                conductance = np.append(conductance, conductance0[1])
                capacitance = np.append(capacitance, capacitance0)

            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            x = frequency
            for ax, y, title in zip(axs, [conductance, capacitance], ['Conductance', 'Capacitance']):
                x_, dx, ux = a.separate_measurement(x)
                y_, dy, uy = a.separate_measurement(y)
                ax.errorbar(x_, y_, xerr=dx, yerr=dy, marker='o')
                ax.set_title(title)
                ax.set_xlabel(fr"$f$ [${ux:~L}$]")
                sym = 'G' if title == 'Conductance' else 'C'
                ax.set_ylabel(fr"${sym}$ [${uy:~L}$]")
            res_image = os.path.join(RES_DIR, f"{chip}_{pair}_{fmt(temperature, sep='')}_capacitance.png")
            fig.savefig(res_image)
            plt.close()


if __name__ == "__main__":
    dh = a.DataHandler()

    data_dict = {
        'SPC2': {
            'P2-P4': {
                'nums': [186] + list(range(188, 193)),
                'bias_win': [-4, 4] * ur.V,
            },
        },
        'SOC3': {
            'P2-P4': {
                'nums': [44, 45, 46, 49],
                'bias_win': [-1.5, 1.5] * ur.V,
            },
        },
    }

    capacitance_study(dh, data_dict)
