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


EXPERIMENT = 'length_characterization'


def length_characterization(names):
    """Compute resistance over length characterization."""
    global dh

    bias_win = [-0.2, 0.2] * ur.V
    length = {
        '2p': [] * ur.um,
        '4p': [] * ur.um,
    }
    resistance = {
        '2p': [] * ur['Mohm'],
        '4p': [] * ur['Mohm'],
    }
    pair = {'2p': [], '4p': []}

    for name in names:
        dh.load(name)
        inj = dh.prop['injection'].lower()
        length[inj] = np.append(length[inj], dh.get_length())
        resistance[inj] = np.append(
            resistance[inj],
            dh.get_resistance(bias_win=bias_win, time_win=[0.25, 0.75], debug=False)
        )
        pair[inj] = np.append(pair[inj], dh.prop['pair'])

    coeffs, reslen_model = a.fit_linear(length['4p'], resistance['4p'])
    resistivity = coeffs[0] * a.NUM_FIBERS * np.pi * a.FIBER_RADIUS**2
    conductivity = (1 / resistivity).to('S / cm')
    print("conductivity:", conductivity)

    for inj in resistance:
        fig, ax = plt.subplots(figsize=(20, 10))
        inj_label = inj.replace('p', '-probe')
        ax.set_title(f"{inj_label} resistance")
        x = length[inj]
        y = resistance[inj]
        pair_label = [f"{p}\n{x[i]}" for i, p in enumerate(pair[inj])]
        x_, dx, ux = a.separate_measurement(x)
        y_, dy, uy = a.separate_measurement(y)
        ax.xaxis.set_major_locator(ticker.FixedLocator((x_)))
        ax.xaxis.set_major_formatter(ticker.FixedFormatter((pair_label)))
        plt.setp(ax.get_xticklabels(), rotation=60, horizontalalignment='right', fontsize='x-small')
        ax.errorbar(x_, y_, xerr=dx, yerr=dy, fmt='o', label=inj_label)
        ax.set_ylabel(f"$R$ [${uy:~L}$]")
        if inj == '4p':
            y1 = reslen_model(x_)
            ax.plot(x_, y1, '-', label=f'{inj} fit')
        plt.tight_layout()
        res_image = os.path.join(res_dir, f"{inj}_resistance.png")
        plt.savefig(res_image)

    # make compatible
    common_length = [] * ur.um
    contact_resistance = [] * ur['Mohm']
    common_pair_label = []
    for i, p in enumerate(pair['2p']):
        for j, p_ in enumerate(pair['4p']):
            if p_ == p:
                common_length = np.append(common_length, length['2p'][i])
                contact_resistance = np.append(contact_resistance, resistance['2p'][i] - resistance['4p'][j])
                common_pair_label = np.append(common_pair_label, f"{p}\n{common_length[-1]}")

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.set_title("Contact Resistance")
    x = common_length
    y = contact_resistance
    x_, dx, ux = a.separate_measurement(x)
    y_, dy, uy = a.separate_measurement(y)
    ax.errorbar(x_, y_, xerr=dx, yerr=dy, fmt='o')
    ax.xaxis.set_major_locator(ticker.FixedLocator((x_)))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter((common_pair_label)))
    plt.setp(ax.get_xticklabels(), rotation=60, horizontalalignment='right', fontsize='x-small')
    ax.set_ylabel(f"$R_{{cont}}$ [${uy:~L}$]")
    plt.tight_layout()
    res_image = os.path.join(res_dir, "contact_resistance.png")
    plt.savefig(res_image)


if __name__ == "__main__":
    # chip = "SJC9"
    # chip = "SKC6"
    # chip = "SLC7"
    # chip = "SKC7"
    # chip = "SIC1x"
    chip = "SPC2"
    # chip = "SPC3"
    data_dir = os.path.join('data', chip)
    res_dir = os.path.join('results', EXPERIMENT, chip)
    os.makedirs(res_dir, exist_ok=True)

    dh = a.DataHandler(data_dir)

    if chip == "SPC2":
        nums = np.concatenate([
            np.arange(1, 23),
            np.arange(24, 31),
        ])
        names = [f"{chip}_{i}" for i in nums]
    elif chip == "SPC3":
        # nums = np.arange(1, 18)
        nums = np.concatenate([
            np.arange(1, 10),
            np.arange(11, 13),
            np.arange(14, 18),
        ])
        names = [f"{chip}_{i}" for i in nums]

    length_characterization(names)
