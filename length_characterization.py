# -*- coding: utf-8 -*-
"""analyse.

Collection of analysis routines.
"""


import os
import numpy as np
from matplotlib import pyplot as plt
import analysis as a
from analysis import ur


EXPERIMENT = 'length_characterization'


def length_characterization(names):
    """Compute resistance over length characterization."""
    global dh

    bias_window = [-0.2, 0.2] * ur.V
    length = {
        '2p': [] * ur.um,
        '4p': [] * ur.um,
    }
    resistance = {
        '2p': [] * ur['Mohm'],
        '4p': [] * ur['Mohm'],
    }

    for name in names:
        dh.load(name)
        inj = dh.prop['injection'].lower()
        length[inj] = np.append(length[inj], dh.get_length())
        resistance[inj] = np.append(resistance[inj], dh.get_resistance(bias_window=bias_window, only_return=True))

    coeffs, reslen_model = a.fit_linear(length['4p'], resistance['4p'])
    resistivity = (coeffs[0] * a.NUM_FIBERS * np.pi * a.FIBER_RADIUS**2).to('ohm * cm')
    conductivity = (1 / resistivity).to('S / cm')
    print(f"conductivity: {conductivity:~P}")

    for inj in resistance:
        fig, ax = plt.subplots()
        ax.set_title("2-probe and 4-probe resistance")
        x = length[inj]
        y = resistance[inj]
        x_, dx, ux = a.separate_measurement(x)
        y_, dy, uy = a.separate_measurement(y)
        ax.errorbar(x_, y_, xerr=dx, yerr=dy, fmt='o', label=inj.replace('p', '-probe'))
        ax.set_xlabel(f"Segment Length [${ux:~L}$]")
        ax.set_ylabel(f"$R$ [${uy:~L}$]")
        if inj == '4p':
            x1 = np.insert(x_, 0, 0)
            y1 = reslen_model(x1)
            ax.plot(x1, y1, '-', label=f'{inj} fit')
        plt.tight_layout()
        res_image = os.path.join(res_dir, f"{inj}_resistance.png")
        plt.savefig(res_image)

    # make compatible
    cond = (length['2p'] - length['4p']) != 0
    length = length['2p'][cond]
    contact_resistance = resistance['2p'] - resistance['4p']

    fig, ax = plt.subplots()
    x = length
    y = contact_resistance
    x_, dx, ux = a.separate_measurement(x)
    y_, dy, uy = a.separate_measurement(y)
    ax.errorbar(x_, y_, xerr=dx, yerr=dy, fmt='o')
    ax.set_xlabel(f"Segment length [${ux:~L}$]")
    ax.set_ylabel(f"$R_{{contact}}$ [${uy:~L}$]")
    plt.tight_layout()
    res_image = os.path.join(res_dir, "contact_resistance.png")
    plt.savefig(res_image)


if __name__ == "__main__":
    # chip = "SJC9"
    # chip = "SKC6"
    # chip = "SLC7"
    # chip = "SKC7"
    # chip = "SIC1x"
    chip = "SPC3"
    data_dir = os.path.join('data', chip)
    res_dir = os.path.join('results', EXPERIMENT, chip)
    os.makedirs(res_dir, exist_ok=True)

    dh = a.DataHandler(data_dir)

    names = [f"{chip}_{i}" for i in range(11, 35)]
    length_characterization(names)
