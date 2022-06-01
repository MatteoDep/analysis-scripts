# -*- coding: utf-8 -*-
"""temp_characterization.

Analize temperature dependence.
"""


import os
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from analysis import ur
import analysis as a


EXPERIMENT = 'temp_characterization'


def capacitance_study(names, bias_win=[-1.5, 1.5]*ur.V):
    global dh
    rel_time_wins = [[0.75, 1], [0.25, 0.75]]
    conductance = [] * ur.S
    frequency = [] * ur.hertz
    temperature = [] * ur.K
    capacitance = [] * ur.pF

    for name in names:
        print(name)
        data = dh.load(name)
        temperature = np.append(temperature, dh.get_temperature())
        conductance0 = [] * ur.S
        offset = [] * ur.A

        # prepare data
        bias_cond = a.is_between(data['voltage'], bias_win)
        total_time = data['time'][-1]
        for rel_time_win in rel_time_wins:
            time_win = [x * total_time for x in rel_time_win]
            time_cond = a.is_between(data['time'], time_win)
            cond = bias_cond * time_cond

            coeffs, model = a.fit_linear(data['voltage'][cond], data['current'][cond])
            conductance0 = np.append(conductance0, coeffs[0])
            offset = np.append(offset, coeffs[1])
        frequency = np.append(frequency, 1 / total_time)
        capacitance0 = 0.5 * np.abs(np.diff(offset)) * (0.25 * total_time / np.max(data['voltage']))
        conductance = np.append(conductance, conductance0[1])
        capacitance = np.append(capacitance, capacitance0)
        print("capacitance:", capacitance0.to('pF'))
        print("conductance:", conductance0.to('pS'))
    temperature = np.mean(temperature)

    fig, axs = plt.subplots(1, 2)
    x = frequency
    for ax, y, title in zip(axs, [conductance, capacitance], ['Conductance', 'Capacitance']):
        x_, dx, ux = a.separate_measurement(x)
        y_, dy, uy = a.separate_measurement(y)
        ax.errorbar(x_, y_, xerr=dx, yerr=dy, marker='o')
        ax.plot(x_, model(x_))
        ax.set_title(title)
        ax.set_xlabel(fr"$f$ [${ux:~L}$]")
        sym = 'G' if title == 'Conductance' else 'C'
        ax.set_ylabel(fr"${sym}$ [${uy:~L}$]")
        ax.set_yscale('log')
    res_image = os.path.join(res_dir, "capacitance_study.png")
    fig.savefig(res_image, dpi=300)
    plt.close()


if __name__ == "__main__":
    # chip = "SOC3"
    chip = "SPC2"
    pair = "P2-P4"
    prefix = f"{chip}_{pair}_"
    data_dir = os.path.join('data', chip)
    res_dir = os.path.join('results', EXPERIMENT, chip)
    os.makedirs(res_dir, exist_ok=True)

    dh = a.DataHandler(data_dir)

    if chip == 'SPC2':
        if pair == 'P2-P4':
            # nums = np.arange(186, 193)
            nums = [186] + list(range(188, 193))
            bias_win = [-4, 4] * ur.V
    if chip == 'SOC3':
        if pair == 'P2-P4':
            nums = [44, 45, 46, 49]
            bias_win = [-1.5, 1.5] * ur.V
    names = [f"{chip}_{i}" for i in nums]

    capacitance_study(names, bias_win=bias_win)
