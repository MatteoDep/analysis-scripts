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


def temp_dependence(names, fields=np.arange(0, 2, 0.05) * ur['V/um'], prefix=""):
    """Main analysis routine.
    names: list
    """
    global dh
    if not hasattr(fields, 'units'):
        fields *= ur['V/um']
    delta_field = 0.05 * ur['V/um']
    temp_field_thresh = 1 * ur['V*K/um']

    temperature = [] * ur.K
    conductance = [[] * ur.S for f in fields]
    max_i = len(fields)
    for j, name in enumerate(names):
        dh.load(name)
        temperature0 = dh.get_temperature()
        temperature = np.append(temperature, temperature0)
        length0 = dh.get_length()
        if j == 0:
            print("Pair {} of length {}.".format(dh.prop['pair'], length0.to('um')))
            cond = dh.data['voltage'] > temp_field_thresh * length0 / temperature0
            conductance_lowtemp = dh.data['current'][cond] / dh.data['voltage'][cond]
            fields_lowtemp = dh.data['voltage'][cond] / length0
        for i, field in enumerate(fields):
            bias_win = [] * ur.V
            bias_win = np.append(bias_win, (field - delta_field) * length0)
            bias_win = np.append(bias_win, (field + delta_field) * length0)
            if i < max_i and max(field, delta_field) * temperature0 > temp_field_thresh:
                conductance0 = dh.get_conductance(bias_win=bias_win, time_win=[0.25, 0.75])
                if conductance0 is None:
                    max_i = i
                    if j == 0:
                        fields = fields[:max_i]
            else:
                conductance0 = np.nan
            conductance[i] = np.append(conductance[i], conductance0)

    # sort with increasing temperature
    indices = np.argsort(temperature)
    temperature = temperature[indices]
    for i, _ in enumerate(fields):
        conductance[i] = conductance[i][indices]

    # plot temperature dependence
    temp_win = [101, 350] * ur.K
    cond = a.is_between(temperature, temp_win)
    x = 100 / temperature
    x_, dx, ux = a.separate_measurement(x)
    fig, ax = plt.subplots(figsize=(12, 9))
    cols = plt.cm.viridis(fields.magnitude / np.max(fields.magnitude))
    for i, field in enumerate(fields):
        y = conductance[i]
        y_, dy, uy = a.separate_measurement(y)
        nancond = (np.isnan(y_) == 0)
        if nancond.any():
            ax.errorbar(x_, y_, xerr=dx, yerr=dy, marker='o', c=cols[i], linewidth=0, label=fr'$V_b = {field}$')
            cond *= nancond
            if cond.any() and i == 0:
                coeffs, model = a.fit_exponential(x[cond], y[cond])
                act_energy = (- coeffs[0] * 100 * ur.k_B).to('meV')
                print("Activation energy:", act_energy)
                x1 = 100 / temp_win.magnitude
                ax.plot(x1, model(x1), c='r', label=fr"$U_A = {act_energy}$")
    ax.legend()
    ax.set_title("Conductance")
    ax.set_xlabel(fr"$\frac{{100}}{{T}}$ [${ux:~L}$]")
    ax.set_ylabel(fr"$G$ [${uy:~L}$]")
    ax.set_yscale('log')
    res_image = os.path.join(res_dir, f"{prefix}temperature_dep.png")
    fig.savefig(res_image, dpi=300)
    plt.close()

    # Power law
    temp_win = [20, 100] * ur.K
    cond = a.is_between(temperature, temp_win)
    x = temperature
    y = conductance[0]
    coeffs, model = a.fit_powerlaw(x[cond], y[cond])
    alpha = coeffs[0]
    print("alpha:", alpha)
    x = fields_lowtemp
    y = conductance_lowtemp
    coeffs, model = a.fit_powerlaw(x, y)
    beta = coeffs[0]
    print("beta:", beta)


def plot_ivs(names):
    global dh

    for name in names:
        dh.load(name)
        fig, ax1 = plt.subplots(figsize=(15, 10))
        dh.plot(ax1)
        ax2 = inset_axes(ax1, width='35%', height='35%', loc=4, borderpad=4)
        dh.plot(ax2, x_win=[-2, 2]*ur.V)
        fig.suptitle(f"IV ({dh.prop['temperature']})")
        res_image = os.path.join(
            res_dir,
            "{1}_{2}_iv_{0.magnitude}{0.units}.png".format(
                dh.prop['temperature'],
                dh.chip_name,
                dh.prop['pair']
            )
        )
        fig.savefig(res_image, dpi=300)
        plt.close()


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
    # pair = "P2-P4"
    # pair = "P17-P18"
    pair = "P15-P17"
    # pair = "P7-P8"
    # pair = "P13-P14"
    prefix = f"{chip}_{pair}_"
    data_dir = os.path.join('data', chip)
    res_dir = os.path.join('results', EXPERIMENT, chip)
    os.makedirs(res_dir, exist_ok=True)

    dh = a.DataHandler(data_dir)

    biases = np.arange(0, 14, 2)
    if chip == 'SPC2':
        if pair == 'P2-P4':
            nums = np.arange(45, 74)
        elif pair == 'P17-P18':
            nums = np.arange(78, 108)
        elif pair == 'P15-P17':
            nums = np.arange(124, 155)
        elif pair == 'P7-P8':
            nums = np.arange(156, 186)
            biases = np.arange(0, 4)
        elif pair == 'P13-P14':
            nums = np.concatenate([np.arange(195, 201),
                                   np.arange(202, 203),
                                   np.arange(204, 208),
                                   np.arange(209, 226)])
            biases = np.arange(0, 5)
    if chip == 'SOC3':
        nums = np.arange(49, 91)
    names = [f"{chip}_{i}" for i in nums]

    temp_dependence(names, prefix=prefix)

    # plot_ivs(names)

    # if chip == 'SPC2':
    #     if pair == 'P2-P4':
    #         # nums = np.arange(186, 193)
    #         nums = [186] + list(range(188, 193))
    #         bias_win = [-4, 4] * ur.V
    # if chip == 'SOC3':
    #     if pair == 'P2-P4':
    #         nums = [44, 45, 46, 49]
    #         bias_win = [-1.5, 1.5] * ur.V
    # names = [f"{chip}_{i}" for i in nums]

    # capacitance_study(names, bias_win=bias_win)
