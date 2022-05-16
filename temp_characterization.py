# -*- coding: utf-8 -*-
"""temp_characterization.

Analize temperature dependence.
"""


import os
import numpy as np
from matplotlib import pyplot as plt

from analysis import ur
import analysis as a


EXPERIMENT = 'temp_characterization'


def temp_dependence(names):
    """Main analysis routine.
    names: list
    """
    global dh
    biases = np.arange(0, 10, 2) * ur.V
    delta_bias = 0.3 * ur.V

    temperature = [] * ur.K
    conductance = [[] * ur.S for b in biases]
    for name in names:
        dh.load(name)
        temperature = np.append(temperature, dh.get_temperature())
        for i, bias in enumerate(biases):
            bias_win = [bias - delta_bias, bias + delta_bias]
            if bias + delta_bias < dh.prop['voltage']:
                conductance0 = dh.get_conductance(bias_win=bias_win, time_win=[0.25, 0.75])
            else:
                conductance0 = np.nan
            conductance[i] = np.append(conductance[i], conductance0)

    temp_thresh = 100 * ur.K
    cond = a.is_between(temperature, [temp_thresh, 350 * ur.K])
    x = 100 / temperature
    x_, dx, ux = a.separate_measurement(x)
    fig, ax = plt.subplots(figsize=(12, 9))
    for i, bias in enumerate(biases):
        y = conductance[i]
        y_, dy, uy = a.separate_measurement(y)
        ax.errorbar(x_, y_, xerr=dx, yerr=dy, marker='o', linewidth=0, c=f'C{i}', label=fr'$V_b = {bias}$')
        condtot = cond * (np.isnan(y_) == 0)
        plt.legend()
        if condtot.any():
            coeffs, model = a.fit_exponential(x[condtot], y[condtot], debug=False)
            act_energy = (- coeffs[0] * 100 * ur.k_B).to('meV')
            print("Activation energy ({}): {}".format(bias, act_energy))
            ax.plot(x_[cond], model(x_[cond]), c=f'C{i}', label=fr"$U_A = {act_energy}$")
    ax.set_title("Conductance")
    ax.set_xlabel(fr"$\frac{{1}}{{k_BT}}$ [${ux:~L}$]")
    ax.set_ylabel(fr"$G$ [${uy:~L}$]")
    ax.set_yscale('log')
    res_image = os.path.join(res_dir, "temperature_dep.png")
    fig.savefig(res_image, dpi=300)
    plt.close()


def plot_ivs(names):
    global dh

    for name in names:
        fig, ax = plt.subplots()
        dh.load(name)
        dh.plot(ax)
        ax.set_title(f"IV ({dh.prop['temperature']})")
        res_image = os.path.join(
            res_dir,
            "iv_{0.magnitude}{0.units}_{1}.png".format(
                dh.prop['temperature'],
                name
            )
        )
        fig.savefig(res_image, dpi=300)
        plt.close()


def capacitance_study(names):
    global dh
    bias_win = [-1.5, 1.5] * ur.V
    rel_time_wins = [[0.75, 1], [0.25, 0.75]]
    conductance = [] * ur.S
    frequency = [] * ur.hertz
    temperature = [] * ur.K
    capacitance = [] * ur.pF

    for name in names:
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
    data_dir = os.path.join('data', chip)
    res_dir = os.path.join('results', EXPERIMENT, chip)
    os.makedirs(res_dir, exist_ok=True)

    dh = a.DataHandler(data_dir)

    do = []
    do.append('temp_dependence')
    # do.append('plot_ivs')
    # do.append('capacitance_study')

    if 'temp_dependence' in do:
        if chip == 'SPC2':
            nums = np.arange(45, 74)
            names = [f"{chip}_{i}" for i in nums]
        if chip == 'SOC3':
            names = [
                'SOC3_15',
                'SOC3_16',
                'SOC3_17',
                'SOC3_18',
                'SOC3_19',
                'SOC3_20',
                'SOC3_21',
                'SOC3_22',
                'SOC3_23',
                'SOC3_24',
                'SOC3_25',
                'SOC3_26',
                'SOC3_27',
                'SOC3_28',
                'SOC3_29',
                'SOC3_30',
                'SOC3_31',
                'SOC3_32',
                'SOC3_49',
                'SOC3_50',
                'SOC3_51',
                'SOC3_52',
                'SOC3_53',
                'SOC3_54',
                'SOC3_55',
                'SOC3_56',
                'SOC3_57',
                'SOC3_58',
                'SOC3_59',
                'SOC3_60',
                'SOC3_61',
                'SOC3_62',
                'SOC3_63',
                'SOC3_64',
                'SOC3_65',
                'SOC3_66',
                'SOC3_67',
                'SOC3_68',
                'SOC3_69',
                'SOC3_70',
                'SOC3_71',
                'SOC3_72',
                'SOC3_73',
                'SOC3_74',
                'SOC3_75',
                'SOC3_76',
                'SOC3_77',
                'SOC3_78',
                'SOC3_79',
                'SOC3_80',
                'SOC3_81',
                'SOC3_82',
                'SOC3_83',
                'SOC3_84',
                'SOC3_85',
                'SOC3_86',
                'SOC3_87',
                'SOC3_88',
                'SOC3_89',
                'SOC3_90',
            ]
        temp_dependence(names)

    if 'plot_ivs' in do:
        if chip == 'SPC2':
            nums = np.arange(45, 74)
            names = [f"{chip}_{i}" for i in nums]
        if chip == 'SOC3':
            names = [f'SOC3_{i}' for i in range(90, 91)]
        plot_ivs(names)

    if 'capacitance_study' in do:
        names = [
            'SOC3_44',
            'SOC3_45',
            'SOC3_46',
            # 'SOC3_47',
            'SOC3_49',
        ]
        capacitance_study(names)
