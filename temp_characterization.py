# -*- coding: utf-8 -*-
"""temp_characterization.

Analize temperature dependence.
"""


import os
import numpy as np
from matplotlib import pyplot as plt
import analysis as a


EXPERIMENT = 'temp_characterization'


def temp_dependence(names):
    """Main analysis routine.
    names: list
    """
    global data_dir, res_dir, props

    # compute temperature and conductance
    conductance = [] * a.ur.S
    temperature = [] * a.ur.K
    for name in names:
        path = os.path.join(data_dir, name + '.xlsx')
        data = a.load_data(path, order=props[name]['order'])

        temperature0 = a.get_mean_std(data['temperature'])
        if temperature0 > 40 * a.ur.K:
            bias_window = [-0.5, 0.5] * a.ur.V
        else:
            bias_window = [-1.5, 1.5] * a.ur.V
        conductance0 = a.get_conductance(data, bias_window=bias_window, only_return=True)
        conductance = np.append(conductance, conductance0)
        temperature = np.append(temperature, temperature0)

    # fit
    temp_thresh = 1 / (a.ur.k_B * 0.2 * a.ur['meV^-1'])
    cond = a.is_between(temperature, [temp_thresh, 350 * a.ur.K])
    y = conductance
    x = (1 / (a.ur.k_B * temperature)).to('meV^-1')
    coeffs, model = a.fit_exponential(x[cond], y[cond], ignore_err=True, debug=True)
    act_energy = - coeffs[0]
    print("act_energy:", act_energy)

    fig, ax = plt.subplots()
    x_, dx, ux = a.separate_measurement(x)
    y_, dy, uy = a.separate_measurement(conductance)
    ax.errorbar(x_, y_, xerr=dx, yerr=dy, marker='o', linewidth=0)
    ax.plot(x_[cond], model(x_[cond]))
    ax.set_title("Conductance")
    ax.set_xlabel(fr"$\frac{{1}}{{k_BT}}$ [${ux:~L}$]")
    ax.set_ylabel(fr"$G$ [${uy:~L}$]")
    ax.set_yscale('log')
    res_image = os.path.join(res_dir, "temperature_dep.png")
    fig.savefig(res_image, dpi=300)
    plt.close()


def plot_ivs(names, correct_offset=True, voltage_window=[-24, 24]*a.ur.V):
    global data_dir, res_dir, props

    for name in names:
        fig, ax = plt.subplots()
        path = os.path.join(data_dir, name + '.xlsx')
        data = a.load_data(path, order=props[name]['order'])
        cond = a.is_between(data['voltage'], voltage_window)
        x = data['voltage'][cond]
        y = data['current'][cond]
        if correct_offset:
            y -= np.mean(y)
        x, dx, ux = a.separate_measurement(x)
        y, dy, uy = a.separate_measurement(y)
        ax.scatter(x, y)
        ax.set_title(f"IV ({props[name]['temperature']})")
        ax.set_xlabel(fr"$V$ [${ux:~L}$]")
        ax.set_ylabel(fr"$I$ [${uy:~L}$]")
        res_image = os.path.join(
            res_dir,
            "iv_{0.magnitude}{0.units}_{1}.png".format(
                props[name]['temperature'],
                name
            )
        )
        fig.savefig(res_image, dpi=300)
        plt.close()


def capacitance_study(names):
    bias_window = [-1.5, 1.5] * a.ur.V
    rel_time_windows = [[0.75, 1], [0.25, 0.75]]
    conductance = [] * a.ur.S
    frequency = [] * a.ur.hertz
    temperature = [] * a.ur.K
    capacitance = [] * a.ur.pF

    for name in names:
        print(name)
        path = os.path.join(data_dir, name + '.xlsx')
        data = a.load_data(path, order=props[name]['order'])
        temperature = np.append(
            temperature,
            a.get_mean_std(data['temperature'])
        )
        conductance0 = [] * a.ur.S
        offset = [] * a.ur.A

        # prepare data
        bias_cond = a.is_between(data['voltage'], bias_window)
        total_time = data['time'][-1]
        for rel_time_window in rel_time_windows:
            time_window = [x * total_time for x in rel_time_window]
            time_cond = a.is_between(data['time'], time_window)
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
    chip = "SOC3"
    pair = "P2-P4"
    data_dir = os.path.join('data', chip)
    res_dir = os.path.join('results', EXPERIMENT, chip)

    os.makedirs(res_dir, exist_ok=True)
    cp = a.ChipParameters(os.path.join("chips", chip + ".json"))
    length = cp.get_distance(pair)
    print(f"Analyzing {chip} {pair} of length {a.nominal_values(length).to_compact()}.")

    # do = []
    # do.append('temp_dependence')
    # do.append('plot_ivs')
    # do.append('capacitance_study')

    dh = a.DataHandler(data_dir)
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
    ]
    for name in names:
        dh.load_data(name)
        dh.get_conductance

    # if 'temp_dependence' in do:
    #     names = [
    #         'SOC3_15',
    #         'SOC3_16',
    #         'SOC3_17',
    #         'SOC3_18',
    #         'SOC3_19',
    #         'SOC3_20',
    #         'SOC3_21',
    #         'SOC3_22',
    #         'SOC3_23',
    #         'SOC3_24',
    #         'SOC3_25',
    #         'SOC3_26',
    #         'SOC3_27',
    #         'SOC3_28',
    #         'SOC3_29',
    #         'SOC3_30',
    #         'SOC3_31',
    #         'SOC3_32',
    #         'SOC3_49',
    #         'SOC3_50',
    #         'SOC3_51',
    #         'SOC3_52',
    #         'SOC3_53',
    #         'SOC3_54',
    #         'SOC3_55',
    #         'SOC3_56',
    #         'SOC3_57',
    #         'SOC3_58',
    #         'SOC3_59',
    #         'SOC3_60',
    #         'SOC3_61',
    #         'SOC3_62',
    #         'SOC3_63',
    #         'SOC3_64',
    #         'SOC3_65',
    #         'SOC3_66',
    #         'SOC3_67',
    #         'SOC3_68',
    #         'SOC3_69',
    #         'SOC3_70',
    #         'SOC3_71',
    #         'SOC3_72',
    #         'SOC3_73',
    #         'SOC3_74',
    #         'SOC3_75',
    #         'SOC3_76',
    #         'SOC3_77',
    #         'SOC3_78',
    #         'SOC3_79',
    #         'SOC3_80',
    #         'SOC3_81',
    #         'SOC3_82',
    #         'SOC3_83',
    #         'SOC3_84',
    #         'SOC3_85',
    #         'SOC3_86',
    #         'SOC3_87',
    #         'SOC3_88',
    #         'SOC3_89',
    #         'SOC3_90',
    #     ]
    #     temp_dependence(names)

    # if 'plot_ivs' in do:
    #     names = [f'SOC3_{i}' for i in range(90, 91)]
    #     plot_ivs(names)

    # if 'capacitance_study' in do:
    #     names = [
    #         'SOC3_44',
    #         'SOC3_45',
    #         'SOC3_46',
    #         # 'SOC3_47',
    #         'SOC3_49',
    #     ]
    #     capacitance_study(names)
