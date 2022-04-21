# -*- coding: utf-8 -*-
"""temp_characterization.

Analize temperature dependence.
"""


import os
import numpy as np
from matplotlib import pyplot as plt
import analysis as a


EXPERIMENT = 'temp_characterization'


def get_temperature_dependence(names, props, **get_conductance_kwargs):
    """Compute conductance over temperature with and without light."""
    global data_dir
    # load data properties
    conductance = [] * a.ur.S
    temperature = [] * a.ur.K
    for name in names:
        path = os.path.join(data_dir, name + '.xlsx')
        data = a.load_data(path, order=props[name]['order'])

        conductance = np.append(
            conductance,
            a.get_conductance(data, **get_conductance_kwargs)
        )
        temperature = np.append(
            temperature,
            a.get_mean_std(data['temperature'])
        )
    return temperature, conductance


def temp_dependence(names, bias_window=[-1.5, 1.5]*a.ur.V):
    """Main analysis routine.
    names: list
    """
    global data_dir, res_dir, props
    if not hasattr(bias_window, 'units'):
        bias_window *= a.DEFAULT_UNITS['voltage']

    # compute temperature and conductance
    temperature, conductance = get_temperature_dependence(names, props, bias_window=bias_window)
    y = conductance
    x = (1 / (a.ur.k_B * temperature)).to('meV^-1')

    # fit
    indices = [0] + list(range(6, len(x)))
    coeffs, model = a.fit_exponential(x[indices], y[indices], ignore_err=True)
    act_energy = - coeffs[0]
    print("act_energy:", act_energy)

    fig, ax = plt.subplots()
    x_, dx, ux = a.separate_measurement(x)
    y_, dy, uy = a.separate_measurement(conductance)
    ax.errorbar(x_, y_, xerr=dx, yerr=dy, marker='o')
    ax.plot(x_, model(x_))
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
            "iv_{0.magnitude}{0.units}".format(
                props[name]['temperature'],
            )
        )
        fig.savefig(res_image, dpi=300)
        plt.close()


if __name__ == "__main__":
    chip = "SOC3"
    pair = "P27-P28"
    data_dir = os.path.join('data', chip)
    res_dir = os.path.join('results', EXPERIMENT, chip)
    prop_path = os.path.join(data_dir, 'properties.csv')

    os.makedirs(res_dir, exist_ok=True)
    cp = a.ChipParameters(os.path.join("chips", chip + ".json"))
    segment = cp.pair_to_segment(pair)
    length = cp.get_distance(segment)
    print(f"Analyzing {chip} {pair} of length {a.nominal_values(length).to_compact()}.")

    # load properties
    props = a.load_properties(prop_path)

    do = []
    do.append('temp_dependence')
    # do.append('plot_ivs')

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

    if 'temp_dependence' in do:
        temp_dependence(names)

    if 'plot_ivs' in do:
        plot_ivs(names)
