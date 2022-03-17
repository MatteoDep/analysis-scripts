# -*- coding: utf-8 -*-
"""analyse.

Collection of analysis routines.
"""


import os
import numpy as np
from matplotlib import pyplot as plt
import analysis as a


EXPERIMENT = 'light_effect'


def temperature_dependence(names, props, data_dir, bias_window=None):
    """Compute conductance over temperature with and without light."""
    # load data properties
    conductance = []
    temperature = []
    for name in names:
        path = os.path.join(data_dir, name + '.xlsx')
        data = a.load_data(path, order=props[name]['order'])

        conductance.append(
            a.get_conductance(data['voltage'], data['current'], bias_window=bias_window)
        )
        temperature.append(np.mean(data['temperature']))

    temperature = a.qlist_to_array(temperature)
    conductance = a.qlist_to_array(conductance)

    return temperature, conductance


if __name__ == "__main__":
    chip = "SIC1x"
    pair = "P27-P28"
    data_dir = os.path.join('data', 'SIC1x')
    res_dir = os.path.join('results', EXPERIMENT, 'SIC1x')
    os.makedirs(res_dir, exist_ok=True)
    prop_path = os.path.join(data_dir, 'properties.json')

    cp = a.ChipParameters(os.path.join("chips", chip + ".json"))

    segment = cp.pair_to_segment(pair)
    length = cp.get_distance(segment)
    print(f"Analyzing {chip} {pair} of length {length}.")

    # temperature dependence
    bias_window = {
        'lb': [-1.5, 1.5],
        'hb': [22, 24],
    }
    names = {
        'lb': {
            '100%': [
                'SIC1x_725',
                'SIC1x_729',
                'SIC1x_733',
                'SIC1x_737',
                'SIC1x_741',
            ],
            'dark': [
                'SIC1x_726',
                'SIC1x_730',
                'SIC1x_734',
                'SIC1x_738',
                'SIC1x_742',
            ],
        },
        'hb': {
            '100%': [
                'SIC1x_711',
                'SIC1x_715',
                'SIC1x_719',
                'SIC1x_723',
                'SIC1x_727',
                'SIC1x_731',
                'SIC1x_735',
                'SIC1x_739',
            ],
            'dark': [
                'SIC1x_712',
                'SIC1x_716',
                'SIC1x_720',
                'SIC1x_724',
                'SIC1x_728',
                'SIC1x_732',
                'SIC1x_736',
                'SIC1x_740',
            ],
        },
    }

    conductance = {}
    temperature = {}
    for r in names:
        temperature[r] = {}
        conductance[r] = {}
        for ls in names[r]:
            props = a.load_properties(prop_path, names[r][ls])
            temperature[r][ls], conductance[r][ls] = temperature_dependence(
                names[r][ls], props, data_dir, bias_window=bias_window[r]
            )

    fig, ax = plt.subplots()
    high_bias = np.mean(bias_window['hb'])
    labels = ["zero bias", f"high bias ({high_bias}V)"]
    for r, label in zip(names.keys(), labels):
        for ls in names[r]:
            x = 100 / temperature[r][ls]
            y = conductance[r][ls]
            x_, dx = a.separate_measurement(x)
            y_, dy = a.separate_measurement(y)
            ax.errorbar(x_, y_, xerr=dx, yerr=dy, fmt='o', label=f"{ls}, " + label)
    ax.set_title("Light Effect Conductance")
    ax.set_xlabel(f"100/T [${x.units:~L}$]")
    ax.set_ylabel(f"G [${y.units:~L}$]")
    ax.set_xlim(a.get_lim(x_))
    ax.set_yscale('log')
    plt.legend()
    plt.tight_layout()
    res_image = os.path.join(res_dir, "conductance_vs_temperature.png")
    plt.savefig(res_image, dpi=300)

    fig, ax = plt.subplots()
    for r, label in zip(names.keys(), labels):
        x = temperature[r]['dark']
        x_, dx = a.separate_measurement(x)
        y = conductance[r]['100%'] / conductance[r]['dark']
        y_, dy = a.separate_measurement(y)
        ax.errorbar(x_, y_, xerr=dx, yerr=dy, fmt='o', label=label)
    ax.set_title("Light Effect Relative Conductance")
    ax.set_xlabel(f"T [${x.units:~L}$]")
    ax.set_ylabel("G_light/G_dark")
    ax.set_xlim(a.get_lim(x_))
    plt.legend()
    plt.tight_layout()
    res_image = os.path.join(res_dir, "relative.png")
    plt.savefig(res_image, dpi=300)
