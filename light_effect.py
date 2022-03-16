# -*- coding: utf-8 -*-
"""analyse.

Collection of analysis routines.
"""


import os
import numpy as np
from matplotlib import pyplot as plt
import analysis as a


EXPERIMENT = 'light_effect'


def get_zero_bias_conductance(voltage, current, max_bias, noise_thres=1e-12):
    """Calculate low-bias conuctance."""
    # prepare data
    x = np.abs(a.strip_units(voltage))
    condition = x < max_bias
    indices = np.nonzero(condition)
    voltage = voltage[indices]
    current = current[indices]
    if np.amax(current).to(a.ur.A).magnitude < noise_thres:
        return np.nan * current.units / voltage.units

    # correct offset
    # current -= np.mean(current)

    # calculate conductance
    # conuctance = current / voltage
    # conductance = np.mean(conuctance).plus_minus(np.std(conuctance))

    # calculate conductance
    coeffs, model = a.fit(voltage, current)
    conductance = coeffs[0]

    return conductance


def get_conductance_at_bias(voltage, current, window):
    """Calculate low-bias conuctance."""
    # prepare data
    x = a.strip_units(np.abs(voltage))
    condition = (x > window[0]) * (x < window[1])
    indices = np.nonzero(condition)
    voltage = voltage[indices]
    current = current[indices]

    # calculate conductance
    coeffs, model = a.fit(voltage, current)
    conductance = coeffs[0]

    return conductance


def light_effect(chip, pair,
                 data_dir=None, prop_path=None, res_dir=None,
                 high_bias_window=[22.5, 23.5], low_bias_max=1.5,
                 low_high_bias_thresh=15):
    """Compute resistance over length characterization."""
    if data_dir is None:
        data_dir = os.path.join('data', EXPERIMENT, 'temperature_dependence', chip)
    # create results dir if it doesn't exist
    if res_dir is None:
        res_dir = os.path.join("results", EXPERIMENT, "temperature_dependence", chip)
    os.makedirs(res_dir, exist_ok=True)

    # load data properties
    if prop_path is None:
        prop_path = os.path.join(data_dir, 'properties.json')
    prop = a.load_properties(prop_path)

    # load chip parameters
    chip = a.ChipParameters(os.path.join("chips", chip + ".json"))

    # check pair argument
    if isinstance(pair, str):
        segment = chip.pair_to_segment(pair)
    elif isinstance(pair, list):
        segment = pair
        pair = chip.segment_to_pair(segment)

    # length = chip.get_distance(segment)

    regimes = ['zb', 'hb']
    light_statuses = ['100%', 'dark']
    conductance = {ls: {r: [] for r in regimes} for ls in light_statuses}
    temperature = {ls: {r: [] for r in regimes} for ls in light_statuses}
    for name in prop:
        if prop[name]['pair'] != pair:
            continue

        path = os.path.join(data_dir, name + '.xlsx')
        data = a.load_data(path, order=prop[name]['order'])

        for ls_ in light_statuses:
            if prop[name]['comment'].lower().find(ls_) > -1:
                ls = ls_

        if prop[name]['voltage'] < low_high_bias_thresh * a.ur.V:
            conductance[ls]['zb'].append(
                get_zero_bias_conductance(data['voltage'], data['current'], low_bias_max)
            )
            temperature[ls]['zb'].append(np.mean(data['temperature']))
            pass
        else:
            conductance[ls]['hb'].append(
                get_conductance_at_bias(data['voltage'], data['current'], high_bias_window)
            )
            temperature[ls]['hb'].append(np.mean(data['temperature']))
    for ls in light_statuses:
        for r in regimes:
            temperature[ls][r] = a.qlist_to_array(temperature[ls][r])
            conductance[ls][r] = a.qlist_to_array(conductance[ls][r]).to('nS')

    fig, ax = plt.subplots()
    high_bias = np.mean(high_bias_window)
    labels = ["zero bias", f"high bias ({high_bias}V)"]
    for r, label in zip(regimes, labels):
        for ls in light_statuses:
            x = 100 / temperature[ls][r]
            y = conductance[ls][r]
            x_, dx = a.separate_measurement(x)
            y_, dy = a.separate_measurement(y)
            ax.errorbar(x_, y_, xerr=dx, yerr=dy, fmt='o', label=f"{ls}, " + label)
    # ax.plot(np.arange(10) + 1, np.ones(10)*150/8, label="not pasted")
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
    ref_temp = temperature['dark']['hb']
    x = ref_temp
    x_, dx = a.separate_measurement(x)
    for r, label in zip(regimes, labels):
        y = conductance['100%'][r] / conductance['dark'][r]
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


if __name__ == "__main__":
    chip = "SIC1x"
    pair = "P27-P28"

    light_effect(chip, pair)
