# -*- coding: utf-8 -*-
"""analyse.

Collection of analysis routines.
"""


import os
import re
from glob import glob
import numpy as np
from matplotlib import pyplot as plt
import analysis as a


def get_low_bias_conductance(voltage, current, max_bias):
    """Calculate low-bias conuctance."""
    # prepare data
    x = a.strip_units(np.abs(voltage))
    condition = x < max_bias
    indices = np.nonzero(condition)
    voltage = voltage[indices]
    current = current[indices]

    # correct offset
    current += np.mean(current)

    # calculate conductance
    conductance = np.mean(current / voltage)
    return conductance


def get_high_bias_conductance(voltage, current, window):
    """Calculate low-bias conuctance."""
    # prepare data
    x = a.strip_units(np.abs(voltage))
    condition = (x > window[0]) * (x < window[1])
    indices = np.nonzero(condition)
    voltage = voltage[indices]
    current = current[indices]

    # fit data
    coeffs, model = a.fit(voltage, current)

    # calculate conductance
    conductance = coeffs[0]
    return conductance


def light_effect(chip, pair,
                 data_dir=None, prop_path=None, res_dir=None,
                 high_bias_window=[23, 24], low_bias_max=0.5,
                 low_high_bias_thresh=15):
    """Compute resistance over length characterization."""
    if data_dir is None:
        data_dir = os.path.join('data', 'light_effect', 'SIC1x')
    # create results dir if it doesn't exist
    if res_dir is None:
        res_dir = os.path.join("results", "light_effect", chip)
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

    regimes = ['lb', 'hb']
    light_statuses = ['100%', 'dark']
    conductance = {ls: {r: [] for r in regimes} for ls in light_statuses}
    temperature = {ls: {r: [] for r in regimes} for ls in light_statuses}
    pattern = os.path.join(data_dir, '*.xlsx')
    for path in np.sort(glob(pattern)):
        name = os.path.splitext(os.path.basename(path))[0]
        if name not in prop:
            continue
        if prop[name]['pair'] != pair:
            continue

        data = a.load_data(path, order=prop[name]['order'])

        for ls_ in light_statuses:
            if prop[name]['comment'].lower().find(ls_) > -1:
                ls = ls_

        if prop[name]['voltage'] < low_high_bias_thresh * a.ur.V:
            # TODO
            conductance[ls]['lb'].append(
                get_low_bias_conductance(data['voltage'], data['current'], low_bias_max)
            )
            temperature[ls]['lb'].append(np.mean(data['temperature']))
            pass
        else:
            conductance[ls]['hb'].append(
                get_high_bias_conductance(data['voltage'], data['current'], high_bias_window)
            )
            temperature[ls]['hb'].append(np.mean(data['temperature']))
    for ls in light_statuses:
        for r in regimes:
            temperature[ls][r] = a.qlist_to_array(temperature[ls][r])
            conductance[ls][r] = a.qlist_to_array(conductance[ls][r]).to('nS')

    fig, ax = plt.subplots()
    r = 'hb'
    for ls in light_statuses:
        x = temperature[ls][r]
        y = conductance[ls][r]
        x_, dx = a.separate_measurement(x)
        y_, dy = a.separate_measurement(y)
        ax.errorbar(x_, y_, xerr=dx, yerr=dy, fmt='o', label=ls)
    ax.set_title("Light Effect")
    ax.set_xlabel(f"T [${x.units:~L}$]")
    ax.set_ylabel(f"G [${y.units:~L}$]")
    ax.set_xlim(a.get_lim(x_))
    ax.set_ylim(a.get_lim(y_))
    ax.set_yscale('log')
    plt.legend()
    plt.tight_layout()
    res_image = os.path.join(res_dir, "conductance_vs_temperature.png")
    plt.savefig(res_image)

    fig, ax = plt.subplots()
    r = 'hb'
    for ls in light_statuses:
        x = temperature[ls][r]
        y = conductance[ls][r]
        x_, dx = a.separate_measurement(x)
        y_, dy = a.separate_measurement(y)
        ax.errorbar(x_, y_, xerr=dx, yerr=dy, fmt='o', label=ls)
    ax.set_title("Light Effect")
    ax.set_xlabel(f"T [${x.units:~L}$]")
    ax.set_ylabel(f"G [${y.units:~L}$]")
    ax.set_xlim(a.get_lim(x_))
    ax.set_ylim(a.get_lim(y_))
    ax.set_yscale('log')
    plt.legend()
    plt.tight_layout()
    res_image = os.path.join(res_dir, "relative.png")
    plt.savefig(res_image)


if __name__ == "__main__":
    chip = "SIC1x"

    light_effect(chip, 'P27-P28')
