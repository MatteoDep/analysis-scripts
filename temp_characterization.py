# -*- coding: utf-8 -*-
"""temp_characterization.

Analize temperature dependence.
"""


import os
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import warnings

from analysis import ur
import analysis as a


EXPERIMENT = 'temp_characterization'


def get_conductance_vs_temperature():
    global dh, noise_level
    fields = np.concatenate([[0.01], np.arange(0.05, 2, 0.05)]) * ur['V/um']
    delta_field = 0.01 * ur['V/um']
    time_win = [0.25, 0.75]

    temperature = [] * ur.K
    conductance = []
    max_j = len(fields)
    for i, name in enumerate(names):
        dh.load(name)
        temperature0 = dh.get_temperature()
        temperature = np.append(temperature, temperature0)
        if i == 0:
            length = dh.get_length()
            chip_name = dh.chip_name
            pair = dh.prop['pair']
            fields = fields[(fields + delta_field) * length < dh.prop['voltage']]
            print("Chip {}, Pair {} of length {}.".format(chip_name, pair, length))
        print(f"\rProcessing {i+1} of {len(names)}.", end='', flush=True)
        full_conductance = dh.get_conductance(time_win=time_win)
        full_conductance_masked = dh.get_conductance(time_win=time_win, noise_level=noise_level)
        for j, field in enumerate(fields):
            if j < max_j:
                bias_win = [
                    (field - delta_field) * length,
                    (field + delta_field) * length
                ]
                cond = a.is_between(dh.data['voltage'], bias_win)
                part_conductance = full_conductance_masked[cond]
                if dh.prop['voltage'] < bias_win[1]:
                    max_j = j
                    conductance0 = np.nan
                else:
                    # Note that since using the time window [0.25, 0.75] 50% will always be masked
                    nans_ratio = np.sum(a.isnan(part_conductance)) / np.sum(cond)
                    if nans_ratio == 1:
                        conductance0 = np.nan
                    elif nans_ratio < 0.51:
                        conductance0 = a.average(part_conductance)
                    else:
                        part_conductance = full_conductance[cond]
                        conductance0 = a.average(part_conductance)
            else:
                conductance0 = np.nan
            if i == 0:
                conductance.append([] * ur.S)
            conductance[j] = np.append(conductance[j], conductance0)
        if plot_ivs:
            plot_iv(length)
    print()
    return temperature, conductance, fields, chip_name, length


def temp_dependence(names, lowtemp_lowbias_name=None, lowtemp_highbias_name=None):
    """Main analysis routine.
    names: list
    """
    global dh, noise_level

    temperature, conductance, fields, chip_name, length = get_conductance_vs_temperature()

    # sort with increasing temperature
    indices = np.argsort(temperature)
    temperature = temperature[indices]
    for i, _ in enumerate(fields):
        conductance[i] = conductance[i][indices]
    names = np.array(names)[indices]

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
        ax.errorbar(x_, y_, xerr=dx, yerr=dy, fmt='--o', zorder=i, c=cols[i], label=fr'$V_b = {field}$')
        if i == 0:
            coeffs, model = a.fit_exponential(x[cond], y[cond])
            if coeffs is not None:
                act_energy = (- coeffs[0] * 100 * ur.k_B).to('meV')
                print("Activation energy:", act_energy)
                x1 = 100 / temp_win.magnitude
                ax.plot(x1, model(x1), c='r', zorder=len(fields), label=fr"$U_A = {act_energy}$")
    ax.legend()
    ax.set_title(f"Conductance ({chip_name} {pair})")
    ax.set_xlabel(fr"$\frac{{100}}{{T}}$ [${ux:~L}$]")
    ax.set_ylabel(fr"$G$ [${uy:~L}$]")
    ax.set_yscale('log')
    res_image = os.path.join(res_dir, f"{chip_name}_{pair}_temperature_dep.png")
    fig.savefig(res_image, dpi=300)
    plt.close()

    # Power law
    # temperature dependence at low bias
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 9))
    fig.suptitle(f'Power Law ({chip_name} {pair})')
    temp_win = [0 * ur.K, 100 * ur.K]
    cond = a.is_between(temperature, temp_win)
    x = temperature[cond]
    y = conductance[0][cond]
    x, y = a.strip_nan(x, y)
    coeffs, model = a.fit_powerlaw(x, y, check_nan=False)
    if coeffs is not None:
        alpha = coeffs[0]
        print("alpha:", alpha)
        x_, dx, ux = a.separate_measurement(x)
        y_, dy, uy = a.separate_measurement(y)
        ax1.errorbar(x_, y_, xerr=dx, yerr=dy, fmt='o', zorder=1, label='data')
        ax1.plot(x_, model(x_), zorder=2, label=fr'$\alpha = {alpha}$')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel(fr"$T$ [${ux:~L}$]")
    ax1.set_ylabel(fr"$G$ [${uy:~L}$]")
    ax1.legend()

    # bias dependence at low temperature
    if lowtemp_lowbias_name is None:
        lowtemp_lowbias_name = names[0]
    lowtemp_names = [lowtemp_lowbias_name]
    if lowtemp_highbias_name is not None:
        lowtemp_names.append(lowtemp_highbias_name)
    current_threshold = noise_level
    for i, name in enumerate(lowtemp_names):
        dh.load(name)
        temperature = dh.get_temperature()
        x = dh.data['voltage']
        y = dh.get_conductance('all', time_win=[0.25, 0.75], noise_level=current_threshold)
        x, y = a.strip_nan(x, y)
        coeffs, model = a.fit_powerlaw(x, y)
        if coeffs is not None:
            beta = coeffs[0]
            print(f"beta_{i+1}:", beta)
            x_, dx, ux = a.separate_measurement(x)
            y_, dy, uy = a.separate_measurement(y)
            ax2.errorbar(x_, y_, xerr=dx, yerr=dy, fmt='o', zorder=1)
            ax2.plot(x_, model(x_), zorder=2, label=fr'$\beta = {beta}$')
        current_threshold = 50 * ur.pA
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel(fr"$V_b$ [${ux:~L}$]")
    ax2.set_ylabel(fr"$G$ [${uy:~L}$]")
    ax2.legend()
    res_image = os.path.join(res_dir, f"{chip_name}_{pair}_power_law.png")
    fig.savefig(res_image, dpi=300)
    plt.close()


def plot_iv(length=None):
    global dh
    field_win = [-0.05, 0.05] * ur['V/um']
    if length is None:
        length = dh.get_length()
    voltage_win = field_win * length
    fig, ax1 = plt.subplots(figsize=(12, 9))
    dh.plot(ax1)
    ax2 = inset_axes(ax1, width='35%', height='35%', loc=4, borderpad=4)
    dh.plot(ax2, x_win=voltage_win)
    fig.suptitle(f"{dh.chip_name} {dh.prop['pair']} at {dh.prop['temperature']}")
    res_image = os.path.join(
        res_dir,
        "{1}_{2}_iv_{0.magnitude}{0.units}.png".format(
            dh.prop['temperature'],
            dh.chip_name,
            dh.prop['pair']
        )
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=UserWarning)
        fig.savefig(res_image, dpi=300)
    plt.close()


if __name__ == "__main__":
    # chip = "SOC3"
    chip = "SPC2"
    pair = "P2-P4"
    # pair = "P17-P18"
    # pair = "P15-P17"
    # pair = "P7-P8"
    # pair = "P13-P14"
    # pair = "P16-P17"
    # pair = "P15-P16"
    # pair = "P1-P4"
    data_dir = os.path.join('data', chip)
    res_dir = os.path.join('results', EXPERIMENT, chip)
    os.makedirs(res_dir, exist_ok=True)

    dh = a.DataHandler(data_dir)

    plot_ivs = False
    noise_level = 0.5 * ur.pA
    lowtemp_lowbias_name = None
    lowtemp_highbias_name = None
    if chip == 'SPC2':
        if pair == 'P2-P4':
            nums = np.arange(45, 74)
            lowtemp_highbias_name = f"{chip}_276"
        elif pair == 'P17-P18':
            nums = np.concatenate([
                np.arange(78, 102),
                np.arange(103, 109)
            ])
            lowtemp_highbias_name = f"{chip}_279"
        elif pair == 'P15-P17':
            nums = np.arange(124, 155)
            lowtemp_highbias_name = f"{chip}_280"
        elif pair == 'P7-P8':
            nums = np.arange(156, 186)
            lowtemp_highbias_name = f"{chip}_278"
        elif pair == 'P13-P14':
            nums = np.concatenate([
                np.arange(195, 201),
                np.arange(202, 203),
                np.arange(204, 208),
                np.arange(209, 226)
            ])
            lowtemp_highbias_name = f"{chip}_277"
        elif pair == 'P16-P17':
            nums = np.arange(227, 258)
            lowtemp_highbias_name = f"{chip}_281"
        elif pair == 'P15-P16':
            nums = np.concatenate([
                np.arange(283, 301),
                [301, 303, 305, 307, 309, 311, 313, 316, 318, 320, 328, 330, 332],
            ])
            lowtemp_highbias_name = f"{chip}_282"
        elif pair == 'P1-P4':
            nums = np.arange(335, 351)
            lowtemp_highbias_name = f"{chip}_234"
    if chip == 'SOC3':
        if pair == 'P2-P4':
            nums = np.arange(49, 91)
            noise_level = 10 * ur.pA
    names = [f"{chip}_{i}" for i in nums]

    temp_dependence(names, lowtemp_lowbias_name=lowtemp_lowbias_name, lowtemp_highbias_name=lowtemp_highbias_name)
