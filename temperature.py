# -*- coding: utf-8 -*-
"""temperature.

Analize temperature dependence.
"""


import os
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import warnings

from analysis import ur
import analysis as a


EXPERIMENT = os.path.splitext(os.path.basename(__file__))[0]


def get_temperature_dependence(names, fields, delta_field, noise_level=0.5*ur.pA):
    global dh, res_dir, plot_ivs
    time_win = [0.25, 0.75]

    temperature = [] * ur.K
    conductance = [[] * ur.S for f in fields]
    max_j = len(fields)
    for i, name in enumerate(names):
        print(f"\rProcessing {i+1} of {len(names)}.", end='', flush=True)
        dh.load(name)
        temperature0 = dh.get_temperature()
        temperature = np.append(temperature, temperature0)
        full_conductance = dh.get_conductance(time_win=time_win)
        full_conductance_masked = dh.get_conductance(time_win=time_win, noise_level=noise_level)
        for j, field in enumerate(fields):
            if j < max_j:
                bias_win = [
                    (field - delta_field) * dh.get_length(),
                    (field + delta_field) * dh.get_length()
                ]
                cond = a.is_between(dh.data['bias'], bias_win)
                part_conductance = full_conductance_masked[cond]
                if dh.prop['bias'] < bias_win[1]:
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
            conductance[j] = np.append(conductance[j], conductance0)
        if plot_ivs:
            plot_iv()
    print()
    return temperature, conductance


def main(data_dict, noise_level=0.5*ur.pA):
    """Main analysis routine.
    names: list
    """
    global dh, res_dir
    fields = np.concatenate([[0.01], np.arange(0.05, 2, 0.05)]) * ur['V/um']
    delta_field = 0.01 * ur['V/um']
    dh = a.DataHandler()
    for chip in data_dict:
        dh.load_chip(chip)
        res_dir = os.path.join('results', EXPERIMENT, chip)
        os.makedirs(res_dir, exist_ok=True)
        for pair in data_dict[chip]:
            nums = data_dict[chip][pair]['nums']
            names = np.array([f"{chip}_{i}" for i in nums])
            names = names[np.argsort(ur.Quantity.from_list([dh.props[name]['temperature'] for name in names]))]
            if 'lowtemp_lowbias_num' in data_dict[chip][pair]:
                lowtemp_lowbias_name = f"{chip}_{data_dict[chip][pair]['lowtemp_lowbias_num']}"
            else:
                lowtemp_lowbias_name = names[0]
            if 'lowtemp_highbias_num' in data_dict[chip][pair]:
                lowtemp_highbias_name = f"{chip}_{data_dict[chip][pair]['lowtemp_highbias_num']}"
            else:
                lowtemp_highbias_name = None

            fields = fields[(fields + delta_field) * dh.get_length() < dh.props[names[0]]['bias']]
            print("Chip {}, Pair {} of length {}.".format(chip, pair, dh.cps[chip].get_distance(pair)))

            if 'noise_level' in data_dict[chip][pair]:
                nl = data_dict[chip][pair]['noise_level']
            else:
                nl = noise_level

            temperature, conductance = get_temperature_dependence(names, fields, delta_field, noise_level=nl)

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
                ax.errorbar(x_, y_, xerr=dx, yerr=dy, fmt='--o', zorder=i, c=cols[i], label=fr'$E_b = {field}$')
                if i == 0:
                    coeffs, model = a.fit_exponential(x[cond], y[cond])
                    if coeffs is not None:
                        act_energy = (- coeffs[0] * 100 * ur.k_B).to('meV')
                        print("Activation energy:", act_energy)
                        x1 = 100 / temp_win.magnitude
                        ax.plot(x1, model(x1), c='r', zorder=len(fields), label=fr"$U_A = {act_energy}$")
            ax.legend()
            ax.set_title(f"Conductance ({chip} {pair})")
            ax.set_xlabel(fr"$\frac{{100}}{{T}}$ [${ux:~L}$]")
            ax.set_ylabel(fr"$G$ [${uy:~L}$]")
            ax.set_yscale('log')
            res_image = os.path.join(res_dir, f"{chip}_{pair}_temperature_dep.png")
            fig.savefig(res_image, dpi=300)
            plt.close()

            # Power law
            # temperature dependence at low bias
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 9))
            fig.suptitle(f'Power Law ({chip} {pair})')
            temp_win = [0 * ur.K, 100 * ur.K]
            cond = a.is_between(temperature, temp_win)
            x = temperature[cond]
            y = conductance[0][cond]
            x, y = a.strip_nan(x, y)
            coeffs, model = a.fit_powerlaw(x, y, check_nan=False)
            if coeffs is not None:
                alpha = coeffs[0]
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
                x = dh.data['bias']
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
            res_image = os.path.join(res_dir, f"{chip}_{pair}_power_law.png")
            fig.savefig(res_image, dpi=300)
            plt.close()

            print()


def plot_iv():
    global dh, res_dir
    field_win = [-0.05, 0.05] * ur['V/um']
    bias_win = field_win * dh.get_length()
    fig, ax1 = plt.subplots(figsize=(12, 9))
    dh.plot(ax1)
    ax2 = inset_axes(ax1, width='35%', height='35%', loc=4, borderpad=4)
    dh.plot(ax2, x_win=bias_win)
    fig.suptitle(f"{dh.chip} {dh.prop['pair']} at {dh.prop['temperature']}")
    res_image = os.path.join(
        res_dir,
        "{1}_{2}_iv_{0.magnitude}{0.units}.png".format(
            dh.prop['temperature'],
            dh.chip,
            dh.prop['pair']
        )
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=UserWarning)
        fig.savefig(res_image, dpi=300)
    plt.close()


if __name__ == "__main__":
    data_dict = {
        'SPC2': {
            "P2-P4": {
                'nums': np.arange(45, 74),
                'lowtemp_lowbias_num': 45,
                'lowtemp_highbias_num': 74,
            },
            "P17-P18": {
                'nums': np.concatenate([
                    np.arange(78, 102),
                    np.arange(103, 109)
                ]),
                'lowtemp_lowbias_num': 78,
                'lowtemp_highbias_num': 279,
            },
            "P15-P17": {
                'nums': np.arange(124, 155),
                'lowtemp_lowbias_num': 124,
                'lowtemp_highbias_num': 280,
            },
            "P7-P8": {
                'nums': np.arange(156, 186),
                'lowtemp_lowbias_num': 156,
                'lowtemp_highbias_num': 278,
            },
            "P13-P14": {
                'nums': np.concatenate([
                    np.arange(195, 201),
                    np.arange(202, 203),
                    np.arange(204, 208),
                    np.arange(209, 226)
                ]),
                'lowtemp_lowbias_num': 195,
                'lowtemp_highbias_num': 277,
            },
            "P16-P17": {
                'nums': np.arange(227, 258),
                'lowtemp_lowbias_num': 227,
                'lowtemp_highbias_num': 281,
            },
            "P15-P16": {
                'nums': np.concatenate([
                    np.arange(283, 301),
                    [301, 303, 305, 307, 309, 311, 313, 316, 318, 320, 328, 330, 332],
                ]),
                'lowtemp_lowbias_num': 283,
                'lowtemp_highbias_num': 282,
            },
            "P1-P4": {
                'nums': np.arange(335, 366),
                'lowtemp_lowbias_num': 335,
                'lowtemp_highbias_num': 334,
            },
            "P1-P15": {
                'nums': np.arange(367, 399),
                'lowtemp_lowbias_num': 367,
                'lowtemp_highbias_num': 366,
            },
        },
        'SOC3': {
            "P2-P4": {
                'nums': np.arange(45, 74),
                'lowtemp_lowbias_num': 45,
            },
        },
    }

    plot_ivs = True
    main(data_dict, noise_level=0.5 * ur.pA)
