# -*- coding: utf-8 -*-
"""temperature.

Analize temperature dependence.
"""


import os
import numpy as np
from matplotlib import pyplot as plt
from scipy.special import gamma as sp_gamma

from analysis import ur
import analysis as a
from data_plotter import plot_iv_with_inset, get_cbar_and_cols


EXPERIMENT = os.path.splitext(os.path.basename(__file__))[0]


def get_temperature_dependence(dh, names, fields, delta_field, noise_level=0.5*ur.pA):
    temperature = [] * ur.K
    conductance = [[] * ur.S for f in fields]
    max_j = len(fields)
    for i, name in enumerate(names):
        print(f"\rProcessing {i+1} of {len(names)}.", end='', flush=True)
        dh.load(name)
        temperature_ = dh.get_temperature()
        temperature = np.append(temperature, temperature_)
        full_conductance = dh.get_conductance(correct_offset=True, time_win=[0.25, 0.75])
        full_conductance_denoised = dh.get_conductance(correct_offset=True, time_win=[0.25, 0.75], noise_level=noise_level)
        for j, field in enumerate(fields):
            if j < max_j:
                field_win = [field - delta_field, field + delta_field]
                if dh.prop['bias'] < field_win[1] * dh.get_length():
                    max_j = j
                    conductance_ = np.nan
                else:
                    mask = dh.get_mask(field_win=field_win, time_win=[0.25, 0.75])
                    if np.sum(a.isnan(full_conductance_denoised[mask])) < np.sum(mask):
                        conductance_ = a.average(full_conductance[mask])
                    else:
                        conductance_ = np.nan
            else:
                conductance_ = np.nan
            conductance[j] = np.append(conductance[j], conductance_)
    print()
    return temperature, conductance


def main(data_dict, noise_level=1*ur.pA, plot_iv=False):
    """Main analysis routine.
    names: list
    """
    fields = np.concatenate([[0.02], np.arange(0.05, 2, 0.05)]) * ur['V/um']
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

            dh.load(names[0])
            fields = fields[(fields + delta_field) * dh.get_length() < dh.prop['bias']]
            print("Chip {}, Pair {} of length {}.".format(chip, pair, dh.get_length()))

            if 'noise_level' in data_dict[chip][pair]:
                nl = data_dict[chip][pair]['noise_level']
            else:
                nl = noise_level

            if plot_iv:
                for name in names:
                    dh.load(name)
                    plot_iv_with_inset(dh, res_dir)

            temperature, conductance = get_temperature_dependence(dh, names, fields, delta_field, noise_level=nl)

            # plot temperature dependence
            temp_win = [101, 350] * ur.K
            mask = a.is_between(temperature, temp_win)
            x = 100 / temperature
            x_, dx, ux = a.separate_measurement(x)
            fig, ax = plt.subplots(figsize=(12, 9))
            cbar, cols = get_cbar_and_cols(fig, fields, vmin=0)
            for i, field in enumerate(fields):
                y = conductance[i]
                y_, dy, uy = a.separate_measurement(y)
                ax.errorbar(x_, y_, xerr=dx, yerr=dy, fmt='--o', zorder=i, c=cols[i], label=fr'$E_b = {field}$')
                if i == 0:
                    coeffs, model = a.fit_exponential(x[mask], y[mask])
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
            cbar.ax.set_ylabel("$E_{{bias}}$")
            res_image = os.path.join(res_dir, f"{chip}_{pair}_temperature_dep.png")
            fig.savefig(res_image, dpi=100)
            plt.close()

            # Power law
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 9))
            fig.suptitle(f'Power Law ({chip} {pair})')
            # temperature dependence at low bias
            temp_win = [0 * ur.K, 100 * ur.K]
            mask = a.is_between(temperature, temp_win)
            x = temperature[mask]
            y = conductance[0][mask]
            x, y = a.strip_nan(x, y)
            coeffs, model = a.fit_powerlaw(x, y, check_nan=False)
            if coeffs is not None:
                alpha0 = coeffs[0]
                print("alpha0:", alpha0)
                x_, dx, ux = a.separate_measurement(x)
                y_, dy, uy = a.separate_measurement(y)
                ax1.errorbar(x_, y_, xerr=dx, yerr=dy, fmt='o', zorder=1, label='data')
                ax1.plot(x_, model(x_), zorder=2, label=fr'$\alpha = {alpha0}$')
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
            x, y = [], []
            for i, name in enumerate(lowtemp_names):
                dh.load(name)
                x.append(dh.raw['bias'])
                y.append(dh.get_conductance(correct_offset=True, time_win=[0.25, 0.75], noise_level=3*nl))
            x = np.concatenate(x)
            y = np.concatenate(y)
            x, y = a.strip_nan(x, y)
            coeffs, model = a.fit_powerlaw(x, y, check_nan=False)
            alpha1 = coeffs[0]
            print("alpha1:", alpha1)
            x_, dx, ux = a.separate_measurement(x)
            y_, dy, uy = a.separate_measurement(y)
            ax2.errorbar(x_, y_, xerr=dx, yerr=dy, fmt='o', zorder=1)
            ax2.plot(x_, model(x_), zorder=2, label=fr'$\alpha = {alpha1}$')
            ax2.set_xscale('log')
            ax2.set_yscale('log')
            ax2.set_xlabel(fr"$V_b$ [${ux:~L}$]")
            ax2.set_ylabel(fr"$G$ [${uy:~L}$]")
            ax2.legend()
            res_image = os.path.join(res_dir, f"{chip}_{pair}_power_law.png")
            fig.savefig(res_image, dpi=100)
            plt.close()

            # universal scaling curve
            alpha_ = a.separate_measurement(alpha1)[0] + 1
            x, y = [], []
            fig, ax = plt.subplots(figsize=(12, 9))
            cond = temperature < 150*ur.K
            cbar, cols = get_cbar_and_cols(fig, a.strip_err(temperature[cond]), log=True, ticks=[10, 15, 30, 60, 100, 150])
            for i, name in enumerate(names[cond]):
                dh.load(name)
                mask = dh.get_mask(current_win=[3*nl, None], time_win=[0.25, 0.72])
                temperature_ = a.strip_err(dh.get_temperature())
                x.append((ur.e * dh.raw['bias'][mask] / (ur.k_B * temperature_)).to(ur.dimensionless))
                y.append(dh.raw['current'][mask] / (temperature_ ** (alpha_)))
                ax.plot(x[i], y[i], '.', c=cols[i])
            x = np.concatenate(x)
            y = np.concatenate(y)
            x_, dx, ux = a.separate_measurement(x)
            y_, dy, uy = a.separate_measurement(y)

            def f(x, gamma, i0):
                return i0 * np.sinh(gamma * x) * np.abs(sp_gamma((2 + alpha_)/2 + 1j*gamma*x/(np.pi)))**2

            coeffs, model = a.fit_generic(
                (x_, dx, ux), (y_, dy, uy), f, [ur.dimensionless, ur.A],
                ignore_err=True, already_separated=True, debug=False,
                p0=[1e-3, 1]
            )
            gamma, i0 = coeffs
            label = fr'$\gamma={gamma}$, $I_0={i0}$'
            x_ = np.linspace(np.amin(x_), np.amax(x_), int(1e4))
            ax.plot(x_, model(x_), c='r', label=label)
            cbar.ax.set_ylabel(fr"$T$ [${temperature_.units:~L}$]")
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel(r'$\frac{eV}{k_B T}$')
            ax.set_ylabel(r'$\frac{I}{T^{\beta}}$ ' + f'[${uy:~L}$]')
            ax.legend()
            res_image = os.path.join(res_dir, f"{chip}_{pair}_universal_scaling.png")
            fig.savefig(res_image, dpi=100)
            plt.close()

            print()


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
                'nums': np.arange(367, 396),
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
        'SLBC2': {
            "P2-P3": {
                'nums': np.concatenate([
                    np.arange(24, 30),
                    np.arange(48, 52),
                    np.arange(70, 73),
                    np.arange(91, 93),
                    np.arange(111, 114),
                    np.arange(132, 134),
                    np.arange(152, 154),
                    np.arange(172, 174),
                    [192],
                ]),
                'lowtemp_lowbias_num': 24,
                'lowtemp_highbias_num': 5,
            },
        },
    }

    main(data_dict, plot_iv=False)
