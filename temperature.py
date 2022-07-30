# -*- coding: utf-8 -*-
"""temperature.

Analize temperature dependence.
"""


import os
import time
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.special import gamma
import pandas as pd

from analysis import ur, fmt
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
    df = pd.DataFrame({
        'chip': [],
        'pair': [],
        'ns': [],
        'i0': [],
        'alpha0': [],
        'alpha1': [],
        'alpha2': [],
    })

    for chip in data_dict:
        dh.load_chip(chip)
        res_dir = os.path.join('results', EXPERIMENT, chip)
        os.makedirs(res_dir, exist_ok=True)
        for pair in data_dict[chip]:
            nums = data_dict[chip][pair]['nums']
            names = np.array([f"{chip}_{i}" for i in nums])
            names = names[np.argsort(ur.Quantity.from_list([dh.props[name]['temperature'] for name in names]))]

            dh.load(names[0])
            fields = fields[(fields + delta_field) * dh.get_length() < dh.prop['bias']]
            print("Chip {}, Pair {} of length {}.".format(chip, pair, fmt(dh.get_length())))

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
                ax.errorbar(x_, y_, xerr=dx, yerr=dy, fmt='--o', zorder=i, c=cols[i], label=fr'$E_b = {fmt(field)}$')
                if i == 0:
                    coeffs, model = a.fit_exponential(x[mask], y[mask])
                    if coeffs is not None:
                        act_energy = (- coeffs[0] * 100 * ur.k_B).to('meV')
                        print("Activation energy:", fmt(act_energy))
                        x1 = 100 / temp_win.magnitude
                        ax.plot(x1, model(x1), c='r', zorder=len(fields), label=fr"$U_A = {fmt(act_energy)}$")
            ax.legend()
            ax.set_title(f"Conductance ({chip} {pair})")
            ax.set_xlabel(fr"$\frac{{100}}{{T}}$ [${fmt(ux)}$]")
            ax.set_ylabel(fr"$G$ [${fmt(uy)}$]")
            ax.set_yscale('log')
            cbar.ax.set_ylabel("$E_{bias}$")
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
                print("alpha0:", fmt(alpha0))
                x_, dx, ux = a.separate_measurement(x)
                y_, dy, uy = a.separate_measurement(y)
                ax1.errorbar(x_, y_, xerr=dx, yerr=dy, fmt='o', zorder=1, label='data')
                ax1.plot(x_, model(x_), zorder=2, label=fr'$\alpha = {fmt(alpha0)}$')
            ax1.set_xscale('log')
            ax1.set_yscale('log')
            ax1.set_xlabel(fr"$T$ [${fmt(ux)}$]")
            ax1.set_ylabel(fr"$G$ [${fmt(uy)}$]")
            ax1.legend()
            # bias dependence at low temperature
            if 'lowtemp_highbias_num' in data_dict[chip][pair]:
                lowtemp_highbias_name = f"{chip}_{data_dict[chip][pair]['lowtemp_highbias_num']}"
            else:
                lowtemp_highbias_name = names[0]
            dh.load(lowtemp_highbias_name)
            x = dh.raw['bias']
            y = dh.get_conductance(correct_offset=True, time_win=[0.25, 0.75], field_win=[0.2, np.inf]*ur['V/um'])
            x, y = a.strip_nan(x, y)
            coeffs, model = a.fit_powerlaw(x, y, check_nan=False)
            alpha1 = coeffs[0]
            print("alpha1:", fmt(alpha1))
            x_, dx, ux = a.separate_measurement(x)
            y_, dy, uy = a.separate_measurement(y)
            ax2.errorbar(x_, y_, xerr=dx, yerr=dy, fmt='o', zorder=1)
            ax2.plot(x_, model(x_), zorder=2, label=fr'$\alpha = {fmt(alpha1)}$')
            ax2.set_xscale('log')
            ax2.set_yscale('log')
            ax2.set_xlabel(fr"$V_b$ [${fmt(ux)}$]")
            ax2.set_ylabel(fr"$G$ [${fmt(uy)}$]")
            ax2.legend()
            fig.tight_layout()
            res_image = os.path.join(res_dir, f"{chip}_{pair}_power_law.png")
            fig.savefig(res_image, dpi=100)
            plt.close()

            # universal scaling curve
            alpha2 = a.separate_measurement(alpha0)[0]
            x, y = [], []
            fig, ax = plt.subplots(figsize=(12, 9))
            fig.suptitle('Universal Scaling Curve')
            cond = temperature < 110*ur.K
            cbar, cols = get_cbar_and_cols(fig, a.strip_err(temperature[cond]), log=True, ticks=[10, 15, 30, 60, 100])
            l1 = []
            temperature = []
            current = []
            for i, name in enumerate(names[cond]):
                dh.load(name)
                mask = dh.get_mask(current_win=[3*ur.pA, np.inf], time_win=[0.25, 0.75])
                # temperature_ = a.strip_err(dh.raw['temperature'][mask])
                temperature_ = dh.raw['temperature'][mask]
                temperature.append(temperature_.magnitude)
                x.append((ur.e * dh.raw['bias'][mask] / (2 * np.pi * ur.k_B * temperature_)).to(''))
                y.append(dh.raw['current'][mask] / (temperature_ ** (1 + alpha2)))
                current.append(a.separate_measurement(dh.raw['current'][mask])[0])
                x_, dx, ux = a.separate_measurement(x[i])
                y_, dy, uy = a.separate_measurement(y[i])
                l1.append(ax.errorbar(x_, y_, xerr=dx, yerr=dy, fmt='.', zorder=i, c=cols[i])[0])
            x = np.concatenate(x)
            y = np.concatenate(y)
            x_, dx, ux = a.separate_measurement(x)
            y_, dy, uy = a.separate_measurement(y)

            ax.set_ylim([np.amin(y_) / 3, 3 * np.amax(y_)])
            cbar.ax.set_ylabel(fr"$T$ [${fmt(ur.K)}$]")
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel(r'$\frac{eV}{2\pi k_B T}$')
            ax.set_ylabel(r'$\frac{I}{T^{1+\alpha}}$ ' + f'[${fmt(uy)}$]')

            # Make a horizontal slider to control the frequency.
            sl_ax = plt.axes([0.2, 0.8, 0.4, 0.03])
            sl = Slider(sl_ax, r'$\alpha$', valmin=1, valmax=10, valinit=alpha2)
            bt_ax = plt.axes([0.2, 0.6, 0.15, 0.07])
            bt = Button(bt_ax, r'Confirm $\alpha$')

            def update(val):
                y = []
                for i, l1_ in enumerate(l1):
                    y.append(current[i] / (temperature[i] ** (1 + val)))
                    l1_.set_ydata(y[i])
                    ax.set_ylabel(r'$\frac{I}{T^{1+\alpha}}$ ' + f'[${fmt(ur.A/(ur.T**(1+sl.val)))}$]')
                y = np.concatenate(y)
                ax.set_ylim([np.amin(y) / 3, 3 * np.amax(y)])
                fig.canvas.draw_idle()
                return [val, y]

            ss = SliderSetter(sl, bt, update)
            alpha2, y_ = ss.run()
            print("alpha2:", fmt(alpha2))
            ax.text(np.amin(x_), np.amax(y_), fr'$\alpha = {fmt(alpha2)}$')

            def f(beta, x):
                ns, i0 = beta
                return i0 * np.sinh(np.pi * x / ns) * np.abs(gamma((2 + alpha2)/2 + 1j*x/ns))**2

            ns0 = 9 * a.separate_measurement(dh.get_length())[0]
            coeffs_units = [ur.dimensionless, ur.A]
            coeffs, model = a.fit_generic(
                (x_, None, ux), (y_, None, uy), f, [ns0, ns0*1e-27], coeffs_units,
                log='', already_separated=True, debug=False,
            )
            ns, i0 = coeffs
            print(f'ns: {fmt(ns)}\ni0: {fmt(i0)}')
            ax.text(np.amin(x_), np.amax(y_) / 3, fr'$N_{{sites}} = {fmt(ns)}$')

            x_ = np.linspace(np.amin(x_), np.amax(x_), int(1e3))
            l2, = ax.plot(x_, model(x_), zorder=100, c='r')
            plt.show()
            res_image = os.path.join(res_dir, f"{chip}_{pair}_universal_scaling.png")
            fig.savefig(res_image, dpi=100)
            plt.close()

            df0 = pd.DataFrame({
                'chip': [chip],
                'pair': [dh.prop['pair']],
                'ns': [fmt(ns)],
                'i0': [fmt(i0)],
                'alpha0': [fmt(alpha0)],
                'alpha1': [fmt(alpha1)],
                'alpha2': [fmt(alpha2)],
            })
            df = pd.concat([df, df0], ignore_index=True)

            print()
    df.to_csv(os.path.join('results', EXPERIMENT, 'params.csv'), index=False)
    df.to_latex(os.path.join('results', EXPERIMENT, 'params.tex'), index=False)


class SliderSetter:
    """Set value using a slider and a confirmation button."""

    def __init__(self, sl, bt, update_func):
        self.done = False
        self.res = None
        self.sl = sl
        self.bt = bt
        self.update_func = update_func
        self.sl.on_changed(self.update_func)
        self.bt.on_clicked(self.destroy)

    def destroy(self, event):
        self.sl.ax.set_visible(False)
        self.bt.ax.set_visible(False)
        self.res = self.update_func(self.sl.val)
        self.done = True

    def run(self):
        plt.show(block=False)
        while not self.done:
            plt.pause(1)
        return self.res


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
        # 'SOC3': {
        #     "P2-P4": {
        #         'nums': np.arange(45, 74),
        #         'lowtemp_lowbias_num': 45,
        #     },
        # },
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
