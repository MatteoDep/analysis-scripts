# -*- coding: utf-8 -*-
"""temperature.

Analize temperature dependence.
"""


import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.special import gamma
import pandas as pd

from analysis import ur, fmt, ulbl
import analysis as a
from data_plotter import plot_iv_with_inset, get_cbar_and_cols


EXPERIMENT = os.path.splitext(os.path.basename(__file__))[0]


def get_temperature_dependence(dh, names, fields, delta_field, noise_level=0.5*ur.pA):
    temperature = [] * ur.K
    conductance = [[] * ur.S for f in fields]
    max_j = len(fields)
    for i, name in enumerate(names):
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
    return temperature, conductance


def main(data_dict, noise_level=1*ur.pA, adjust_alpha=False, show=False, plot_iv=False):
    """Main analysis routine.
    names: list
    """
    fields = np.concatenate([[0.02], np.arange(0.05, 2, 0.05)]) * ur['V/um']
    delta_field = 0.01 * ur['V/um']
    dh = a.DataHandler()
    if adjust_alpha:
        csv_path = os.path.join('results', EXPERIMENT, 'params_adjusted.csv')
    else:
        csv_path = os.path.join('results', EXPERIMENT, 'params.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame({
            'chip': [],
            'pair': [],
            'length': [],
            'd_length': [],
            'alpha0': [],
            'd_alpha0': [],
            'alpha1': [],
            'd_alpha1': [],
            'alpha2': [],
            'ns': [],
            'd_ns': [],
            'i0': [],
            'd_i0': [],
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
            print("Chip {}, Pair {} of length {}.".format(chip, pair, fmt(dh.get_length())))
            fields = fields[(fields + delta_field) * dh.get_length() < dh.prop['bias']]

            if 'noise_level' in data_dict[chip][pair]:
                nl = data_dict[chip][pair]['noise_level']
            else:
                nl = noise_level

            for i, name in enumerate(names):
                print(f"\rLoading {i+1} of {len(names)}.", end='', flush=True)
                dh.load(name)
                if plot_iv:
                    plot_iv_with_inset(dh, res_dir)
            print()

            temperature, conductance = get_temperature_dependence(dh, names, fields, delta_field, noise_level=nl)

            # plot temperature dependence
            temp_win = [101, 350] * ur.K
            mask = a.is_between(temperature, temp_win)
            x = 100 / temperature
            x_, dx, ux = a.separate_measurement(x)
            fig, ax = plt.subplots()
            fig.suptitle(f"Conductance ({chip} {pair})")
            cbar, cols = get_cbar_and_cols(fig, fields, vmin=0)
            for i, field in enumerate(fields):
                y = conductance[i]
                y_, dy, uy = a.separate_measurement(y)
                ax.errorbar(x_, y_, xerr=dx, yerr=dy, fmt='--o', zorder=i, c=cols[i])
                if i == 0:
                    coeffs, model = a.fit_exponential(x[mask], y[mask])
                    if coeffs is not None:
                        act_energy = (- coeffs[0] * 100 * ur.k_B).to('meV')
                        print("Activation energy:", fmt(act_energy))
                        ax.plot(x_[mask], model(x_[mask]), c='r', zorder=len(fields), label=fr"fit ($U_A = {fmt(act_energy, latex=True)}$)")
            ax.legend()
            ax.set_xlabel(r"$\frac{{100}}{{T}}$" + ulbl(ux))
            ax.set_ylabel(r"$G$" + ulbl(uy))
            ax.set_yscale('log')
            cbar.ax.set_ylabel("$E_{bias}$" + ulbl(fields.u))
            res_image = os.path.join(res_dir, f"{chip}_{pair}_temperature_dep.png")
            fig.savefig(res_image)
            plt.close()

            # Power law
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
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
                ax1.plot(x_, model(x_), zorder=2, label=fr'fit ($\alpha = {fmt(alpha0, latex=True)}$)')
            ax1.set_xscale('log')
            ax1.set_yscale('log')
            ax1.set_xlabel(r"$T$" + ulbl(ux))
            ax1.set_ylabel(r"$G$" + ulbl(uy))
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
            ax2.errorbar(x_, y_, xerr=dx, yerr=dy, fmt='o', zorder=1, label='data')
            ax2.plot(x_, model(x_), zorder=2, label=fr'fit ($\alpha = {fmt(alpha1, latex=True)}$)')
            ax2.set_xscale('log')
            ax2.set_yscale('log')
            ax2.set_xlabel(r"$V_b$" + ulbl(ux))
            ax2.set_ylabel(r"$G$" + ulbl(uy))
            ax2.legend()
            fig.tight_layout()
            res_image = os.path.join(res_dir, f"{chip}_{pair}_power_law.png")
            fig.savefig(res_image)
            plt.close()

            # universal scaling curve

            # plot scaled data
            alpha0_, dalpha0, _ = a.separate_measurement(alpha0)
            alpha1_, dalpha1, _ = a.separate_measurement(alpha1)
            alpha2 = alpha0_
            fig, ax = plt.subplots()
            fig.suptitle('Universal Scaling Curve')
            indices = np.nonzero(temperature < 110*ur.K)
            cbar, cols = get_cbar_and_cols(fig, a.strip_err(temperature[indices]), log=True, ticks=[10, 15, 30, 60, 100])
            cbar.ax.set_ylabel(r"$T$" + ulbl(ur.K))
            x, y = [], []
            current = []
            temperature = []
            lines = []
            for i, name in enumerate(names[indices]):
                dh.load(name)
                mask = dh.get_mask(current_win=[3*ur.pA, 1], time_win=[0.25, 0.5], field_win=[0*ur['V/um'], 1])
                temperature.append(dh.raw['temperature'][mask].m)
                current.append(dh.raw['current'][mask].m)
                x.append((ur.e * dh.raw['bias'][mask] / (2 * np.pi * ur.k_B * dh.raw['temperature'][mask])).to(''))
                y.append(dh.raw['current'][mask] / (dh.raw['temperature'][mask] ** (1 + alpha2)))
                lines.append(ax.plot(x[i].m, y[i].m, '.', zorder=i, c=cols[i])[0])
            x = np.concatenate(x)
            y = np.concatenate(y)
            x_, dx, ux = a.separate_measurement(x)
            y_, dy, uy = a.separate_measurement(y)
            ax.set_xlabel(r'$\frac{eV}{2\pi k_B T}$')
            ax.set_ylabel(r'$\frac{I}{T^{1+\alpha}}$ ' + ulbl(uy))
            ax.set_xscale('log')
            ax.set_yscale('log')

            if adjust_alpha:
                res_image = os.path.join(res_dir, f"{chip}_{pair}_universal_scaling_adjusted.png")

                # Make a horizontal slider to adjust parameter alpha
                sl_ax = plt.axes([0.2, 0.8, 0.4, 0.03])
                sl = Slider(sl_ax, r'$\alpha$', valmin=1, valmax=10, valinit=alpha2)
                bt_ax = plt.axes([0.2, 0.6, 0.15, 0.07])
                bt = Button(bt_ax, r'Confirm $\alpha$')

                def update(val):
                    y_ = []
                    for i, line in enumerate(lines):
                        y_.append(current[i] / (temperature[i] ** (1 + val)))
                        line.set_ydata(y_[i])
                    y_ = np.concatenate(y_)
                    ax.set_ylim([np.amin(y_) / 3, 3 * np.amax(y_)])
                    ax.set_ylabel(r'$\frac{I}{T^{1+\alpha}}$ ' + ulbl(ur.A / ur.K ** (1 + val)))
                    fig.canvas.draw_idle()
                    return y_

                ss = SliderSetter(sl, bt, update)
                alpha2 = ss.run()
                print("alpha2:", fmt(alpha2))
                y_ = update(alpha2)
            else:
                res_image = os.path.join(res_dir, f"{chip}_{pair}_universal_scaling.png")

            # fit data
            def f(beta, x):
                ns, i0 = beta
                return i0 * np.sinh(np.pi * x / ns) * np.abs(gamma((2 + alpha2)/2 + 1j*x/ns))**2

            ns0 = 9 * a.separate_measurement(dh.get_length())[0]
            coeffs_units = [ur[''], ur.A]
            coeffs, model = a.fit_generic(
                (x_, dx, ux), (y_, dy, uy), f, [ns0, ns0*1e-25], coeffs_units,
                log='', already_separated=True, debug=False,
            )
            ns, i0 = coeffs
            print(f'ns: {fmt(ns)}\ni0: {fmt(i0)}')

            x_ = np.linspace(np.amin(x_), np.amax(x_), int(1e3))
            label = fr'fit ($\alpha = {fmt(alpha0, latex=True)}$, $N_{{sites}} = {fmt(ns, latex=True)}$)'
            ax.plot(x_, model(x_), zorder=100, c='r', label=label)
            ax.legend()
            if adjust_alpha:
                plt.show()
                res_image = os.path.join(res_dir, f"{chip}_{pair}_universal_scaling_adjusted.png")
                csv_path = os.path.join(res_dir, 'params_adjusted.csv')
            else:
                res_image = os.path.join(res_dir, f"{chip}_{pair}_universal_scaling.png")
            fig.savefig(res_image)
            plt.close()

            length_, dlength, _ = a.separate_measurement(dh.get_length())
            ns_, dns, _ = a.separate_measurement(ns)
            i0_, di0, _ = a.separate_measurement(i0)
            df0 = pd.DataFrame({
                'chip': [chip],
                'pair': [pair],
                'length': [length_],
                'd_length': [dlength],
                'alpha0': [alpha0_],
                'd_alpha0': [dalpha0],
                'alpha1': [alpha1_],
                'd_alpha1': [dalpha1],
                'alpha2': [alpha2],
                'ns': [ns_],
                'd_ns': [dns],
                'i0': [i0_],
                'd_i0': [di0],
            })
            index = np.nonzero(np.array((df['chip'] == chip) * (df['pair'] == pair)))[0]
            if len(index) > 0:
                df.iloc[index] = df0.iloc[0]
            else:
                df = df0.merge(df, how='outer')

            print()

    df.to_csv(csv_path, index=False)


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
        plt.show(block=False)
        self.done = True

    def run(self):
        plt.show(block=False)
        while not self.done:
            plt.pause(1)
        return self.sl.val


if __name__ == "__main__":
    adjust_alpha = False
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

    main(data_dict, adjust_alpha=adjust_alpha, plot_iv=False)
    res_dir = os.path.join('results', EXPERIMENT)
    if adjust_alpha:
        csv_path = os.path.join(res_dir, 'params_adjusted.csv')
        res_image = os.path.join(res_dir, "number_of_sites_adjusted.png")
    else:
        csv_path = os.path.join(res_dir, 'params.csv')
        res_image = os.path.join(res_dir, "number_of_sites.png")
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots()
    fig.suptitle('Number of Sites')
    for chip in data_dict:
        df0 = df.set_index('chip').loc[chip]
        x_ = np.array(df0['length'])
        dx = np.array(df0['d_length'])
        y_ = np.array(df0['ns'])
        dy = np.array(df0['d_ns'])
        ax.errorbar(x_, y_, xerr=dx, yerr=dy, fmt='.', label=chip)
        if len(x_.shape) > 0:
            _, model = a.fit_linear((x_, dx, ur.um), (y_, dy, ur['']), already_separated=True)
            x_model = np.array([0, np.amax(x_)*1.1])
            ax.plot(x_model, model(x_model), c='r')
    ax.set_xlabel('Length' + ulbl(ur.um))
    ax.set_ylabel(r'$N_{sites}$')
    ax.legend()
    fig.savefig(res_image)
    plt.close()
    df.to_csv(csv_path, index=False)

    df_latex = df.loc[:, ['chip', 'pair']]
    keys = ['length', 'alpha0', 'ns']
    titles = ['Length' + ulbl(ur.um), r'$\alpha$', r'$N_{sites}$']
    if adjust_alpha:
        tex_path = os.path.join(res_dir, 'params_adjusted.tex')
        keys.append('alpha2')
        titles.append(r'$\alpha_{manual}')
    else:
        tex_path = os.path.join(res_dir, 'params.tex')
    for key, title in zip(keys, titles):
        ux = ur.um if key == 'length' else ur['']
        x_ = np.array(df[key])
        dx = np.array(df['d_' + key])
        x = [fmt((l_ * ur['']).plus_minus(dl).m, latex=True) for l_, dl in zip(x_, dx)]
        df_latex[title] = x
    df_latex.set_index(['chip', 'pair']).style.to_latex(tex_path)
