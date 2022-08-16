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
import data_plotter as dp


EXPERIMENT = os.path.splitext(os.path.basename(__file__))[0]


def get_temperature_dependence(dh, names, fields, delta_field):
    temperature = [] * ur.K
    conductance = [[] * ur.S for f in fields]
    max_j = len(fields)
    for i, name in enumerate(names):
        dh.load(name)
        temperature_ = dh.get_temperature(method='average')
        temperature = np.append(temperature, temperature_)
        mask_denoise = dh.get_mask(current_win=[0.5*ur.pA, 1])
        full_conductance = dh.get_conductance()
        full_conductance_denoised = dh.get_conductance(mask=mask_denoise)
        for j, field in enumerate(fields):
            if j < max_j:
                field_win = [field - delta_field, field + delta_field]
                if dh.prop['bias'] < field_win[1] * dh.get_length():
                    max_j = j
                    conductance_i = np.nan
                else:
                    mask = dh.get_mask(field_win=field_win)
                    if np.sum(a.isnan(full_conductance_denoised[mask])) < np.sum(mask):
                        conductance_i = a.average(full_conductance[mask])
                    else:
                        conductance_i = np.nan
            else:
                conductance_i = np.nan
            conductance[j] = np.append(conductance[j], conductance_i)
    return temperature, conductance


def full(data_dict, adjust_alpha=False, plot_iv=False):
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
    df_keys = ['chip', 'pair', 'length', 'd_length', 'act_energy', 'd_act_energy', 'alpha0', 'd_alpha0',
               'alpha1', 'd_alpha1', 'alpha2', 'ns', 'd_ns', 'i0', 'd_i0']
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        for k in df_keys:
            if k not in df:
                df[k] = [None for i in df.index]
        df = df[df_keys]
    else:
        df = pd.DataFrame({k: [] for k in df_keys})

    for chip in data_dict:
        dh.load_chip(chip)
        res_dir = os.path.join('results', EXPERIMENT, chip)
        os.makedirs(res_dir, exist_ok=True)
        for pair in data_dict[chip]:
            nums = data_dict[chip][pair]['nums']
            names = np.array([f"{chip}_{i}" for i in nums])
            names = names[np.argsort(a.qlist_to_qarray([dh.props[name]['temperature'] for name in names]))]

            dh.load(names)
            length = dh.get_length()
            length_, dlength, _ = a.separate_measurement(length)
            print(f"Chip {chip}, Pair {pair} of length {fmt(length)}.")
            fields = fields[(fields + delta_field) * length < dh.props[names[0]]['bias']]

            if 'lowtemp_highbias_num' in data_dict[chip][pair]:
                lowtemp_highbias_name = f"{chip}_{data_dict[chip][pair]['lowtemp_highbias_num']}"
            else:
                lowtemp_highbias_name = names[0]
            if plot_iv:
                dp.plot_iv(dh, names, res_dir, zoom='auto')
                dp.plot_iv(dh, [lowtemp_highbias_name], res_dir, zoom=False, appendix='_high_bias')

            df_index = np.nonzero(np.array((df['chip'] == chip) * (df['pair'] == pair)))[0]

            temperature, conductance = get_temperature_dependence(dh, names, fields, delta_field)

            # plot temperature dependence
            indices = a.is_between(temperature, [101, 350]*ur.K)
            x = 100 / temperature
            x_, dx, ux = a.separate_measurement(x)
            fig, ax = plt.subplots()
            fig.suptitle(f"Conductance ({chip} {pair})")
            cbar, cols = dp.get_cbar_and_cols(fig, fields, vmin=0)
            for i, field in enumerate(fields):
                y = conductance[i]
                x_, dx, ux = a.separate_measurement(x)
                y_, dy, uy = a.separate_measurement(y)
                x_, y_, dx, dy = a.strip_nan(x_, y_, dx, dy)
                ax.errorbar(x_, y_, xerr=dx, yerr=dy, fmt='--o', zorder=i, c=cols[i])
                if i == 0 and np.sum(indices) > 1:
                    coeffs, model = a.fit_exponential(x[indices], y[indices])
                    act_energy = (- coeffs[0] * 100 * ur.k_B).to('meV')
                    act_energy_, dact_energy, _ = a.separate_measurement(act_energy)
                    print("Activation energy:", fmt(act_energy))
                    x_model = 100 / np.array([80, 350])
                    ax.plot(x_model, model(x_model), c='r', zorder=len(fields), label=fr"fit ($U_A = {fmt(act_energy, latex=True)}$)")
            ax.legend()
            ax.set_xlabel(r"$100/T$" + ulbl(ux))
            ax.set_ylabel(r"$G$" + ulbl(uy))
            ax.set_yscale('log')
            cbar.ax.set_ylabel("$E_{bias}$" + ulbl(fields.u))
            res_image = os.path.join(res_dir, f"{chip}_{pair}_temperature_dep.png")
            fig.savefig(res_image)
            plt.close()

            #############
            # Power law #
            #############
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
            fig.suptitle(f'Power Law ({chip} {pair})')

            # power law temperature dependence at low bias
            indices = a.is_between(temperature, [20, 100]*ur.K)
            x = temperature[indices]
            y = conductance[0][indices]
            x, y = a.strip_nan(x, y)
            if np.sum(indices) > 1:
                coeffs, model = a.fit_powerlaw(x, y, check_nan=False)
                alpha0 = coeffs[0]
                alpha0_, dalpha0, _ = a.separate_measurement(alpha0)
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

            # power law bias dependence at low temperature
            dh.load(lowtemp_highbias_name)
            mask = dh.get_mask(field_win=[0.3*ur['V/um'], 1])
            x = dh.get_bias()
            y = dh.get_conductance(mask=mask)
            x, y = a.strip_nan(x, y)
            coeffs, model = a.fit_powerlaw(x, y, check_nan=False)
            alpha1 = coeffs[0]
            alpha1_, dalpha1, _ = a.separate_measurement(alpha1)
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

            ###########################
            # universal scaling curve #
            ###########################

            # plot scaled data
            indices = np.nonzero(temperature < 110*ur.K)
            alpha2 = alpha0_
            fig, ax = plt.subplots()
            cbar, cols = dp.get_cbar_and_cols(fig, a.strip_err(temperature[indices]), log=True, ticks=[10, 15, 30, 60, 100])
            cbar.ax.set_ylabel(r"$T$" + ulbl(ur.K))
            x, y = [], []
            current = []
            temperature = []
            lines = []
            for i, name in enumerate(names[indices]):
                dh.load(name)
                mask = dh.get_mask(current_win=[4*ur.pA, 1])
                temperature_i = dh.get_temperature()[mask]
                bias_i = dh.get_bias()[mask]
                current_i = dh.get_current()[mask]
                temperature.append(temperature_i.m)
                current.append(current_i.m)
                x.append((ur.e * bias_i / (2 * ur.k_B * temperature_i)).to(''))
                y.append(current_i / (temperature_i ** (1 + alpha2)))
                lines.append(ax.plot(x[i].m, y[i].m, '.', zorder=i, c=cols[i])[0])
            x = np.concatenate(x)
            y = np.concatenate(y)
            x_, dx, ux = a.separate_measurement(x)
            y_, dy, uy = a.separate_measurement(y)
            ax.set_xlabel(r'$eV/2 k_B T$')
            ax.set_ylabel(r'$I/T^{1+\alpha}$ ' + ulbl(uy))
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_ylim([np.amin(y_) / 3, 3 * np.amax(y_)])

            if adjust_alpha:
                res_image = os.path.join(res_dir, f"{chip}_{pair}_universal_scaling_adjusted.png")
                fig.suptitle(fr'Universal Scaling Curve - Adjusted $\alpha$ ({chip} {pair})')

                def update(val):
                    y_ = []
                    for i, line in enumerate(lines):
                        y_.append(current[i] / (temperature[i] ** (1 + val)))
                        line.set_ydata(y_[i])
                    y_ = np.concatenate(y_)
                    ax.set_ylim([np.amin(y_) / 3, 3 * np.amax(y_)])
                    ax.set_ylabel(r'$I/T^{1+\alpha}$ ' + ulbl(ur.A / ur.K ** (1 + val)))
                    fig.canvas.draw_idle()
                    return y_

                if df.loc[df_index, 'alpha2'].empty:
                    # Make a horizontal slider to adjust parameter alpha
                    sl_ax = plt.axes([0.2, 0.8, 0.4, 0.03])
                    sl = Slider(sl_ax, r'$\alpha$', valmin=1, valmax=10, valinit=alpha2)
                    bt_ax = plt.axes([0.2, 0.6, 0.15, 0.07])
                    bt = Button(bt_ax, r'Confirm $\alpha$')

                    ss = SliderSetter(sl, bt, update)
                    alpha2 = ss.run()
                else:
                    alpha2 = a.q_from_df(df.loc[df_index], 'alpha2').m
                y_ = update(alpha2)
                print("alpha2:", fmt(alpha2))
            else:
                res_image = os.path.join(res_dir, f"{chip}_{pair}_universal_scaling.png")
                fig.suptitle(f'Universal Scaling Curve ({chip} {pair})')

            def f(beta, x):
                ns, i0 = beta
                return i0 * np.sinh(x / ns) * np.abs(gamma((2 + alpha2)/2 + 1j*x/(np.pi*ns)))**2

            ns0 = 9 * a.separate_measurement(dh.get_length())[0]
            coeffs_units = [ur[''], ur.A]
            coeffs, model = a.fit_generic(
                (x_, dx, ux), (y_, dy, uy), f, [ns0, ns0*1e-25], coeffs_units,
                log='', already_separated=True, debug=False,
            )
            ns, i0 = coeffs
            ns_, dns, _ = a.separate_measurement(ns)
            i0_, di0, _ = a.separate_measurement(i0)
            print(f'ns: {fmt(ns)}\ni0: {fmt(i0)}')

            x_ = np.linspace(np.amin(x_), np.amax(x_), int(1e3))
            label = fr'fit ($\alpha = {fmt(alpha2, latex=True)}$, $N_{{sites}} = {fmt(ns, latex=True)}$)'
            ax.plot(x_, model(x_), zorder=100, c='r', label=label)
            ax.legend()
            fig.savefig(res_image)
            plt.close()

            df0 = pd.DataFrame({
                'chip': [chip],
                'pair': [pair],
                'length': [length_],
                'd_length': [dlength],
                'act_energy': [act_energy_],
                'd_act_energy': [dact_energy],
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
            if df.loc[df_index].empty:
                df = df0.merge(df, how='outer')
            else:
                df.iloc[df_index] = df0.iloc[0]
            dh.clear()
            print()
    df.to_csv(csv_path, index=False)


def nsites(data_dict, adjust_alpha=False):
    res_dir = os.path.join('results', EXPERIMENT)
    title = 'Number of Sites'
    if adjust_alpha:
        csv_path = os.path.join(res_dir, 'params_adjusted.csv')
        res_image = os.path.join(res_dir, "number_of_sites_adjusted.png")
        title += r' - Adjusted $\alpha$'
    else:
        csv_path = os.path.join(res_dir, 'params.csv')
        res_image = os.path.join(res_dir, "number_of_sites.png")
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots()
    fig.suptitle(title)
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

    df_latex = df.loc[:, ['chip', 'pair']]
    keys = ['length', 'act_energy', 'alpha0', 'alpha1', 'ns']
    titles = ['Length' + ulbl(ur.um), r'$U_A$' + ulbl(ur['meV']), r'$\alpha_0$', r'$\alpha_1$', r'$N_{sites}$']
    for key, title in zip(keys, titles):
        x_ = np.array(df[key])
        dx = np.array(df['d_' + key])
        x = ['$' + fmt((l_ * ur['']).plus_minus(dl).m, latex=True) + '$' for l_, dl in zip(x_, dx)]
        df_latex[title] = x
    if adjust_alpha:
        x_ = np.array(df['alpha2'])
        x = ['$' + fmt((l_), latex=True) + '$' for l_ in x_]
        title = r'$\alpha_2$'
        df_latex[title] = x
        tex_path = os.path.join(res_dir, 'params_adjusted.tex')
    else:
        tex_path = os.path.join(res_dir, 'params.tex')
    df_latex.set_index(['chip', 'pair']).style.to_latex(tex_path)


def compare_stabtime(names):
    fields = [0.02] * ur['V/um']
    delta_field = 0.01 * ur['V/um']
    dh = a.DataHandler()

    for chip in data_dict:
        dh.load_chip(chip)
        res_dir = os.path.join('results', EXPERIMENT, chip)
        os.makedirs(res_dir, exist_ok=True)
        for pair in data_dict[chip]:
            nums = data_dict[chip][pair]['nums']
            stabtime = [] * ur.s
            names = []
            for time_key, time_nums in nums.items():
                stabtime = np.append(stabtime, ur.Quantity(time_key))
                names.append([f"{chip}_{i}" for i in time_nums])

            fig, ax = plt.subplots()
            fig.suptitle(f"Conductance ({chip} {pair})")
            fit_cols = list('rkgmb')
            for i, stabtime_i in enumerate(stabtime):
                temperature, conductance = get_temperature_dependence(dh, names[i], fields, delta_field)
                conductance = conductance[0]

                indices = a.is_between(temperature, [101, 350]*ur.K)
                x = 100 / temperature[indices]
                y = conductance[indices]
                x_, dx, ux = a.separate_measurement(x)
                y_, dy, uy = a.separate_measurement(y)
                ax.errorbar(x_, y_, xerr=dx, yerr=dy, fmt='o', zorder=i, c=f'C{i}', label=f'data after ${fmt(stabtime_i, latex=True)}$')

                coeffs, model = a.fit_exponential(x, y)
                act_energy = (- coeffs[0] * 100 * ur.k_B).to('meV')
                print(f"Activation energy ({fmt(stabtime_i)}): {fmt(act_energy)}")
                x_model = 100 / np.array([80, 350])
                ax.plot(x_model, model(x_model), c=fit_cols[i], zorder=len(stabtime)+i,
                        label=fr"fit after ${fmt(stabtime_i, latex=True)}$ ($U_A={fmt(act_energy, latex=True)}$)")
            ax.legend()
            ax.set_xlabel(r"$100/T$" + ulbl(ux))
            ax.set_ylabel(r"$G$" + ulbl(uy))
            ax.set_yscale('log')
            res_image = os.path.join(res_dir, f"{chip}_{pair}_stabtime_comparison.png")
            fig.savefig(res_image)
            plt.close()


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
                'lowtemp_highbias_num': 74,
            },
            "P17-P18": {
                'nums': np.concatenate([
                    np.arange(78, 102),
                    np.arange(103, 109)
                ]),
                'lowtemp_highbias_num': 279,
            },
            "P15-P17": {
                'nums': np.arange(124, 155),
                'lowtemp_highbias_num': 280,
            },
            "P7-P8": {
                'nums': np.arange(156, 186),
                'lowtemp_highbias_num': 278,
            },
            "P13-P14": {
                'nums': np.concatenate([
                    np.arange(195, 201),
                    np.arange(202, 203),
                    np.arange(204, 208),
                    np.arange(209, 226)
                ]),
                'lowtemp_highbias_num': 277,
            },
            "P16-P17": {
                'nums': np.arange(227, 258),
                'lowtemp_highbias_num': 281,
            },
            "P15-P16": {
                'nums': np.concatenate([
                    np.arange(283, 301),
                    [301, 303, 305, 307, 309, 311, 313, 316, 318, 320, 328, 330, 332],
                ]),
                'lowtemp_highbias_num': 282,
            },
            "P1-P4": {
                'nums': np.arange(335, 366),
                'lowtemp_highbias_num': 334,
            },
            "P1-P15": {
                'nums': np.arange(367, 396),
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
                'lowtemp_highbias_num': 5,
            },
        },
    }

    full(data_dict, adjust_alpha=adjust_alpha, plot_iv=False)
    nsites(data_dict, adjust_alpha=adjust_alpha)

    data_dict = {
        'SPC2': {
            'P15-P16': {
                'nums': {
                    '10s': [301, 303, 305, 307, 309, 311, 313, 316, 318, 320, 328, 330, 332],
                    '20s': [300, 302, 304, 306, 308, 310, 312, 315, 317, 319, 327, 329, 331],
                }
            }
        }
    }

    compare_stabtime(data_dict)
