# -*- coding: utf-8 -*-
"""temperature.

Analize temperature dependence.
"""


import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib import ticker
from scipy.special import gamma
import pandas as pd

from analysis import ur, fmt, ulbl
import analysis as a
import data_plotter as dp


EXPERIMENT = os.path.splitext(os.path.basename(__file__))[0]
DEFAULT_NOISE_LEVEL = 0.5*ur.pA
RES_DIR = os.path.join('results', EXPERIMENT)
os.makedirs(RES_DIR, exist_ok=True)


def get_temperature_dependence(dh, names, fields, delta_field, noise_level=DEFAULT_NOISE_LEVEL):
    """Get the temperature dependence of the conductance for different field values."""
    temperature = [] * ur.K
    conductance = [[] * ur.nS for f in fields]
    for i, name in enumerate(names):
        dh.load(name)
        temperature_i = dh.get_temperature(method='average')
        temperature = np.append(temperature, temperature_i)
        mask_denoise = dh.get_mask(current_win=[noise_level, 1])
        full_conductance = dh.get_conductance(mask=mask_denoise)
        for j, field in enumerate(fields):
            field_win = [field - delta_field, field + delta_field]
            mask = dh.get_mask(field_win=field_win)
            if field == 0 * ur['V/um']:
                cond = np.sum(a.isnan(full_conductance[mask])) < 0.9 * np.sum(mask)
                if cond:
                    conductance_i = dh.get_conductance(method='fit', mask=mask)
                else:
                    conductance_i = np.nan
            else:
                cond = (np.sum(a.isnan(full_conductance[mask])) < 0.1 * np.sum(mask)) * (field_win[1] <= dh.prop['bias'] / dh.get_length())
                if cond:
                    conductance_i = a.average(full_conductance[mask])
                else:
                    conductance_i = np.nan
            conductance[j] = np.append(conductance[j], conductance_i)
    return temperature, conductance


def low_temperature(dh, data_dict):
    """Main routine function.
    :param dh: DataHandler object.
    :param data_dict: dictionary like
        {
            'chipA': {
                'pair1': {
                    'nums': array-like of file numbers,
                    'lowtemp_highbias_num': number of high bias measurement at lowest temperature.
                }
                'pair2: ...
            'chipB': ...
        }
    """
    print("Low Temperature Analysis")
    fields = np.arange(0, 2, 0.05) * ur['V/um']
    delta_field = 0.005 * ur['V/um']

    csv_path = os.path.join('results', EXPERIMENT, 'ltparams.csv')
    df_keys = ['chip', 'pair', 'length', 'd_length', 'alpha0', 'd_alpha0',
               'alpha1', 'd_alpha1', 'alpha2', 'ns0', 'd_ns0', 'i0', 'd_i0', 'ns2', 'd_ns2', 'i02', 'd_i02']
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
        for pair in data_dict[chip]:
            nums = data_dict[chip][pair]['nums']
            names = np.array([f"{chip}_{i}" for i in nums])
            names = names[np.argsort(a.qlist_to_qarray([dh.props[name]['temperature'] for name in names]))]

            dh.load(names[0])
            length = dh.get_length()
            length_, dlength, _ = a.separate_measurement(length)
            print(f"Chip {chip}, Pair {pair} of length {fmt(length)}.")

            if 'lowtemp_highbias_num' in data_dict[chip][pair]:
                lowtemp_highbias_name = f"{chip}_{data_dict[chip][pair]['lowtemp_highbias_num']}"
            else:
                lowtemp_highbias_name = names[0]
            max_field = dh.prop['bias'] / dh.get_length()
            fact = max(a.separate_measurement((fields[4] + delta_field) // max_field)[0] + 1, 1)
            fields_ = fields / fact
            delta_field_ = delta_field / fact
            fields_ = fields_[fields_ + delta_field_ < max_field]

            df_index = np.nonzero(np.array((df['chip'] == chip) * (df['pair'] == pair)))[0]

            if 'noise_level' in data_dict[chip][pair]:
                noise_level = data_dict[chip][pair]['noise_level']
            else:
                noise_level = DEFAULT_NOISE_LEVEL
            temperature, conductance = get_temperature_dependence(dh, names, fields_, delta_field, noise_level=noise_level)

            # plot temperature dependence
            x = 100 / temperature
            x_, dx, ux = a.separate_measurement(x)
            fig, ax = plt.subplots()
            fig.suptitle("Conductance Temperature dependence")
            cbar, cols = dp.get_cbar_and_cols(fig, fields_, vmin=0)
            for i, field in enumerate(fields_):
                y = conductance[i]
                x_, dx, ux = a.separate_measurement(x)
                y_, dy, uy = a.separate_measurement(y)
                x_, y_, dx, dy = a.strip_nan(x_, y_, dx, dy)
                ax.errorbar(x_, y_, xerr=dx, yerr=dy, fmt='--o', zorder=i, c=cols[i])
            ax.set_xlabel(r"$100/T$" + ulbl(ux))
            ax.set_ylabel(r"$G$" + ulbl(uy))
            ax.set_yscale('log')
            cbar.ax.set_ylabel(r"$\mathcal{E}_{bias}$" + ulbl(fields_.u))
            res_image = os.path.join(RES_DIR, f"{chip}_{pair}_temperature_dep.png")
            fig.savefig(res_image)
            plt.close()

            #############
            # Power law #
            #############
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            fig.suptitle('Power Law')

            # power law temperature dependence at low bias
            indices = a.is_between(temperature, [20, 110]*ur.K)
            x = temperature[indices]
            y = conductance[1][indices]
            x, y = a.strip_nan(x, y)
            if len(x) > 1:
                x_, dx, ux = a.separate_measurement(x)
                y_, dy, uy = a.separate_measurement(y)
                coeffs, model = a.fit_powerlaw((x_, dx, ux), (y_, dy, uy), check_nan=False)
                alpha0 = coeffs[0]
                alpha0_, dalpha0, _ = a.separate_measurement(alpha0)
                print("alpha0:", fmt(alpha0))
                ax1.errorbar(x_, y_, xerr=dx, yerr=dy, fmt='o', zorder=1, label='data')
                ax1.plot(x_, model(x_), zorder=2, label='fit')
            ax1.set_xscale('log')
            ax1.set_yscale('log')
            ax1.xaxis.set_minor_formatter(ticker.ScalarFormatter())
            ax1.xaxis.set_major_formatter(ticker.ScalarFormatter())
            ax1.set_xlabel(r"$T$" + ulbl(ux))
            ax1.set_ylabel(r"$G$" + ulbl(uy))
            ax1.legend()

            # power law bias dependence at low temperature
            dh.load(lowtemp_highbias_name)
            # mask = dh.get_mask(current_win=[10*ur.pA, 1])
            field_win = [0.5, 1]*ur['V/um'] / fact
            mask = dh.get_mask(field_win=field_win)
            x = dh.get_field(mask=mask)
            y = dh.get_conductance(mask=mask)
            x_, dx, ux = a.separate_measurement(x)
            y_, dy, uy = a.separate_measurement(y)
            coeffs, model = a.fit_powerlaw((x_, None, ux), (y_, dy, uy))
            alpha1 = coeffs[0]
            alpha1_, dalpha1, _ = a.separate_measurement(alpha1)
            print("alpha1:", fmt(alpha1))
            ax2.errorbar(x_, y_, xerr=None, yerr=dy, fmt='o', zorder=1, label='data')
            ax2.plot(x_, model(x_), zorder=2, label='fit')
            ax2.set_xscale('log')
            ax2.set_yscale('log')
            ax2.xaxis.set_minor_formatter(ticker.ScalarFormatter())
            ax2.xaxis.set_major_formatter(ticker.ScalarFormatter())
            ax2.set_xlabel(r"$\mathcal{E}_{bias}$" + ulbl(ux))
            ax2.set_ylabel(r"$G$" + ulbl(uy))
            ax2.legend()
            res_image = os.path.join(RES_DIR, f"{chip}_{pair}_power_law.png")
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
                mask = dh.get_mask(field_win=[0, 1], current_win=[6*noise_level, 1])
                temperature_i = dh.get_temperature()[mask]
                bias_i = dh.get_bias()[mask]
                current_i = dh.get_current()[mask].to('nA')
                temperature.append(temperature_i.m)
                current.append(current_i.m)
                x.append((ur.e * bias_i / (2 * ur.k_B * temperature_i)).to(''))
                y.append(current_i / (temperature_i ** (1 + alpha2)))
                lines.append(ax.plot(x[i].m, y[i].m, '.', c=cols[i])[0])
            x = np.concatenate(x)
            y = np.concatenate(y)
            x_, dx, ux = a.separate_measurement(x)
            y_, dy, uy = a.separate_measurement(y)
            ax.set_xlabel(r'$eV/2 k_B T$')
            ax.set_ylabel(r'$I/T^{1+\alpha}$' + ulbl(uy))
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_ylim([np.amin(y_) / 3, 3 * np.amax(y_)])

            # get alpha2
            def update(val):
                y2_ = []
                for i, line in enumerate(lines):
                    y2_.append(current[i] / (temperature[i] ** (1 + val)))
                    line.set_ydata(y2_[i])
                y2_ = np.concatenate(y2_)
                ax.set_ylim([np.amin(y2_) / 3, 3 * np.amax(y2_)])
                ax.set_ylabel(r'$I/T^{1+\alpha}$ ' + ulbl(ur.nA / ur.K ** (1 + val)))
                fig.canvas.draw_idle()
                return y2_

            if df.loc[df_index, 'alpha2'].empty:
                # Make a horizontal slider to adjust parameter alpha
                sl_ax = plt.axes([0.2, 0.8, 0.4, 0.03])
                sl = Slider(sl_ax, r'$\alpha_2$', valmin=1, valmax=10, valinit=alpha2)
                bt_ax = plt.axes([0.2, 0.6, 0.15, 0.07])
                bt = Button(bt_ax, r'Confirm $\alpha_2$')
                ss = SliderSetter(sl, bt, update)
                alpha2 = ss.run()
            else:
                alpha2 = a.q_from_df(df.loc[df_index], 'alpha2').m
            print("alpha2:", fmt(alpha2))

            def fit_scaling_curve(alpha, y_, factor):
                def f(beta, x):
                    ns0, i0 = beta
                    return i0 * np.sinh(x / ns0) * np.abs(gamma((2 + alpha)/2 + 1j*x/(np.pi*ns0)))**2

                ns0 = 9 * length_
                coeffs_units = [ur[''], ur.nA]
                coeffs, model = a.fit_generic(
                    (x_, None, None), (y_, None, None), f, [ns0, ns0*factor], coeffs_units,
                    log='xy', debug=False,
                )
                return coeffs, model

            # using alpha0
            fig.suptitle(r'Universal Scaling Curve - Using $\alpha_0$')
            y_ = update(alpha0_)
            # txt = ax.text(0.17, 0.8, fr'$\alpha_0 = {fmt(alpha0, latex=True)}$', transform=ax.transAxes)
            (ns0, i0), model = fit_scaling_curve(alpha0_, y_, 1e-16)
            ns0_, dns0, _ = a.separate_measurement(ns0)
            i0_, di0, _ = a.separate_measurement(i0)
            print(f'ns0: {fmt(ns0)}\ni0: {fmt(i0)}')
            x_model = np.linspace(np.amin(x_), np.amax(x_), int(1e3))
            label = 'fit'
            fit_line, = ax.plot(x_model, model(x_model), c='r', label=label)
            ax.legend()
            res_image = os.path.join(RES_DIR, f"{chip}_{pair}_universal_scaling.png")
            fig.savefig(res_image)

            # using alpha2
            fig.suptitle(r'Universal Scaling Curve - Using manual $\alpha$')
            y2_ = update(alpha2)
            # txt.set_text(fr'$\alpha_2 = {fmt(alpha2, latex=True)}$')
            (ns2, i02), model = fit_scaling_curve(alpha2, y2_, 1e-18)
            ns2_, dns2, _ = a.separate_measurement(ns2)
            i02_, di02, _ = a.separate_measurement(i02)
            fit_line.set_ydata(model(x_model))
            fit_line.set_label('fit')
            print(f'ns2: {fmt(ns2)}\ni02: {fmt(i02)}')
            ax.legend()
            res_image = os.path.join(RES_DIR, f"{chip}_{pair}_universal_scaling_adjusted.png")
            fig.savefig(res_image)
            plt.close()

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
                'ns0': [ns0_],
                'd_ns0': [dns0],
                'i0': [i0_],
                'd_i0': [di0],
                'ns2': [ns2_],
                'd_ns2': [dns2],
                'i02': [i02_],
                'd_i02': [di02],
            })
            if df.loc[df_index].empty:
                df = df0.merge(df, how='outer')
            else:
                df.iloc[df_index] = df0.iloc[0]
            dh.clear()
            print()
    df.to_csv(csv_path, index=False)


def nsites(data_dict):
    print('Plot number of sites')
    RES_DIR = os.path.join('results', EXPERIMENT)
    csv_path = os.path.join(RES_DIR, 'ltparams.csv')
    df = pd.read_csv(csv_path)

    ykeys = ['ns0', 'ns2']
    res_images = [
        os.path.join(RES_DIR, "number_of_sites.png"),
        os.path.join(RES_DIR, "number_of_sites_adjusted.png"),
    ]
    titles = [
        'Number of Sites',
        r'Number of Sites - Adjusted $\alpha$',
    ]
    for i, ykey in enumerate(ykeys):
        fig, ax = plt.subplots()
        fig.suptitle(titles[i])
        for chip in data_dict:
            df0 = df.set_index('chip').loc[chip]
            x_ = np.array(df0['length'])
            dx = np.array(df0['d_length'])
            y_ = np.array(df0[ykey])
            dy = np.array(df0['d_'+ykey])
            ax.errorbar(x_, y_, xerr=dx, yerr=dy, fmt='o', label=chip)
            if len(x_.shape) > 0:
                _, model = a.fit_linear((x_, dx, ur.um), (y_, dy, ur['']))
                x_model = np.array([0, np.amax(x_)*1.1])
                ax.plot(x_model, model(x_model), c='r')
        ax.set_xlabel('Length' + ulbl(ur.um))
        ax.set_ylabel(r'$N_{sites'+ykey.replace('ns', ',')+r'}$')
        ax.legend()
        fig.savefig(res_images[i])
        plt.close()

    keys = ['length', 'alpha0', 'alpha1', 'alpha2', 'ns0', 'ns2']
    units = [ur.um, ur[''], ur[''], ur[''], ur[''], ur[''], ur['']]
    titles = ['Length', r'$\alpha_0$', r'$\alpha_1$', r'$\alpha_2$', r'$N_{sites,0}$', r'$N_{sites,2}$']
    table = create_table(df, keys, units, titles)
    table = table.replace('tabular}{ll', 'tabular}{ll|')
    df_tex_path = os.path.join(RES_DIR, 'ltparams.tex')
    with open(df_tex_path, 'w') as f:
        print(table, file=f)


def create_table(df, keys, units, titles):
    averages_str = r"\multicolumn{2}{c|}{\textbf{average}}"
    df_indexed = df.set_index(['chip', 'length']).sort_index()
    df = df_indexed.reset_index()
    df_latex = df.loc[:, ['chip', 'pair']]
    for key, title, unit in zip(keys, titles, units):
        x_ = np.array(df[key])
        if key == 'alpha2':
            x = x_
            x_str = ['$' + fmt((x_i), latex=True) + '$' for x_i in x_]
        else:
            dx = np.array(df['d_' + key])
            x = a.qlist_to_qarray([(x_i * unit).plus_minus(dxi) for x_i, dxi in zip(x_, dx)])
            x_str = ['$' + fmt(xi.m, latex=True) + '$' for xi in x]
        averages_str += ' & '
        if key not in ['length', 'ns0', 'ns2']:
            averages_str += fmt(a.strip_err(a.average(x)), latex=True)
        df_latex[title+ulbl(unit)] = x_str

    # write table
    latex_str = df_latex.set_index(['chip', 'pair']).style.to_latex()
    end_str = r'\end{tabular}'
    table = ""
    for i, line in enumerate(latex_str.split('\n')):
        if i > 2 and not line.startswith(' '):
            table += '\n' + r'\hline'
        if line.startswith(end_str):
            break
        if line != "":
            table += '\n' + line
    table += '\n' + averages_str
    table += '\n' + end_str
    return table


def high_temperature(dh, data_dict):
    print("High Temperature Analysis")
    fields = np.arange(0, 2, 0.05) * ur['V/um']
    delta_field = 0.005 * ur['V/um']
    x_model = 100 / np.array([100, 350])

    csv_path = os.path.join('results', EXPERIMENT, 'htparams.csv')
    df_keys = ['chip', 'pair', 'length', 'd_length', 'act_energy', 'd_act_energy']
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        for k in df_keys:
            if k not in df:
                df[k] = [None for i in df.index]
        df = df[df_keys]
    else:
        df = pd.DataFrame({k: [] for k in df_keys})

    x = []
    y = []
    act_energy = [] * ur.meV
    markers_list = ['o', 'd', 's', '^', 'v']
    markers_index = 0
    markers = []
    models = []
    for chip in data_dict:
        dh.load_chip(chip)
        for pair in data_dict[chip]:
            nums = data_dict[chip][pair]['nums']
            names = [f"{chip}_{i}" for i in nums if dh.props[f"{chip}_{i}"]['temperature'] > 100*ur.K]

            dh.load(names[0])
            length = dh.get_length()
            length_, dlength, _ = a.separate_measurement(length)
            print(f"Chip {chip}, Pair {pair} of length {fmt(length)}.")

            max_field = dh.prop['bias'] / dh.get_length()
            fact = max(a.separate_measurement((fields[4] + delta_field) // max_field)[0] + 1, 1)
            fields_ = fields / fact
            delta_field_ = delta_field / fact
            fields_ = fields_[fields_ + delta_field_ < max_field]

            df_index = np.nonzero(np.array((df['chip'] == chip) * (df['pair'] == pair)))[0]

            temperature, conductance = get_temperature_dependence(dh, names, fields, delta_field)
            xi = 100 / temperature
            act_energy_, dact_energy = None, None
            x.append(xi)
            y.append(conductance[0])
            coeffs, model = a.fit_exponential(x[-1], y[-1])
            if coeffs is not None:
                act_energyi = (- coeffs[0] * 100 * ur.k_B).to('meV')
                act_energy_, dact_energy, _ = a.separate_measurement(act_energyi)
                act_energy = np.append(act_energy, act_energyi)
                models.append(model)
                markers.append(markers_list[markers_index])

            fig, ax = plt.subplots()
            fig.suptitle("High Temperature Conductance")
            cbar, cols = dp.get_cbar_and_cols(fig, fields_, vmin=0)
            for i, field in enumerate(fields_):
                x_, dx, ux = a.separate_measurement(xi)
                y_, dy, uy = a.separate_measurement(conductance[i])
                x_, y_, dx, dy = a.strip_nan(x_, y_, dx, dy)
                ax.errorbar(x_, y_, xerr=dx, yerr=dy, fmt='--o', zorder=i, c=cols[i])
                if i == 0 and act_energy_ is not None:
                    print("Activation energy:", fmt(act_energyi))
                    ax.plot(x_model, model(x_model), c='r', zorder=len(fields_), label="fit")
            ax.set_xlabel(r"$100/T$" + ulbl(ux))
            ax.set_ylabel(r"$G$" + ulbl(uy))
            cbar.ax.set_ylabel(r"$\mathcal{E}_{bias}$" + ulbl(fields_.u))
            ax.set_yscale('log')
            res_image = os.path.join(RES_DIR, f"{chip}_{pair}_high_temperature.png")
            fig.savefig(res_image)
            plt.close()

            df0 = pd.DataFrame({
                'chip': [chip],
                'pair': [pair],
                'length': [length_],
                'd_length': [dlength],
                'act_energy': [act_energy_],
                'd_act_energy': [dact_energy],
            })
            if df.loc[df_index].empty:
                df = df0.merge(df, how='outer')
            else:
                df.iloc[df_index] = df0.iloc[0]
            dh.clear()
            print()
        markers_index += 1
    df.to_csv(csv_path, index=False)

    fig, ax = plt.subplots()
    fig.suptitle("High Temperature Conductance")
    act_energy_ = a.separate_measurement(act_energy)[0]
    ticks = np.arange(np.amin(act_energy_), np.amax(act_energy_), 5)
    cbar, cols = dp.get_cbar_and_cols(fig, act_energy, ticks=ticks)
    for i, act_energy_i in enumerate(act_energy):
        x_, dx, ux = a.separate_measurement(x[i])
        y_, dy, uy = a.separate_measurement(y[i])
        ax.errorbar(x_, y_, xerr=dx, yerr=dy, fmt=markers[i], zorder=i, c=cols[i])
        ax.plot(x_model, models[i](x_model), c=cols[i], zorder=len(act_energy)+i)
        if i == 0:
            cbar.ax.set_ylabel(r"$E_A$" + ulbl(ur.meV))
            ax.set_xlabel(r"$100/T$" + ulbl(ux))
            ax.set_ylabel(r"$G$" + ulbl(uy))
    ax.legend(handles=[
        plt.Line2D([0], [0], color='k', linestyle='', marker=m, label=f"{chip}")
        for m, chip in zip(np.unique(markers), data_dict.keys())
    ])
    ax.set_yscale('log')
    res_image = os.path.join(RES_DIR, "high_temperature_all.png")
    fig.savefig(res_image)
    plt.close()

    keys = ['length', 'act_energy']
    units = [ur.um, ur.meV]
    titles = ['Length', r'$E_A$']
    table = create_table(df, keys, units, titles)
    table = table.replace('tabular}{ll', 'tabular}{ll|')
    df_tex_path = os.path.join(RES_DIR, 'htparams.tex')
    with open(df_tex_path, 'w') as f:
        print(table, file=f)


def interface(dh, data_dict, fit_indices=[0]):
    print("Interface Analysis")
    fields = np.arange(0, 2, 0.05) * ur['V/um']
    delta_field = 0.005 * ur['V/um']
    x_model = 100 / np.array([100, 350])
    fit_indices = [0, 1]

    csv_path = os.path.join('results', EXPERIMENT, 'htparams_alt.csv')
    df_keys = ['chip', 'pair', 'length', 'd_length', 'act_energy0', 'd_act_energy0', 'act_energy1', 'd_act_energy1']
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
        for pair in data_dict[chip]:
            nums = data_dict[chip][pair]['nums']
            names = [f"{chip}_{i}" for i in nums if dh.props[f"{chip}_{i}"]['temperature'] > 100*ur.K]

            dh.load(names[0])
            length = dh.get_length()
            length_, dlength, _ = a.separate_measurement(length)
            print(f"Chip {chip}, Pair {pair} of length {fmt(length)}.")

            max_field = dh.prop['bias'] / dh.get_length()
            fact = max(a.separate_measurement((fields[4] + delta_field) // max_field)[0] + 1, 1)
            fields_ = fields / fact
            delta_field_ = delta_field / fact
            fields_ = fields_[fields_ + delta_field_ < max_field]

            df_index = np.nonzero(np.array((df['chip'] == chip) * (df['pair'] == pair)))[0]

            temperature, conductance = get_temperature_dependence(dh, names, fields, delta_field)
            act_energy = []
            act_energy_, dact_energy = [[None, None] for i in fit_indices]
            fig, ax = plt.subplots()
            fig.suptitle("High Temperature Conductance")
            cbar, cols = dp.get_cbar_and_cols(fig, fields_, vmin=0)
            for i, field in enumerate(fields_):
                x = 100 / temperature
                y = conductance[i]
                x_, dx, ux = a.separate_measurement(x)
                y_, dy, uy = a.separate_measurement(y)
                x_, y_, dx, dy = a.strip_nan(x_, y_, dx, dy)
                ax.errorbar(x_, y_, xerr=dx, yerr=dy, fmt='--o', zorder=i, c=cols[i])
                if i in fit_indices and act_energy_ is not None:
                    coeffs, model = a.fit_exponential(x, y)
                    act_energy.append((- coeffs[0] * 100 * ur.k_B).to('meV'))
                    print("Activation energy:", fmt(act_energy[i]))
                    ax.plot(x_model, model(x_model), c=['r', 'k'][i], zorder=len(fields_)+i, label="$E_A="+fmt(act_energy[i], latex=True)+"$")
            ax.set_xlabel(r"$100/T$" + ulbl(ux))
            ax.set_ylabel(r"$G$" + ulbl(uy))
            cbar.ax.set_ylabel(r"$\mathcal{E}_{bias}$" + ulbl(fields_.u))
            ax.set_yscale('log')
            ax.legend()
            res_image = os.path.join(RES_DIR, f"{chip}_{pair}_high_temperature_alt.png")
            fig.savefig(res_image)
            plt.close()

            dfdict = {
                'chip': [chip],
                'pair': [pair],
                'length': [length_],
                'd_length': [dlength],
            }
            for i in fit_indices:
                act_energy_, dact_energy, _ = a.separate_measurement(act_energy[i])
                dfdict.update({
                    f'act_energy{i}': [act_energy_],
                    f'd_act_energy{i}': [dact_energy],
                 })
            df0 = pd.DataFrame(dfdict)
            if df.loc[df_index].empty:
                df = df0.merge(df, how='outer')
            else:
                df.iloc[df_index] = df0.iloc[0]
            dh.clear()
            print()
    df.to_csv(csv_path, index=False)

    keys = ['length', 'act_energy0', 'act_energy1']
    units = [ur.um, ur.meV, ur.meV]
    titles = ['Length', r'$E_A$ ('+fmt(fields_[0], latex=True)+')', r'$E_A$ ('+fmt(fields_[1], latex=True)+')']
    table = create_table(df, keys, units, titles)
    table = table.replace('tabular}{ll', 'tabular}{ll|')
    df_tex_path = os.path.join(RES_DIR, 'htparams_alt.tex')
    with open(df_tex_path, 'w') as f:
        print(table, file=f)


def compare_stabtime(dh, data_dict):
    print('Compare stabilization time')
    fields = [0.01] * ur['V/um']
    delta_field = 0.005 * ur['V/um']

    for chip in data_dict:
        dh.load_chip(chip)
        for pair in data_dict[chip]:
            nums = data_dict[chip][pair]['nums']
            stabtime = [] * ur.s
            names = []
            for time_key, time_nums in nums.items():
                stabtime = np.append(stabtime, ur.Quantity(time_key))
                names.append([f"{chip}_{i}" for i in time_nums if dh.props[f"{chip}_{i}"]['temperature'] > 100*ur.K])

            fig, ax = plt.subplots()
            fig.suptitle("Conductance")
            fit_cols = list('rkgmb')
            for i, stabtime_i in enumerate(stabtime):
                temperature, conductance = get_temperature_dependence(dh, names[i], fields, delta_field)
                conductance = conductance[0]

                x = 100 / temperature
                y = conductance
                x_, dx, ux = a.separate_measurement(x)
                y_, dy, uy = a.separate_measurement(y)
                ax.errorbar(x_, y_, xerr=dx, yerr=dy, fmt='o', zorder=i, c=f'C{i}', label=f'data after ${fmt(stabtime_i, latex=True)}$')

                coeffs, model = a.fit_exponential(x, y)
                act_energy = (- coeffs[0] * 100 * ur.k_B).to('meV')
                print(f"Activation energy ({fmt(stabtime_i)}): {fmt(act_energy)}")
                x_model = 100 / np.array([100, 350])
                ax.plot(x_model, model(x_model), c=fit_cols[i], zorder=len(stabtime)+i,
                        label=fr"fit after ${fmt(stabtime_i, latex=True)}$")
            ax.legend()
            ax.set_xlabel(r"$100/T$" + ulbl(ux))
            ax.set_ylabel(r"$G$" + ulbl(uy))
            ax.set_yscale('log')
            res_image = os.path.join(RES_DIR, f"{chip}_{pair}_stabtime_comparison.png")
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
    dh = a.DataHandler()

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
            "P1-P4": {
                'nums': np.arange(335, 366),
                'lowtemp_highbias_num': 334,
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
                    [70, 71, 91, 92, 111, 112, 113, 132, 133, 152, 153, 172, 173, 192],
                ]),
                'lowtemp_highbias_num': 5,
            },
        },
        'SQC1': {
            'P2-P3': {
                'nums': np.arange(46, 77),
                'lowtemp_highbias_num': 45,
                'noise_level': 2*ur.pA,
            },
            'P1-P4': {
                'nums': np.concatenate([
                    np.arange(80, 95),
                    np.arange(96, 110),
                ]),
                'noise_level': 2*ur.pA,
            },
            'P2-P9': {
                'nums': np.arange(114, 146),
                'noise_level': 1*ur.pA,
            },
            'P9-P15': {
                'nums': np.arange(147, 178),
                'lowtemp_highbias_num': 146,
                'noise_level': 1*ur.pA,
            },
        },
    }
    # data_dict_ = data_dict
    data_dict_ = {}
    low_temperature(dh, data_dict_)
    nsites(data_dict)
    high_temperature(dh, data_dict)

    data_dict_ = {k: v for k, v in data_dict.items() if k in ['SQC1']}
    interface(dh, data_dict_)

    data_dict = {
        'SPC2': {
            'P15-P16': {
                'nums': {
                    '10s': [300, 302, 304, 306, 308, 310, 312, 315, 317, 319, 327, 329, 331],
                    '20s': [301, 303, 305, 307, 309, 311, 313, 316, 318, 320, 328, 330, 332],
                }
            }
        }
    }
    # compare_stabtime(dh, data_dict)
