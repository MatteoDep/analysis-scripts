# -*- coding: utf-8 -*-
"""analyse.

Collection of analysis routines.
"""


import os
import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt
from analysis import ur, ulbl, fmt
import analysis as a


EXPERIMENT = os.path.splitext(os.path.basename(__file__))[0]


def get_tau(data, time_window, const_estimate_time=5*ur.s):
    """Calculate time constant tau."""
    t = data['time']
    i = data['current']
    if not hasattr(time_window, 'units'):
        time_window *= ur.s
    t_start = time_window[0]
    t_end = time_window[1]
    i_start = np.mean(data['current'][a.is_between(data['time'], [t_start - const_estimate_time, t_start])])
    i_end = np.mean(data['current'][a.is_between(data['time'], [t_end - const_estimate_time, t_end])])
    amp = i_start - i_end
    offset = i_end

    cond = a.is_between(t, [t_start, t_end])
    t_cut = t[cond]
    i_cut = i[cond]
    sign = (-1) ** int(amp < 0)
    t_end_decay = t_cut[np.nonzero(sign * i_cut < sign * (amp * np.exp(-3) + offset))[0][0]]

    cond = a.is_between(t, [t_start, t_end_decay])
    x = t[cond]
    y = np.log(sign * (i[cond] - offset).m) * ur.dimensionless

    coeffs, model = a.fit_linear(x, y)
    tau = - 1 / coeffs[0]

    return tau


def get_temperature_dependence(dh, names, bias):
    """Get the temperature dependence of the conductance for different field values."""
    temperature = [] * ur.K
    conductance = [] * ur.nS
    method = 'fit' if bias == 0*ur.V else 'average'
    delta_bias = 1 * ur.V

    for i, name in enumerate(names):
        dh.load(name)
        temperature_i = dh.get_temperature(method='average')
        temperature = np.append(temperature, temperature_i)
        bias_win = [bias - delta_bias, bias + delta_bias]
        mask = dh.get_mask(bias_win=bias_win)
        if mask.any():
            conductance_i = dh.get_conductance(method=method, mask=mask)
        else:
            conductance_i = np.nan
        conductance = np.append(conductance, conductance_i)
    return temperature, conductance


def temp_dependence(dh, names):
    """Main analysis routine.
    names: dictionary like {'lb':{ls1: [name1, name2, ...], ls2: [...], ...}, 'hb': {...}}.
    """
    bias = {
        'hb': 23 * ur.V,
        'lb': 0 * ur.V,
    }

    # compute temperature and conductance
    conductance = {}
    temperature = {}
    for key in names:
        temperature[key] = {}
        conductance[key] = {}
        for r in names[key]:
            temperature[key][r] = {}
            conductance[key][r] = {}
            for ls in names[key][r]:
                temperature[key][r][ls], conductance[key][r][ls] = get_temperature_dependence(dh, names[key][r][ls], bias=bias[r])

    fig, ax = plt.subplots()
    bias_labels = {r: fmt(b) for r, b in bias.items()}
    markers = ['o', 's', '^']
    for i, key in enumerate(names):
        r = 'hb'
        for ls in names[key][r]:
            x_, dx, ux = a.separate_measurement(100 / temperature[key][r][ls])
            y_, dy, uy = a.separate_measurement(conductance[key][r][ls])
            ax.errorbar(x_, y_, xerr=dx, yerr=dy, marker=markers[i], label=f"{key}, {ls}")
    ax.set_title("Dark and Light Conductance")
    ax.set_xlabel(r"$\frac{{100}}{{T}}$" + ulbl(ux))
    ax.set_ylabel(r"$G$" + ulbl(uy))
    ax.set_yscale('log')
    plt.legend(loc=(0.38, 0.55))
    res_image = os.path.join(res_dir, "all-temperature_dep.pdf")
    fig.savefig(res_image, dpi=100)
    plt.close()

    fig, ax = plt.subplots()
    for i, key in enumerate(names):
        r = 'hb'
        ls = 'dark'
        x_, dx, ux = a.separate_measurement(100 / temperature[key][r][ls])
        y_, dy, uy = a.separate_measurement(conductance[key][r][ls])
        ax.errorbar(x_, y_, xerr=dx, yerr=dy, marker=markers[i], label=key)
    ax.set_title("Dark Conductance")
    ax.set_xlabel(r"$\frac{{100}}{{T}}$" + ulbl(ux))
    ax.set_ylabel(r"$G$" + ulbl(uy))
    ax.set_yscale('log')
    plt.legend(loc=(0.38, 0.55))
    res_image = os.path.join(res_dir, "all-temperature_dep_dark.pdf")
    fig.savefig(res_image, dpi=100)
    plt.close()

    for key in names:
        fig, ax = plt.subplots()
        for r in names[key]:
            for ls in names[key][r]:
                x_, dx, ux = a.separate_measurement(100 / temperature[key][r][ls])
                y_, dy, uy = a.separate_measurement(conductance[key][r][ls])
                ax.errorbar(x_, y_, xerr=dx, yerr=dy, marker='o', label=f"{ls}, {bias_labels[r]}")
        ax.set_title(f"Temperature Dependence ({key.replace('_', ' ')})")
        ax.set_xlabel(r"$\frac{{100}}{{T}}$" + ulbl(ux))
        ax.set_ylabel(r"$G$" + ulbl(uy))
        ax.set_yscale('log')
        plt.legend()
        res_image = os.path.join(res_dir, f"{key}-temperature_dep.pdf")
        fig.savefig(res_image, dpi=100)
        plt.close()

        fig, ax = plt.subplots()
        for r in names[key]:
            x_, dx, ux = a.separate_measurement(temperature[key][r]['dark'])
            y_, dy, uy = a.separate_measurement(np.divide(*[conductance[key][r][s] for s in names[key][r]]))
            ax.errorbar(x_, y_, xerr=dx, yerr=dy, marker='o', label=bias_labels[r])
        ax.set_title(f"Relative Temperature Dependence ({key.replace('_', ' ')})")
        ax.set_xlabel(r"T" + ulbl(ux))
        ax.set_ylabel(r"$G_{light}/G_{dark}$")
        plt.legend()
        res_image = os.path.join(res_dir, f"{key}-temperature_dep_rel.pdf")
        fig.savefig(res_image, dpi=100)
        plt.close()


def time_constant(dh, names, switches):
    """Compute time constants of dark-light switches."""

    directions = ['fall', 'rise']
    tau = {}
    temperature = {}
    for key in names:
        if not hasattr(switches[key], 'units'):
            switches[key] *= ur.s
        tau[key] = {d: [] * ur.s for d in directions}
        temperature[key] = [] * ur.K
        for i, name in enumerate(names[key]):
            data = dh.load(name)
            dt = np.diff(switches[key][i])[0]
            for d, switch in zip(directions, switches[key][i]):
                time_window = np.append([switch], switch + dt)
                tau[key][d] = np.append(tau[key][d], get_tau(data, time_window))
            temperature[key] = np.append(temperature[key], dh.get_temperature(method='average'))

    for key in names:
        fig, ax = plt.subplots()
        for d in directions:
            x, dx, ux = a.separate_measurement(temperature[key])
            y, dy, uy = a.separate_measurement(tau[key][d])
            ax.errorbar(x, y, xerr=dx, yerr=dy, marker='o', label=d)
        ax.set_title(f"Time Constant ({key.replace('_', ' ')})")
        ax.set_xlabel(r"$T$" + ulbl(ux))
        ax.set_ylabel(r"$\tau$" + ulbl(uy))
        ax.set_yscale('log')
        plt.legend()
        res_image = os.path.join(res_dir, f"{key}-tau.pdf")
        fig.savefig(res_image, dpi=100)
        plt.close()


def color_dependence(dh, names, time_window):
    """Plot the color_dependence."""
    labels = ['blue', 'red', 'green', 'white', 'dark']
    colors = ['blue', 'red', 'green', 'yellow', 'black']

    for key in names:
        if not hasattr(time_window[key], 'units'):
            time_window[key] *= ur.s
        fig, ax = plt.subplots()
        for i, name in enumerate(names[key]):
            dh.load(name)
            mask = dh.get_mask(time_win=time_window[key], only_return=False)
            dh.plot(ax, mode='i/t', color=colors[i], label=labels[i], mask=mask)
        ax.set_title("Color Dependence ({}, {}, {})".format(
            key.replace('_', ' '), dh.prop['bias'], dh.prop['temperature']
        ))
        plt.legend()
        res_image = os.path.join(res_dir, f"{key}-color_dep.pdf")
        fig.savefig(res_image, dpi=100)
        plt.close()


def bias_dependence(dh, names, switches):
    """Compute bias dependence."""

    statuses = ['dark', 'light']
    conductance = {}
    bias = {}
    for key in names:
        if not hasattr(switches[key], 'units'):
            switches[key] *= ur.s
        conductance[key] = {s: [] * ur.nS for s in statuses}
        bias[key] = [] * ur.V
        for i, name in enumerate(names[key]):
            dh.load(name)
            dt = np.diff(switches[key][i])[0] / 2
            for d, switch in zip(statuses, switches[key][i]):
                time_win = np.append([switch], switch + dt)
                mask = dh.get_mask(time_win=time_win, only_return=False)
                conductance[key][d] = np.append(
                    conductance[key][d],
                    dh.get_conductance(method='average', mask=mask, correct_offset=False)
                )
            bias[key] = np.append(bias[key], dh.get_bias(method='average', mask=mask))
    temperature = dh.prop['temperature']

    for key in names:
        fig, ax = plt.subplots()
        for s in statuses:
            x, dx, ux = a.separate_measurement(bias[key])
            y, dy, uy = a.separate_measurement(conductance[key][s])
            ax.errorbar(x, y, xerr=dx, yerr=dy, marker='o', label=s)
        ax.set_title(f"Bias Dependence ({key.replace('_', ' ')}, {fmt(temperature)})")
        ax.set_xlabel("$V_b$" + ulbl(ux))
        ax.set_ylabel("$G$" + ulbl(uy))
        plt.legend()
        res_image = os.path.join(res_dir, f"{key}-{fmt(temperature, sep='')}-bias_dep.pdf")
        fig.savefig(res_image, dpi=100)
        plt.close()

        fig, ax = plt.subplots()
        x, dx, ux = a.separate_measurement(bias[key])
        y, dy, uy = a.separate_measurement(np.divide(*[conductance[key][s] for s in statuses[::-1]]))
        ax.errorbar(x, y, xerr=dx, yerr=dy, marker='o')
        ax.set_title(f"Relative Bias Dependence ({key.replace('_', ' ')}, {fmt(temperature)})")
        ax.set_xlabel("$V_b$" + ulbl(ux))
        ax.set_ylabel(r"$G_{light}/G_{dark}$")
        res_image = os.path.join(res_dir, f"{key}-{fmt(temperature, sep='')}-bias_dep_rel.pdf")
        fig.savefig(res_image, dpi=100)
        plt.close()


def plot_switches(dh, names, switches):

    for key in names:
        if not hasattr(switches[key], 'units'):
            switches[key] *= ur.s
        for i, name in enumerate(names[key]):
            data = dh.load(name)
            fig, ax = plt.subplots()
            dt = 1 * ur.s
            for switch in switches[key][i]:
                index0 = np.argmin(np.abs(data['time'] - (switch - dt)))
                index1 = np.argmin(np.abs(data['time'] - (switch + dt)))
                if data['current'][index0] > data['current'][index1]:
                    d = 'covered'
                    c = 'red'
                else:
                    d = 'uncovered'
                    c = 'green'
                ax.axvline(x=switch.m, c=c, label=d)
            ax = dh.plot(ax, mode='i/t')
            ax.set_title(f"Switch ({key.replace('_', ' ')}, {dh.prop['bias']}, {dh.prop['temperature']})")
            plt.legend()
            res_image = os.path.join(
                res_dir,
                "{3}-switch{2}_{0.m}{0.u}_{1.m}{1.u}.pdf".format(
                    dh.prop['bias'],
                    dh.prop['temperature'],
                    f"_{d}" if len(switches[key][i]) == 1 else "",
                    key
                )
            )
            fig.savefig(res_image, dpi=100)
            plt.close()


def plot_ivs(dh, names):
    labels = ['light', 'dark']

    for key in names:
        for i, names_ in enumerate(names[key]):
            fig, ax = plt.subplots()
            for name, label in zip(names_, labels):
                dh.load(name)
                # mask = dh.get_mask(bias_win=[-24, 24]*ur.V)
                ax = dh.plot(ax, notraw=True)
            ax.set_title(f"IV ({key.replace('_', ' ')}, {dh.prop['temperature']})")
            res_image = os.path.join(
                res_dir,
                "{1}-iv_{0.m}{0.u}.pdf".format(
                    dh.prop['temperature'],
                    key
                )
            )
            fig.savefig(res_image, dpi=100)
            plt.close()


def heat_balance():
    """Estimate expected temperature change with simple heat balance model."""
    # compute temperature from graphic estimations
    # the plot all-temperature_dep is used
    def get_temperature(x):
        """Get temperature from inkscape measurement."""
        return 10 * (182.4 - 53.05)/(x - 53.05) * ur.K

    def eps(temp):
        """Emissivity from Constancio(2020)."""
        return 2.45e-3 * temp / ur.K + 0.116 * ur['']

    area_sensor = (25.4 * ur.mm)**2
    thickness_glass = 1 * ur.mm
    sigma = 5.67e-8 * ur['W * m^-2 * K^-4']
    mass_chip = 1 * ur.g

    phi_lamp = 0.5 * 132 * ur.mW / area_sensor
    phi_env = 20 * ur.mW / area_sensor
    print("phi_env:", phi_env)
    phi_env = (0.07 * sigma * (294 * ur.K)**4).to('mW / mm^2')
    print("phi_env:", phi_env)

    temp_meas = {
        'thermal_paste': {
            'dark': get_temperature(178.3),
            'light': get_temperature(177.4),
        },
        'nothing': {
            'dark': get_temperature(176.8),
            'light': get_temperature(176.3),
        },
        'glass': {
            'dark': get_temperature(176.2),
            'light': get_temperature(175.2),
        },
    }
    temp_est = {
        'thermal_paste': {
            'light': get_temperature(168.3),
        },
        'nothing': {
            'dark': get_temperature(164.7),
            'light': get_temperature(149.4),
        },
        'glass': {
            'dark': get_temperature(67.8),
            'light': get_temperature(67.5),
        },
    }
    print("temp_meas:", temp_meas)
    print("temp_est:", temp_est)
    # Tmeas: {'nothing': {'dark': 14.739393939393938, 'light': 14.799188640973629},
    #           'glass': {'dark': 14.811205846528624, 'light': 14.932460090053214}}
    # Test: {'nothing': {'dark': 16.33676668159427, 'light': 18.930980799169692},
    #          'glass': {'dark': 123.66101694915254, 'light': 126.22837370242212}}

    def get_h(contact, mode):
        if mode == 'dark':
            phi = phi_env  # * 0.6
        elif mode == 'light':
            phi = phi_lamp
        else:
            raise ValueError(f"Invalide mode {mode}.")
        temp_est_ = temp_est[contact][mode]
        temp_meas_ = temp_meas[contact][mode]
        return (phi - 2 * sigma * eps(temp_est_) * temp_est_**4) / (temp_est_ - temp_meas_)

    h = {m: get_h('nothing', m) for m in ['dark', 'light']}
    print("h:", h)
    k = {m: get_h('glass', m) * thickness_glass for m in ['dark', 'light']}
    print("k:", k)

    def heat_cap(temp, debye_temp=631*ur.K):
        """Computes the specific heat capacity using Debye's model."""
        rel_temp = temp / debye_temp
        integral = integrate(lambda x: x**4 * np.exp(x) / (np.exp(x) - 1)**2, 0, rel_temp)
        spec_heat_cap = 9 * rel_temp**3 * integral * ur.k_B * ur.N_A
        return mass_chip * spec_heat_cap / 28.085 * ur['g / mol']


if __name__ == "__main__":
    chip = "SIC1x"
    res_dir = os.path.join('results', EXPERIMENT)
    os.makedirs(res_dir, exist_ok=True)

    # load properties
    dh = a.DataHandler()
    dh.load_chip(chip)

    do = []
    # do.append('temp_dependence')
    # do.append('time_constant')
    # do.append('color_dependence')
    # do.append('bias_dependence')
    # do.append('plot_switches')
    # do.append('plot_ivs')
    # do.append('heat_balance')

    if 'temp_dependence' in do:
        print('temp_dependence')
        names = {
            'thermal_paste': {
                'lb': {
                    'light': [
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
                    'light': [
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
            },
            'nothing': {
                'lb': {
                    'light': [
                        'SIC1x_812',
                        'SIC1x_817',
                        'SIC1x_822',
                        'SIC1x_827',
                        'SIC1x_832',
                    ],
                    'dark': [
                        'SIC1x_813',
                        'SIC1x_818',
                        'SIC1x_823',
                        'SIC1x_828',
                        'SIC1x_833',
                    ],
                },
                'hb': {
                    'light': [
                        'SIC1x_800',
                        'SIC1x_803',
                        'SIC1x_805',
                        'SIC1x_810',
                        'SIC1x_815',
                        'SIC1x_820',
                        'SIC1x_825',
                        'SIC1x_830',
                    ],
                    'dark': [
                        'SIC1x_801',
                        'SIC1x_804',
                        'SIC1x_806',
                        'SIC1x_811',
                        'SIC1x_816',
                        'SIC1x_821',
                        'SIC1x_826',
                        'SIC1x_831',
                    ],
                },
            },
        }
        temp_dependence(dh, names)

    if 'time_constant' in do:
        print('time_constant')
        names = {
            'nothing': [
                'SIC1x_802',
                'SIC1x_809',
                'SIC1x_814',
                'SIC1x_819',
                'SIC1x_824',
                'SIC1x_829',
                'SIC1x_834',
            ],
        }
        switches = {
            'nothing': [
                [67.9, 85.3],
                [69.8, 93.2],
                [50.3, 68.1],
                [46.8, 63.9],
                [48.0, 87.6],
                [72.4, 115.6],
                [62.6, 104.3],
            ] * ur.s,
        }
        time_constant(dh, names, switches)

    if 'color_dependence' in do:
        print('color_dependence')
        names = {
            'thermal_paste': ['SIC1x_772', 'SIC1x_773', 'SIC1x_774', 'SIC1x_775', 'SIC1x_776'],
            'glass': ['SIC1x_788', 'SIC1x_789', 'SIC1x_790', 'SIC1x_791', 'SIC1x_792'],
            'nothing': ['SIC1x_835', 'SIC1x_836', 'SIC1x_837', 'SIC1x_838', 'SIC1x_839'],
        }
        time_windows = {
            'thermal_paste': [95, 130],
            'glass': [40, 70],
            'nothing': [95, 130],
        }
        color_dependence(dh, names, time_windows)

    if 'bias_dependence' in do:
        print('bias_dependence')
        names = {
            'thermal_paste': [
                'SIC1x_747',
                'SIC1x_748',
                'SIC1x_749',
                'SIC1x_750',
                'SIC1x_746',
                'SIC1x_751',
                'SIC1x_752',
            ],
        }
        switches = {
            'thermal_paste': [
                [30.4, 62.1],
                [30.9, 43.4],
                [39.5, 57.0],
                [43.6, 57.1],
                [64.8, 91.8],
                [74.3, 99.0],
                [109.0, 134.1],
            ] * ur.s,
        }
        bias_dependence(dh, names, switches)

    if 'plot_switches' in do:
        print('plot_switches')
        names = {
            'thermal_paste': [
                'SIC1x_746',
                'SIC1x_747',
                'SIC1x_748',
                'SIC1x_749',
                'SIC1x_750',
                'SIC1x_751',
                'SIC1x_752',
            ],
            'glass': [
                'SIC1x_786',
                'SIC1x_787',
            ],
            'nothing': [
                'SIC1x_802',
                'SIC1x_809',
                'SIC1x_814',
                'SIC1x_819',
                'SIC1x_824',
                'SIC1x_829',
                'SIC1x_834',
            ],
        }
        switches = {
            'thermal_paste': [
                [64.8, 91.8],
                [30.4, 62.1],
                [30.9, 43.4],
                [39.5, 57.0],
                [43.6, 57.1],
                [74.3, 99.0],
                [109.0, 134.1],
            ] * ur.s,
            'glass': [
                [97.0],
                [62.9],
            ] * ur.s,
            'nothing': [
                [67.9, 85.3],
                [69.8, 93.2],
                [50.3, 68.1],
                [46.8, 63.9],
                [48.0, 87.6],
                [72.4, 115.6],
                [62.6, 104.3],
            ] * ur.s,
        }
        plot_switches(dh, names, switches)

    if 'plot_ivs' in do:
        print('plot_ivs')
        names = {
            'thermal_paste': [
                ['SIC1x_711',  'SIC1x_712'],
                ['SIC1x_715',  'SIC1x_716'],
                ['SIC1x_719',  'SIC1x_720'],
                ['SIC1x_723',  'SIC1x_724'],
                ['SIC1x_727',  'SIC1x_728'],
                ['SIC1x_731',  'SIC1x_732'],
                ['SIC1x_735',  'SIC1x_736'],
                ['SIC1x_739',  'SIC1x_740'],
            ],
            'glass': [
                ['SIC1x_785', 'SIC1x_784'],
            ],
            'nothing': [
                ['SIC1x_800', 'SIC1x_801'],
                ['SIC1x_803', 'SIC1x_804'],
                ['SIC1x_805', 'SIC1x_806'],
                ['SIC1x_810', 'SIC1x_811'],
                ['SIC1x_815', 'SIC1x_816'],
                ['SIC1x_820', 'SIC1x_821'],
                ['SIC1x_825', 'SIC1x_826'],
                ['SIC1x_830', 'SIC1x_831'],
            ],
        }
        plot_ivs(dh, names)

    if 'heat_balance' in do:
        print('heat_balance')
        heat_balance()
