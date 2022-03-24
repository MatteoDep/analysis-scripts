# -*- coding: utf-8 -*-
"""analyse.

Collection of analysis routines.
"""


import os
import numpy as np
from matplotlib import pyplot as plt
import analysis as a


EXPERIMENT = 'light_effect'


def get_temperature_dependence(names, props, **get_conductance_kwargs):
    """Compute conductance over temperature with and without light."""
    global data_dir
    # load data properties
    conductance = [] * a.ur.S
    temperature = [] * a.ur.K
    for name in names:
        path = os.path.join(data_dir, name + '.xlsx')
        data = a.load_data(path, order=props[name]['order'])

        conductance = np.append(
            conductance,
            a.get_conductance(data, **get_conductance_kwargs)
        )
        temperature = np.append(
            temperature,
            a.get_mean_std(data['temperature'])
        )
    return temperature, conductance


def cond_temp_dependence(names, props, prefix="", hb_window=[22, 24], lb_window=[-1.5, 1.5]):
    """Main analysis routine.
    names: dictionary like {'lb':{ls1: [name1, name2, ...], ls2: [...], ...}, 'hb': {...}}.
    """
    global data_dir, res_dir

    # compute temperature and conductance
    conductance = {}
    temperature = {}
    for r, bw in zip(names.keys(), [lb_window, hb_window]):
        temperature[r] = {}
        conductance[r] = {}
        for ls in names[r]:
            temperature[r][ls], conductance[r][ls] = get_temperature_dependence(
                names[r][ls], props, bias_window=bw
            )

    fig, ax = plt.subplots()
    labels = ["zero bias", f"high bias ({np.mean(hb_window)}V)"]
    for r, label in zip(names.keys(), labels):
        for ls in names[r]:
            x = 100 / temperature[r][ls]
            y = conductance[r][ls]
            x_, dx = a.separate_measurement(x)
            y_, dy = a.separate_measurement(y)
            ax.errorbar(x_, y_, xerr=dx, yerr=dy, fmt='o', label=f"{ls}, " + label)
    ax.set_title("Light Effect Conductance")
    ax.set_xlabel(f"100/T [${x.units:~L}$]")
    ax.set_ylabel(f"G [${y.units:~L}$]")
    ax.set_xlim(a.get_lim(x_))
    ax.set_yscale('log')
    plt.legend()
    plt.tight_layout()
    res_image = os.path.join(res_dir, prefix + "conductance_vs_temperature.png")
    plt.savefig(res_image, dpi=300)

    fig, ax = plt.subplots()
    for r, label in zip(names.keys(), labels):
        x = temperature[r]['dark']
        x_, dx = a.separate_measurement(x)
        y = conductance[r]['100%'] / conductance[r]['dark']
        y_, dy = a.separate_measurement(y)
        ax.errorbar(x_, y_, xerr=dx, yerr=dy, fmt='o', label=label)
    ax.set_title("Light Effect Relative Conductance")
    ax.set_xlabel(f"T [${x.units:~L}$]")
    ax.set_ylabel("G_light/G_dark")
    ax.set_xlim(a.get_lim(x_))
    plt.legend()
    plt.tight_layout()
    res_image = os.path.join(res_dir, prefix + "relative.png")
    plt.savefig(res_image, dpi=300)


def time_constant(names, switches):
    """Compute time constants of dark-light switches."""
    global data_dir, res_dir
    directions = ['heating', 'cooling']
    tau = {d: [] * a.ur.s for d in directions}
    temperature = [] * a.ur.K
    for i, name in enumerate(names):
        path = os.path.join(data_dir, name + '.xlsx')
        data = a.load_data(path, order=props[name]['order'])
        dt = np.diff(switches[i])[0]
        for d, switch in zip(directions, switches[i]):
            tau[d] = np.append(
                tau[d],
                a.get_tau(data, [switch, switch + dt])
            )
        temperature = np.append(
            temperature,
            a.get_mean_std(data['temperature'])
        )

    fig, ax = plt.subplots()
    for d in directions:
        x = temperature
        x_, dx = a.separate_measurement(x)
        y = tau[d]
        y_, dy = a.separate_measurement(y)
        ax.errorbar(x_, y_, xerr=dx, yerr=dy, fmt='o', label=d)
    ax.set_title("Light Effect Time Constant")
    ax.set_xlabel(f"T [${x.units:~L}$]")
    ax.set_ylabel(r"$\tau$ " + f"[${y.units:~L}$]")
    ax.set_xlim(a.get_lim(x_))
    plt.legend()
    plt.tight_layout()
    res_image = os.path.join(res_dir, "tau.png")
    plt.savefig(res_image, dpi=300)


if __name__ == "__main__":
    chip = "SIC1x"
    pair = "P27-P28"
    data_dir = os.path.join('data', 'SIC1x')
    res_dir = os.path.join('results', EXPERIMENT, 'SIC1x')
    prop_path = os.path.join(data_dir, 'properties.csv')

    os.makedirs(res_dir, exist_ok=True)
    cp = a.ChipParameters(os.path.join("chips", chip + ".json"))
    segment = cp.pair_to_segment(pair)
    length = cp.get_distance(segment)
    print(f"Analyzing {chip} {pair} of length {length}.")

    # load properties
    props = a.load_properties(prop_path)

    names = ['SIC1x_802', 'SIC1x_809', 'SIC1x_814', 'SIC1x_819', 'SIC1x_824', 'SIC1x_829', 'SIC1x_834']
    switches = [
        [67.9, 85.3],
        [69.8, 93.2],
        [50.3, 68.1],
        [46.8, 63.9],
        [48.0, 87.6],
        [72.4, 115.6],
        [62.6, 104.3],
    ]
    time_constant(names, switches)

#     names = {
#         'lb': {
#             '100%': [
#                 'SIC1x_725',
#                 'SIC1x_729',
#                 'SIC1x_733',
#                 'SIC1x_737',
#                 'SIC1x_741',
#             ],
#             'dark': [
#                 'SIC1x_726',
#                 'SIC1x_730',
#                 'SIC1x_734',
#                 'SIC1x_738',
#                 'SIC1x_742',
#             ],
#         },
#         'hb': {
#             '100%': [
#                 'SIC1x_711',
#                 'SIC1x_715',
#                 'SIC1x_719',
#                 'SIC1x_723',
#                 'SIC1x_727',
#                 'SIC1x_731',
#                 'SIC1x_735',
#                 'SIC1x_739',
#             ],
#             'dark': [
#                 'SIC1x_712',
#                 'SIC1x_716',
#                 'SIC1x_720',
#                 'SIC1x_724',
#                 'SIC1x_728',
#                 'SIC1x_732',
#                 'SIC1x_736',
#                 'SIC1x_740',
#             ],
#         },
#     }
#     cond_temp_dependence(names, props, prefix="thermal_paste-")

#     names = {
#         'lb': {
#             '100%': [
#                 'SIC1x_812',
#                 'SIC1x_817',
#                 'SIC1x_822',
#                 'SIC1x_827',
#                 'SIC1x_832',
#             ],
#             'dark': [
#                 'SIC1x_813',
#                 'SIC1x_818',
#                 'SIC1x_823',
#                 'SIC1x_828',
#                 'SIC1x_833',
#             ],
#         },
#         'hb': {
#             '100%': [
#                 'SIC1x_800',
#                 'SIC1x_803',
#                 'SIC1x_805',
#                 'SIC1x_810',
#                 'SIC1x_815',
#                 'SIC1x_820',
#                 'SIC1x_825',
#                 'SIC1x_830',
#             ],
#             'dark': [
#                 'SIC1x_801',
#                 'SIC1x_804',
#                 'SIC1x_806',
#                 'SIC1x_811',
#                 'SIC1x_816',
#                 'SIC1x_821',
#                 'SIC1x_826',
#                 'SIC1x_831',
#             ],
#         },
#     }
#     cond_temp_dependence(names, props, prefix="nothing-")
