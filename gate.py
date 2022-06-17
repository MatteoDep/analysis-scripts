# -*- coding: utf-8 -*-
"""gate.

Analize gate dependence.
"""


import os
import numpy as np
from matplotlib import pyplot as plt

from analysis import ur
import analysis as a


EXPERIMENT = os.path.splitext(os.path.basename(__file__))[0]


def get_gate_dependence(names, fields, delta_field, noise_level=0.5*ur.pA):
    global dh, res_dir, plot_ivs
    time_win = [0.25, 0.75]

    gate = [] * ur.V
    conductance = [[] * ur.S for f in fields]
    max_j = len(fields)
    for i, name in enumerate(names):
        print(f"\rProcessing {i+1} of {len(names)}.", end='', flush=True)
        dh.load(name)
        gate0 = dh.get_gate()
        gate = np.append(gate, gate0)
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
    print()
    return gate, conductance


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

            print("Chip {}, Pair {} of length {}.".format(chip, pair, dh.cps[chip].get_distance(pair)))

            if 'noise_level' in data_dict[chip]:
                nl = data_dict[chip][pair]['noise_level']
            else:
                nl = noise_level

            names_dict = {}
            for name in names:
                key = str(dh.props[name]['temperature']).replace(' ', '')
                if key not in names_dict:
                    names_dict[key] = []
                names_dict[key].append(name)

            for temp_key in names_dict:
                print(f'Temperature {temp_key}:')
                names = names_dict[temp_key]
                fields = fields[(fields + delta_field) * dh.get_length() < dh.props[names[0]]['bias']]

                if 'noise_level' in data_dict[chip][pair]:
                    nl = data_dict[chip][pair]['noise_level']
                else:
                    nl = noise_level
                gate, conductance = get_gate_dependence(names, fields, delta_field, noise_level=nl)

                # plot gate dependence
                x = gate
                x_, dx, ux = a.separate_measurement(x)
                fig, ax = plt.subplots(figsize=(12, 9))
                cols = plt.cm.viridis(fields.magnitude / np.max(fields.magnitude))
                for i, field in enumerate(fields):
                    y = conductance[i]
                    y_, dy, uy = a.separate_measurement(y)
                    ax.errorbar(x_, y_, xerr=dx, yerr=dy, fmt='--o', zorder=i, c=cols[i], label=fr'$E_b = {field}$')
                ax.legend()
                ax.set_title(f"Conductance ({chip} {pair} {temp_key})")
                ax.set_xlabel(fr"$V_G$ [${ux:~L}$]")
                ax.set_ylabel(fr"$G$ [${uy:~L}$]")
                res_image = os.path.join(res_dir, f"{chip}_{pair}_{temp_key}_gate_dep.png")
                fig.savefig(res_image, dpi=300)
                plt.close()


if __name__ == "__main__":
    data_dict = {
        'SPC2': {
            "P16-P17": {
                'nums': np.concatenate([
                    np.arange(419, 432),
                    np.arange(433, 483),
                ])
            },
        },
    }

    main(data_dict, noise_level=0.5 * ur.pA)
