# -*- coding: utf-8 -*-
"""analyse.

analyse data to plot conductivity over length.
"""


import os
import numpy as np
from uncertainties import unumpy as unp
from matplotlib import pyplot as plt
from analysis import Analysis, Chip, Quantity
from analysis import Visualise as v


Quantity.load_config("quantities.ini")
NUM_FIBERS = 60
FIBER_RADIUS = Quantity(25e-9, 'length', 1e-10)
AREA = NUM_FIBERS * np.pi * FIBER_RADIUS**2
MODES = ['2p', '4p']


if __name__ == "__main__":
    # chip = "SLC7"
    # chip = "SKC7"
    chip = "SIC1x"
    experiment = "4p_room-temp"
    chip_dir = os.path.join("data", chip)
    data_dir = os.path.join(chip_dir, experiment)
    res_dir = os.path.join("results", chip, experiment)
    verbosity = 1   # 0, 1 or 2

    # load chip parameters
    chip = Chip(os.path.join(chip_dir, chip + ".json"))
    # create results dir if it doesn't exist
    os.makedirs(res_dir, exist_ok=True)
    # define plots to produce
    # couples = [
    #     ["length", "resistance"],
    # ]
    # all_modes = MODES
    # quantities = np.unique(np.array(couples).flatten())
    quantities = ["length", "resistance"]

    a = Analysis(data_dir, chip, verbosity=verbosity)
    qd, segments = a.compute_quantities(quantities)

    resistivity, resistivity_model = a.compute_resistivity(qd['4p']["length"], qd['4p']["resistance"])
    conductivity = 1 / resistivity

    if verbosity > 0:
        print("\nresistivity:", resistivity)
        print("conductivity:", conductivity)

    # create dictionary with available modes for each quantity
    # exclude = {'2p': ['conductivity'],
    #            '4p': []}
    # modes = {}
    # for q in quantities:
    #     modes[q] = []
    #     for mode in all_modes:
    #         if q in exclude[mode]:
    #             continue
    #         elif np.isnan(unp.nominal_values(qd[mode][q])).all():
    #             print(f"Warning: Missing data. Skipping {mode}-{q}.")
    #         else:
    #             modes[q].append(mode)

    # for qx, qy in couples:
    #     # decide which modes to iterate over
    #     modes_ = set(modes[qx])-set(modes[qy])

    #     # generate axis
    #     fig, axs = plt.subplots(len(modes_), 1)
    #     if len(modes_) == 1:
    #         axs = [axs]

    #     # plot quantities
    #     for i, mode in enumerate(modes_):
    #         axs[i].set_title(f"{qy} vs {qx} ({mode})")
    #         x = unp.nominal_values(qd[mode][qx])
    #         y = unp.nominal_values(qd[mode][qy])
    #         dx = unp.std_devs(qd[mode][qx])
    #         dy = unp.std_devs(qd[mode][qy])

    #         axs[i].errorbar(
    #             x * factor[qx],
    #             y * factor[qy],
    #             xerr=dx * factor[qx],
    #             yerr=dy * factor[qy],
    #             fmt='o',
    #             label=f'{mode} data')
    #         axs[i].set_xlim(factor[qx]*get_lim(x))
    #         print(qx, qy)
    #         if qx == 'length' and qy == 'resistance' and mode == '4p':
    #             print(qx, qy)
    #             x = np.insert(x, 0, 0)
    #             y1 = resistivity_model(x)
    #             axs[i].plot(x*factor[qx], y1*factor[qy], '-', label=f'{mode} fit')
    #         if qx == 'length' and qy == 'conductivity':
    #             print(qx, qy)
    #             y1 = conductivity.nominal_value*np.ones(x.shape)
    #             axs[i].plot(x*factor[qx], y1*factor[qy], '-', label=f'{mode} fit')
    #         axs[i].set_ylim(factor[qy]*get_lim(y, y1))

    #         axs[i].set_xlabel(label[qx])
    #         axs[i].set_ylabel(label[qy])
    #         axs[i].legend()
    #     res_image = os.path.join(res_dir, f"{qy}_vs_{qx}.png")
    #     plt.tight_layout()
    #     plt.savefig(res_image)

    # if '2p' in modes and '4p' in modes:
    #     # contact_resistance = []
    #     # for i, segment in enumerate(segments['2p']):
    #     #     contact_resistance.append(qd['2p']['resistance'][i] - qd['4p']['resistance'][i])
    #     # contact_resistance = np.array(contact_resistance)
    #     contact_resistance = qd['2p']['resistance'] - qd['4p']['resistance']

    #     fig, ax = plt.subplots()
    #     ax.set_title("2p-4p resistance vs length")
    #     x = unp.nominal_values(qd['2p']['length'])
    #     y = unp.nominal_values(contact_resistance)
    #     dx = unp.std_devs(qd['2p']['length'])
    #     dy = unp.std_devs(contact_resistance)
    #     ax.errorbar(
    #         x * factor['length'],
    #         y * factor['resistance'],
    #         xerr=dx * factor['length'],
    #         yerr=dy * factor['resistance'],
    #         fmt='o')
    #     ax.set_xlim(factor['length']*get_lim(qd['2p']['length']))
    #     ax.set_ylim(factor['resistance']*get_lim(qd['2p']['resistance']))
    #     ax.set_xlabel(label['length'])
    #     ax.set_ylabel(label['resistance'])
    #     res_image = os.path.join(res_dir, f"{'4p-2p-resistance'}_vs_{'length'}.png")
    #     plt.tight_layout()
    #     plt.savefig(res_image)
