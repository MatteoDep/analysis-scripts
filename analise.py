# -*- coding: utf-8 -*-
"""analyse.

Collection of analysis routines.
"""


import os
from glob import glob
import pprint
import numpy as np
from matplotlib import pyplot as plt
from pint import UnitRegistry
from analysis import DataHandler, ChipParameters, make_compatible, fit, separate_measurement, \
                     strip_units, MODES, FIBER_RADIUS, NUM_FIBERS


ur = UnitRegistry()
ur.setup_matplotlib()
pp = pprint.PrettyPrinter()


def compute_quantities(data_dir, quantities, verbosity=1):
    """Start analysis."""
    qd = {mode: {} for mode in MODES}
    for mode in MODES:
        qd[mode] = {q: [] for q in quantities}
    units = {}

    pattern = os.path.join(data_dir, '*.xlsx')
    for path in np.sort(glob(pattern)):
        # initialize data handler
        dh = DataHandler(cp, path)

        # get properties
        name = dh.name
        mode = dh.mode

        if verbosity > 0:
            print(f"\rLoading {name}", flush=True, end="")
        for q in quantities:
            value = dh.get(q)
            qd[mode][q].append(value)
            if q not in units:
                units[q] = value.units

        if verbosity > 1:
            dh.inspect(res_dir)
    if verbosity > 0:
        print("\rFinished loading data.", flush=True)

    # create arrays from lists
    for mode in MODES:
        for q in quantities:
            qd[mode][q] = np.array(strip_units(qd[mode][q])) * units[q]

    return qd


if __name__ == "__main__":
    # chip = "SJC9"
    # chip = "SKC6"
    # chip = "SLC7"
    # chip = "SKC7"
    chip = "SIC1x"
    experiment = "room-temp_characterization"
    chip_dir = os.path.join("data", chip)
    data_dir = os.path.join(chip_dir, experiment)
    res_dir = os.path.join("results", chip, experiment)
    verbosity = 2   # 0, 1 or 2

    # load chip parameters
    cp = ChipParameters(os.path.join(chip_dir, chip + ".json"))
    # create results dir if it doesn't exist
    os.makedirs(res_dir, exist_ok=True)
    # define plots to produce
    quantities = ["length", "resistance"]

    qd = compute_quantities(data_dir, quantities, verbosity=verbosity)
    qd = make_compatible(qd, 'length')
    if verbosity > 1:
        pp.pprint(qd)
    # order quantities and assign better units
    indices = np.argsort(qd['2p']["length"])
    for m in MODES:
        qd[m]["length"] = qd[m]["length"][indices].to("micrometer")
        qd[m]["resistance"] = qd[m]["resistance"][indices].to("megaohm")

    coeffs, reslen_model = fit(qd['4p']["length"], qd['4p']["resistance"])
    resistivity = (coeffs[0] * NUM_FIBERS * np.pi * FIBER_RADIUS**2).to('ohm * cm')
    conductivity = (1 / resistivity).to('S / cm')

    if verbosity > 0:
        print(f"\nresistivity: {resistivity:~P}")
        print(f"conductivity: {conductivity:~P}")

    def get_lim(*arrays):
        """Get limits for array a."""
        max_a = 0
        min_a = 0
        for array in arrays:
            max_a = np.nanmax([max_a, np.nanmax(array)])
            min_a = np.nanmin([min_a, np.nanmin(array)])
        padding = 0.1 * (max_a - min_a)
        lim = [min_a - padding, max_a + padding]
        return np.array(lim)

    # generate axis
    fig, axs = plt.subplots(len(MODES), 1)
    if len(MODES) == 1:
        axs = [axs]

    # plot quantities
    qx, qy = "length", "resistance"
    for ax, mode in zip(axs, MODES):
        title = f"{qy} vs {qx} ({mode})"
        ax.set_title(title)

        x = qd[mode][qx]
        y = qd[mode][qy]
        x_, dx = separate_measurement(x)
        y_, dy = separate_measurement(y)
        if np.isnan(x_*y_).all():
            print(f"Warning: Missing data. Skipping {title}.")
            ax.remove()
            continue

        ax.errorbar(x_, y_, xerr=dx, yerr=dy, fmt='o', label=f'{mode} data')
        ax.set_xlabel(f"{qy} [${x.units:~L}$]")
        ax.set_ylabel(f"{qy} [${y.units:~L}$]")
        res_image = os.path.join(res_dir, f"{qy}_vs_{qx}.png")
        ax.set_xlim(get_lim(x_))

        if mode == '4p':
            x = np.insert(x_, 0, 0)
            y1 = reslen_model(x)
            ax.plot(x, y1, '-', label=f'{mode} fit')
            ax.set_ylim(get_lim(y_, y1))
        else:
            ax.set_ylim(get_lim(y_))

        plt.tight_layout()
        plt.savefig(res_image)

    fig, ax = plt.subplots()
    contact_resistance = qd['2p'][qy] - qd['4p'][qy]
    x = qd['4p'][qx]
    y = contact_resistance
    x_, dx = separate_measurement(x)
    y_, dy = separate_measurement(y)

    ax.errorbar(x_, y_, xerr=dx, yerr=dy, fmt='o', label=f'{mode} data')
    ax.set_xlabel(f"{qy} [${x.units:~L}$]")
    ax.set_ylabel(f"{qy} [${y.units:~L}$]")
    res_image = os.path.join(res_dir, f"contact-{qy}_vs_{qx}.png")
    ax.set_xlim(get_lim(x_))
    ax.set_ylim(get_lim(y_))

    plt.tight_layout()
    plt.savefig(res_image)
