# -*- coding: utf-8 -*-
"""analyse.

Collection of analysis routines.
"""


import os
import re
from glob import glob
import pprint
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pint import UnitRegistry
import analysis as a


ur = UnitRegistry()
ur.setup_matplotlib()
pp = pprint.PrettyPrinter()


def compute_quantities(data_dir, quantities, chip=None, res_dir="results",
                       verbosity=1):
    """Start analysis."""
    qd = {mode: {} for mode in a.MODES}
    for mode in a.MODES:
        qd[mode] = {q: [] for q in quantities}
    units = {}

    pattern = os.path.join(data_dir, '*.xlsx')
    for path in np.sort(glob(pattern)):
        # initialize data handler
        dh = DataHandler(chip, path)

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
    for mode in a.MODES:
        for q in quantities:
            qd[mode][q] = np.array(a.strip_units(qd[mode][q])) * units[q]

    return qd


def length_characterization(chip, verbosity=0):
    """Compute resistance over length characterization."""
    experiment = "length_characterization"
    data_dir = os.path.join("data", experiment, chip)
    res_dir = os.path.join("results", experiment, chip)
    res_ivs_dir = os.path.join(res_dir, "ivs")

    # load chip parameters
    chip = a.ChipParameters(os.path.join("chips", chip + ".json"))
    # create results dir if it doesn't exist
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(res_ivs_dir, exist_ok=True)
    # define plots to produce
    quantities = ["length", "resistance"]

    qd = compute_quantities(data_dir, quantities, chip=chip,
                            res_dir=res_ivs_dir, verbosity=verbosity)
    qd = a.make_compatible(qd, 'length')
    if verbosity > 1:
        pp.pprint(qd)
    # order quantities and assign better units
    indices = np.argsort(qd['2p']["length"])
    for m in a.MODES:
        qd[m]["length"] = qd[m]["length"][indices].to("micrometer")
        qd[m]["resistance"] = qd[m]["resistance"][indices].to("megaohm")

    coeffs, reslen_model = a.fit(qd['4p']["length"], qd['4p']["resistance"])
    resistivity = (coeffs[0] * a.NUM_FIBERS * np.pi * a.FIBER_RADIUS**2).to('ohm * cm')
    conductivity = (1 / resistivity).to('S / cm')

    if verbosity > 0:
        print(f"\nresistivity: {resistivity:~P}")
        print(f"conductivity: {conductivity:~P}")

    # generate axis
    fig, axs = plt.subplots(len(a.MODES), 1)
    if len(a.MODES) == 1:
        axs = [axs]

    # plot quantities
    qx, qy = "length", "resistance"
    for ax, mode in zip(axs, a.MODES):
        title = f"{qy} vs {qx} ({mode})"
        ax.set_title(title)

        x = qd[mode][qx]
        y = qd[mode][qy]
        x_, dx = a.separate_measurement(x)
        y_, dy = a.separate_measurement(y)
        if np.isnan(x_*y_).all():
            print(f"Warning: Missing data. Skipping {title}.")
            ax.remove()
            continue

        ax.errorbar(x_, y_, xerr=dx, yerr=dy, fmt='o', label=f'{mode} data')
        ax.set_xlabel(f"{qy} [${x.units:~L}$]")
        ax.set_ylabel(f"{qy} [${y.units:~L}$]")
        res_image = os.path.join(res_dir, f"{qy}_vs_{qx}.png")
        ax.set_xlim(a.get_lim(x_))

        if mode == '4p':
            x = np.insert(x_, 0, 0)
            y1 = reslen_model(x)
            ax.plot(x, y1, '-', label=f'{mode} fit')
            ax.set_ylim(a.get_lim(y_, y1))
        else:
            ax.set_ylim(a.get_lim(y_))

        plt.tight_layout()
        plt.savefig(res_image)

    fig, ax = plt.subplots()
    contact_resistance = qd['2p'][qy] - qd['4p'][qy]
    x = qd['4p'][qx]
    y = contact_resistance
    x_, dx = a.separate_measurement(x)
    y_, dy = a.separate_measurement(y)

    ax.errorbar(x_, y_, xerr=dx, yerr=dy, fmt='o')
    ax.set_xlabel(f"{qy} [${x.units:~L}$]")
    ax.set_ylabel(f"{qy} [${y.units:~L}$]")
    ax.set_xlim(a.get_lim(x_))
    ax.set_ylim(a.get_lim(y_))
    res_image = os.path.join(res_dir, f"contact-{qy}_vs_{qx}.png")

    plt.tight_layout()
    plt.savefig(res_image)


class DataHandler():
    """Functions to compute various quantities."""

    def __init__(self, chip_parameters, path=None, **kwargs):
        """Declare quantities."""
        self.cp = chip_parameters

        self.quantities = [a.split("_compute_")[1] for a in sorted(dir(self)) if a.startswith("_compute_")]

        if path is not None:
            self.load(path, **kwargs)

    def load(self, path):
        """Read data from files."""
        self.name = os.path.splitext(os.path.basename(path))[0]

        # get properties from name
        regex = r'([vi]{2})([24]p).*_([0-9]+)-([0-9]+)$'
        m = re.match(regex, self.name, re.M)
        if m is None:
            raise ValueError(f"File {path} does not match regex {regex}. Please stick to the naming convention.")
        order = m.group(1)
        self.mode = m.group(2)
        if self.mode not in a.MODES:
            raise ValueError(f"Unknown mode '{self.mode}'.")
        self.segment = (int(m.group(3)), int(m.group(4)))

        # load data
        d = pd.read_excel(path)
        if order == "iv":
            self.current = np.array(d.loc[0]) * ur['A']
            self.voltage = np.array(d.loc[1]) * ur['V']
        elif order == "vi":
            self.voltage = np.array(d.loc[0]) * ur['V']
            self.current = np.array(d.loc[1]) * ur['A']
        else:
            raise ValueError(f"Unknown order '{order}'.")
        self.time = np.array(d.loc[2]) * ur['s']
        self.temperature = np.array(d.loc[3]) * ur['K']

    def inspect(self, res_dir):
        """Plot input/output over time and output over input."""
        fig, axs = plt.subplots(3, 1)
        plt.tight_layout()
        fig.suptitle(self.name)

        axs[0].plot(self.time, self.current, 'o')
        axs[1].plot(self.time, self.voltage, 'o')
        axs[2].plot(self.current, self.voltage, 'o')

        if hasattr(self, "iv_model"):
            axs[2].plot(self.current, self.iv_model(self.current))

        filename = os.path.join(res_dir, self.name + ".png")
        plt.savefig(filename)

    def get(self, quantity):
        """Call function to compute a certain quantity."""
        if hasattr(self, quantity):
            q = getattr(self, quantity)
        elif hasattr(self, "_compute_" + quantity):
            q = getattr(self, "_compute_" + quantity)()
        else:
            err = f"Computing {quantity} is not implemented.\navailable quantities are: {self.quantities}"
            raise AttributeError(err)
        return q

    def _compute_length(self):
        """Compute cable length between electrodes."""
        self.length = self.cp.get_distance(self.segment)
        return self.length

    def _compute_resistance(self):
        """Compute resistance from current voltage curve."""
        coeffs, iv_model = a.fit(self.current, self.voltage)
        b = [coeffs[0].value, coeffs[1].value]
        self.iv_model = lambda x: iv_model(x, b=b)
        self.resistance = coeffs[0]
        return self.resistance


if __name__ == "__main__":
    # chip = "SJC9"
    # chip = "SKC6"
    # chip = "SLC7"
    # chip = "SKC7"
    chip = "SIC1x"
    verbosity = 2   # 0, 1 or 2
    length_characterization(chip, verbosity=verbosity)
