# -*- coding: utf-8 -*-
"""res_vs_length.

analyse data to plot conductivity over length.
"""


import os
import re
from glob import glob
import toml
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


NUM_FIBERS = 60
FIBER_RADIUS = 25e-9


def p(*args, inner=False):
    """Print in scientific notation."""
    out = ""
    for arg in args:
        if arg is None:
            out += " None"
        elif isinstance(arg, str):
            out += " " + arg
        elif isinstance(arg, list):
            out += " " + p(*arg, inner=True)
        else:
            out += " {:.3e}".format(arg)
    if inner:
        return out
    else:
        print(out)


def linear_fit(x, y):
    """Calculate 1D llinear fit."""
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(y, list):
        y = np.array(y)
    A = np.vstack([x, np.ones(x.shape)]).T
    slope, offset = np.linalg.lstsq(A, y, rcond=None)[0]
    return slope, offset


class ChipParameters():
    """Define chip parameters."""

    CONTACT_WIDTH = 1e-5
    BIG_CONTACT_WIDTH = 1e-4
    SPACING = 4e-5

    def __init__(self, config_dict=None):
        if config_dict is not None:
            self.load(config_dict)

    def set(self, parameter, value):
        """Set parameter to value."""
        setattr(self, parameter, value)

    def load(self, config_dict):
        """Set attributes from configuration dictionary."""
        if isinstance(config_dict, str):
            config_dict = toml.load(open(config_dict))
        for attr_name, attr_value in config_dict.items():
            self.set(attr_name, attr_value)

    def dump(self, path=None):
        """Dump configs to a dict."""
        config_dict = {
            a: getattr(self, a) for a in sorted(dir(self))
            if not a.startswith("__") and not callable(getattr(self, a))
        }
        if path is not None:
            with open(path, 'w') as f:
                toml.dump(config_dict, f)
        return config_dict

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for key, val in self.dump().items():
            print(f"{key:30} {val}")
        print("\n")


class DataHandler():
    """Functions to compute various quantities."""

    QUANTITIES = ["resistance", "length", "resistivity", "conductance"]

    def __init__(self, chip_parameters, path=None, **kwargs):
        """Declare quantities."""
        self.cp = chip_parameters

        for q in self.QUANTITIES:
            setattr(self, q, None)

        if path is not None:
            self.load(path, **kwargs)

    def load(self, path, mode, pad_high, pad_low):
        """Read data from files."""
        self.name = os.path.splitext(os.path.basename(path))[0]
        self.mode = mode
        self.pad_high = int(pad_high)
        self.pad_low = int(pad_low)

        d = pd.read_excel(path)
        self.data = {}
        if mode in ("4p", "2p"):
            self.current = d.loc[0]
            self.voltage = d.loc[1]
        elif mode == "vi":
            self.voltage = d.loc[0]
            self.current = d.loc[1]
        else:
            raise ValueError(f"Unknown mode '{mode}'.")
        self.time = d.loc[2]
        self.temperature = d.loc[3]

    def inspect(self):
        """Plot input/output over time and output over input."""
        current_label = r"current [A]"
        voltage_label = r"voltage [V]"
        time_label = r"time [s]"

        fig, axs = plt.subplots(3, 1)
        plt.tight_layout()
        fig.suptitle(self.name)

        axs[0].plot(self.time, self.current, 'o-')
        axs[0].set_xlabel(time_label)
        axs[0].set_ylabel(current_label)

        axs[1].plot(self.time, self.voltage, 'o-')
        axs[1].set_xlabel(time_label)
        axs[1].set_ylabel(voltage_label)

        axs[2].plot(self.current, self.voltage, 'o-')
        axs[2].set_xlabel(current_label)
        axs[2].set_ylabel(voltage_label)

        plt.show()

    def get(self, quantity):
        """Call function to compute a certain quantity."""
        try:
            q = getattr(self, quantity)
            if q is None:
                q = getattr(self, "_compute_" + quantity)()
        except AttributeError as e:
            print(f"Computing {quantity} is not implemented.")
            print("available quantities are:", self.QUANTITIES)
            raise AttributeError(e)
        return q

    def _compute_resistivity(self):
        """Compute resistivity from resistance and length."""
        length = self.get("length")
        resistance = self.get("resistance")
        surface = FIBER_RADIUS * NUM_FIBERS
        self.resistivity = resistance * surface / length
        return self.resistivity

    def _compute_conductance(self):
        """Compute conductance from resistivity."""
        resistivity = self.get("resistivity")
        self.conductance = 1 / resistivity
        return self.conductance

    def _compute_length(self):
        """Compute cable length between electrodes."""
        spacing_num = np.abs(self.pad_high - self.pad_low)
        big_contact_num = np.abs(int((self.pad_high-1) / 5) - int((self.pad_low-1) / 5))
        contact_num = spacing_num - 1 - big_contact_num

        self.length = (spacing_num * self.cp.SPACING) + \
            (contact_num * self.cp.CONTACT_WIDTH) + \
            (big_contact_num * self.cp.BIG_CONTACT_WIDTH)
        return self.length

    def _compute_resistance(self):
        """Compute resistance from current voltage curve."""
        self.resistance, offset = linear_fit(self.current, self.voltage)
        return self.resistance


def get_from_4p(data_dir, couples, verbose=False):
    """Start analysis."""
    to_compute = np.unique(np.array(couples).flatten())
    quantity_dict = {quantity: [] for quantity in to_compute}

    pattern = os.path.join(data_dir, '4p*.xlsx')
    for path in np.sort(glob(pattern)):
        # get pad numbers from path
        name = os.path.splitext(os.path.basename(path))[0]
        m = re.match(r'4p_.*_([0-9]*)-([0-9]*)', name, re.M)

        # initialize data handler
        dh = DataHandler(cp, path, mode="4p", pad_high=m.group(1), pad_low=m.group(2))

        if verbose:
            dh.inspect()

        p("---", name, "---")
        for q in quantity_dict:
            value = dh.get(q)
            quantity_dict[q].append(value)
            p(f"{q}:", value)

    for x, y in couples:
        fig, ax = plt.subplots()
        ax.set_title(f"{y} over {x}")
        ax.plot(quantity_dict[x], quantity_dict[y], 'o')
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        res_image = os.path.join(res_dir, f"{y}_vs_{x}.png")
        plt.savefig(res_image)

    coeff, _ = linear_fit(quantity_dict["length"], quantity_dict["resistance"])
    resistivity = coeff * NUM_FIBERS * FIBER_RADIUS
    conductance = 1 / resistivity

    p()
    p("resistivity:", resistivity)
    p("conductance:", conductance)

    return 0


if __name__ == "__main__":
    chip = "SIC1x"
    experiment = "4p_room-temp"
    chip_dir = os.path.join("data", chip)
    data_dir = os.path.join(chip_dir, experiment)
    res_dir = os.path.join("results", chip, experiment)
    verbose = False

    # load chip parameters
    cp = ChipParameters(os.path.join(chip_dir, chip + ".toml"))
    # create results dir if it doesn't exist
    os.makedirs(res_dir, exist_ok=True)
    # define plots to produce
    couples = [
        ["length", "resistance"],
        ["length", "conductance"],
    ]
    get_from_4p(data_dir, couples, verbose=verbose)
