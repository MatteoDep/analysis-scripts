# -*- coding: utf-8 -*-
"""res_vs_length.

analyse data to plot conductivity over length.
"""


import os
import re
from glob import glob
import json
import numpy as np
from scipy.optimize import curve_fit
import uncertainties as u
from uncertainties import unumpy as unp
from matplotlib import pyplot as plt
import pandas as pd


NUM_FIBERS = 60
FIBER_RADIUS = u.ufloat(25e-9, 1e-12)


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
        elif isinstance(arg, np.ndarray):
            out += " {}".format(arg)
        elif isinstance(arg, u.UFloat):
            out += " {:.2u}".format(arg)
        else:
            out += " {:.3f}".format(arg)
    if inner:
        return out
    else:
        print(out)


def linear_fit(x, y):
    """Calculate 1D llinear fit."""
    for d in (x, y):
        if isinstance(d, list):
            if len(d) < 1:
                return None, None
            if not isinstance(d[0], u.UFloat):
                d = unp.uarray(d, 1e-9*np.ones(len(d)))

    def line(x, a, b):
        """Linear model."""
        return a*x + b

    sigma_x = unp.std_devs(x)
    sigma_y = unp.std_devs(y)
    if sigma_x.any() or sigma_y.any():
        sigma = np.sqrt(sigma_x**2 + sigma_y**2)
    else:
        sigma = None

    popt, pcov = curve_fit(line, unp.nominal_values(x), unp.nominal_values(y), sigma=sigma)
    perr = np.sqrt(np.diag(pcov))
    a = u.ufloat(popt[0], perr[0])
    b = u.ufloat(popt[1], perr[1])
    return a, b


class ChipParameters():
    """Define chip parameters."""

    def __init__(self, config=None):
        """
        Create object and load configuration.

        :param config: configuration toml file or dictionary.
        """
        # default values
        self.sigma = 0
        self.name = None
        self.layout = None

        if config is not None:
            self.load(config)

    def set(self, parameter, value):
        """Set parameter to value."""
        setattr(self, parameter, value)

    def load(self, config):
        """
        Set attributes from configuration dictionary.

        :param config: configuration toml file or dictionary.
        """
        if isinstance(config, str):
            config = json.load(open(config))
        for attr_name, attr_value in config.items():
            self.set(attr_name, attr_value)

    def dump(self, path=None):
        """Dump configs to a dict."""
        config_dict = {
            a: getattr(self, a) for a in sorted(dir(self))
            if not a.startswith("__") and not callable(getattr(self, a))
        }
        if path is not None:
            with open(path, 'w') as f:
                json.dump(config_dict, f)
        return config_dict

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for key, val in self.dump().items():
            print(f"{key:30} {val}")
        print("\n")

    def get_distance(self, pad_high, pad_low):
        """Get distance between the contacts point correspondent to the pads."""
        if pad_high > pad_low:
            pad_high, pad_low = pad_low, pad_high

        count = 0
        distance = 0
        measure = False
        for item in self.layout:
            if item['type'] == "contact":
                count += 1
            # do not change the order of the if statements below
            if count == pad_low:
                break
            if measure:
                distance += item['width']
            if count == pad_high:
                measure = True

        return u.ufloat(distance, self.sigma)


class DataHandler():
    """Functions to compute various quantities."""

    QUANTITIES = ["resistance", "length", "resistivity", "conductance"]
    SIGMA = 0

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
        if mode in ("4p", "2p"):
            self.current = unp.uarray(d.loc[0], self.SIGMA)
            self.voltage = unp.uarray(d.loc[1], self.SIGMA)
        elif mode == "vi":
            self.voltage = unp.uarray(d.loc[0], self.SIGMA)
            self.current = unp.uarray(d.loc[1], self.SIGMA)
        else:
            raise ValueError(f"Unknown mode '{mode}'.")
        self.time = unp.uarray(d.loc[2], self.SIGMA)
        self.temperature = unp.uarray(d.loc[3], self.SIGMA)

    def inspect(self):
        """Plot input/output over time and output over input."""
        current_label = r"current [A]"
        voltage_label = r"voltage [V]"
        time_label = r"time [s]"

        fig, axs = plt.subplots(3, 1)
        plt.tight_layout()
        fig.suptitle(self.name)

        time = unp.nominal_values(self.time)
        voltage = unp.nominal_values(self.voltage)
        current = unp.nominal_values(self.current)

        axs[0].plot(time, current, 'o-')
        axs[0].set_xlabel(time_label)
        axs[0].set_ylabel(current_label)

        axs[1].plot(time, voltage, 'o-')
        axs[1].set_xlabel(time_label)
        axs[1].set_ylabel(voltage_label)

        axs[2].plot(current, voltage, 'o-')
        axs[2].set_xlabel(current_label)
        axs[2].set_ylabel(voltage_label)

        plt.show()

    def get(self, quantity):
        """Call function to compute a certain quantity."""
        try:
            q = getattr(self, quantity)
            if q is None:
                q = getattr(self, "_compute_" + quantity)()
        except AttributeError:
            err = f"Computing {quantity} is not implemented.\navailable quantities are: {self.QUANTITIES}"
            raise ValueError(err)
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
        self.length = self.cp.get_distance(self.pad_high, self.pad_low)
        return self.length

    def _compute_resistance(self):
        """Compute resistance from current voltage curve."""
        coeffs = linear_fit(self.current, self.voltage)
        self.resistance = coeffs[0]
        return self.resistance


def main(data_dir, couples, verbosity=1):
    """Start analysis."""
    to_compute = np.unique(np.array(couples).flatten())
    quantity_dict = {q: [] for q in to_compute}

    pattern = os.path.join(data_dir, '4p*.xlsx')
    for path in np.sort(glob(pattern)):
        # get pad numbers from path
        name = os.path.splitext(os.path.basename(path))[0]
        m = re.match(r'4p_.*_([0-9]*)-([0-9]*)', name, re.M)

        # initialize data handler
        dh = DataHandler(cp, path, mode="4p", pad_high=m.group(1), pad_low=m.group(2))

        p("---", name, "---")
        for q in quantity_dict:
            value = dh.get(q)
            quantity_dict[q].append(value)
            if verbosity > 0:
                p(f"{q}:", value)

        if verbosity > 1:
            dh.inspect()

    # create arrays from lists
    for q in quantity_dict:
        quantity_dict[q] = np.array(quantity_dict[q])

    for qx, qy in couples:
        fig, ax = plt.subplots()
        ax.set_title(f"{qy} over {qx}")
        x = unp.nominal_values(quantity_dict[qx])
        y = unp.nominal_values(quantity_dict[qy])
        dx = unp.std_devs(quantity_dict[qx])
        dy = unp.std_devs(quantity_dict[qy])
        ax.errorbar(x, y, xerr=dx, yerr=dy, fmt='o')
        ax.set_xlabel(qx)
        ax.set_ylabel(qy)
        res_image = os.path.join(res_dir, f"{qy}_vs_{qx}.png")
        plt.savefig(res_image)
        if verbosity > 0:
            plt.show()

    coeffs = linear_fit(quantity_dict["length"], quantity_dict["resistance"])
    resistivity = coeffs[0] * NUM_FIBERS * FIBER_RADIUS
    offset = coeffs[1]
    conductance = 1 / resistivity

    if verbosity > 0:
        p()
        p("offset:", offset)
        p("resistivity:", resistivity)
        p("conductance:", conductance)

    return 0


if __name__ == "__main__":
    # chip = "SKC7"
    chip = "SIC1x"
    experiment = "4p_room-temp"
    chip_dir = os.path.join("data", chip)
    data_dir = os.path.join(chip_dir, experiment)
    res_dir = os.path.join("results", chip, experiment)
    verbosity = 1

    # load chip parameters
    cp = ChipParameters(os.path.join(chip_dir, chip + ".json"))
    # create results dir if it doesn't exist
    os.makedirs(res_dir, exist_ok=True)
    # define plots to produce
    couples = [
        ["length", "resistance"],
        ["length", "conductance"],
    ]
    main(data_dir, couples, verbosity=verbosity)
