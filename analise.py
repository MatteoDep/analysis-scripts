# -*- coding: utf-8 -*-
"""res_vs_length.

analyse data to plot conductivity over length.
"""


import os
import re
from glob import glob
import json
import pprint
import numpy as np
from scipy.odr import Model, RealData, ODR
from matplotlib import pyplot as plt
import pandas as pd
from pint import UnitRegistry
from uncertainties import unumpy as unp


ur = UnitRegistry()
ur.setup_matplotlib()
pp = pprint.PrettyPrinter()
MODES = {'2p', '4p'}
NUM_FIBERS = 60
FIBER_RADIUS = (25 * ur.nanometer).plus_minus(3)


def make_compatible(qd, q_ref):
    """Set q_ref arrays to be the same for each mode and add nans to make the length of other quantities the same."""
    # build ref
    first = True
    for mode in MODES:
        if first:
            ref = qd[mode][q_ref]
            first = False
        else:
            for elem in qd[mode][q_ref]:
                if elem not in ref:
                    np.append(ref, elem)
    ref = np.sort(ref)

    # adjust sizes
    for mode in MODES:
        for elem in ref:
            if elem not in qd[mode][q_ref]:
                qd[mode][q_ref] = np.append(qd[mode][q_ref], elem)
                for q in set(qd[mode].keys()) - set([q_ref]):
                    qd[mode][q] = np.append(qd[mode][q], np.nan)

        # sort arrays
        indices = np.argsort(qd[mode][q_ref])
        for q in qd[mode].keys():
            qd[mode][q] = qd[mode][q][indices]

    return qd


def separate_measurement(x, y, strip_nan=False):
    """Get parameters to feed ODR from Quantities."""
    x_ = unp.nominal_values(strip_units(x))
    y_ = unp.nominal_values(strip_units(y))
    dx = unp.std_devs(strip_units(x))
    dy = unp.std_devs(strip_units(y))

    # strip nan values
    if strip_nan:
        indices = np.isnan(x_*y_) == 0
        x_ = x_[indices]
        y_ = y_[indices]
        dx = dx[indices]
        dy = dy[indices]

    if (dx == 0).all():
        dx = None
    if (dy == 0).all():
        dy = None
    return x_, y_, dx, dy


def strip_units(a):
    """Strip unit from Quantity x."""
    if isinstance(a, np.ndarray):
        s = np.array([strip_units(x) for x in a])
    if isinstance(a, list):
        s = [strip_units(x) for x in a]
    elif hasattr(a, "magnitude"):
        s = a.magnitude
    else:
        s = a
    return s


def get_units(model_name, x, y):
    """Get units of model parameters."""
    if model_name == "linear":
        u = [(y / x).units, y.units]
    else:
        raise ValueError(f"{model_name} model is not implemented.")
        return None
    return u


def get_model(model_name, b, x):
    """Define model function."""
    if model_name == "linear":
        res = b[0]*x + b[1]
    else:
        raise ValueError(f"{model_name} model is not implemented.")
        return None
    return res


def get_estimate(model_name, data):
    """Parameters estimation."""
    if model_name == "linear":
        b = [data.y[0]/data.x[0], data.y[0]]
    else:
        raise ValueError(f"{model_name} model is not implemented.")
        return None
    return b


def fit(x, y, model_name="linear", debug=False):
    """Calculate 1D llinear fit."""
    def model(b, x):
        return get_model(model_name, b, x)

    def estimate(data):
        return get_estimate(model_name, data)

    odr_model = Model(model, estimate=estimate)

    x_, y_, dx, dy = separate_measurement(x, y, strip_nan=True)
    if debug:
        print("x values:", x_)
        print("y values:", y_)
        print("x errors:", dx)
        print("y errors:", dy)
    data = RealData(x_, y_, sx=dx, sy=dy)
    odr = ODR(data, odr_model)
    res = odr.run()
    if debug:
        print("res.beta:", res.beta)
        print("res.sd_beta:", res.sd_beta)

    # build result as physical quantities
    units = get_units(model_name, x, y)
    coeffs = []
    for i in range(len(res.beta)):
        coeffs.append((res.beta[i] * units[i]).plus_minus(res.sd_beta[i]))
    if debug:
        print("coeffs:", coeffs)

    return coeffs, lambda x, b=res.beta: model(b, x)


class ChipParameters():
    """Define chip parameters."""

    def __init__(self, config=None):
        """
        Create object and load configuration.

        :param config: configuration toml file or dictionary.
        """
        # default values
        self.sigma = 0.1 * ur.micrometer
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

    def get_distance(self, segment):
        """Get distance between the contacts point correspondent to the pads."""
        pad_high = min(segment)
        pad_low = max(segment)

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

        return (distance * ur['m']).plus_minus(self.sigma)


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
        if self.mode not in MODES:
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

    def inspect(self):
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
        coeffs, iv_model = fit(self.current, self.voltage)
        b = [coeffs[0].value, coeffs[1].value]
        self.iv_model = lambda x: iv_model(x, b=b)
        self.resistance = coeffs[0]
        return self.resistance


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
            dh.inspect()
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
    chip = "SKC7"
    # chip = "SIC1x"
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
    pp.pprint(qd)
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
        x_, y_, dx, dy = separate_measurement(x, y)
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
    x_, y_, dx, dy = separate_measurement(x, y)

    ax.errorbar(x_, y_, xerr=dx, yerr=dy, fmt='o', label=f'{mode} data')
    ax.set_xlabel(f"{qy} [${x.units:~L}$]")
    ax.set_ylabel(f"{qy} [${y.units:~L}$]")
    res_image = os.path.join(res_dir, f"contact-{qy}_vs_{qx}.png")
    ax.set_xlim(get_lim(x_))
    ax.set_ylim(get_lim(y_))

    plt.tight_layout()
    plt.savefig(res_image)
