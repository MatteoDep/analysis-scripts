# -*- coding: utf-8 -*-
"""analysis.

API for analysing data.
"""


import os
import re
import json
import numpy as np
from scipy.odr import Model, RealData, ODR
from matplotlib import pyplot as plt
import pandas as pd
from pint import UnitRegistry
from uncertainties import unumpy as unp


ur = UnitRegistry()
ur.setup_matplotlib()
MODES = {'2p', '4p'}
NUM_FIBERS = 60
FIBER_RADIUS = (25 * ur.nanometer).plus_minus(3)


def measurement_is_equal(x, y):
    """Compare 2 independent measurements."""
    x_, dx = separate_measurement(x)
    y_, dy = separate_measurement(y)
    if x_ == y_ and dx == dy:
        return True
    return False


def measurement_is_present(x, arr):
    """Check if x is present in arr."""
    x_, dx = separate_measurement(x)
    arr_, darr = separate_measurement(arr)
    for y in arr:
        if measurement_is_equal(x, y):
            return True
    return False


def make_compatible(qd, q_ref):
    """Set q_ref arrays to be the same for each mode and add nans to make the length of other quantities the same."""
    # build ref
    first = True
    for mode in MODES:
        if first:
            ref = qd[mode][q_ref]
            first = False
        else:
            for x in qd[mode][q_ref]:
                if not measurement_is_present(x, ref):
                    np.append(ref, x)
    ref = np.sort(ref)

    # adjust sizes
    for mode in MODES:
        for x in ref:
            if not measurement_is_present(x, qd[mode][q_ref]):
                qd[mode][q_ref] = np.append(qd[mode][q_ref], x)
                for q in set(qd[mode].keys()) - set([q_ref]):
                    qd[mode][q] = np.append(qd[mode][q], np.nan)

        # sort arrays
        indices = np.argsort(qd[mode][q_ref])
        for q in qd[mode].keys():
            qd[mode][q] = qd[mode][q][indices]

    return qd


def separate_measurement(x):
    """Get parameters to feed ODR from Quantities."""
    x_ = unp.nominal_values(strip_units(x))
    dx = unp.std_devs(strip_units(x))

    if (dx == 0).all():
        dx = None
    return x_, dx


def strip_nan(*args):
    """Strip value if is NaN in any of the arguments."""
    args = [x for x in args if x is not None]
    prod = True
    for x in args:
        prod *= x
    indices = np.isnan(prod) == 0
    for x in args:
        x = x[indices]


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

    x_, dx = separate_measurement(x)
    y_, dy = separate_measurement(y)
    strip_nan(x_, y_, dx, dy)
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
        coeffs, iv_model = fit(self.current, self.voltage)
        b = [coeffs[0].value, coeffs[1].value]
        self.iv_model = lambda x: iv_model(x, b=b)
        self.resistance = coeffs[0]
        return self.resistance
