# -*- coding: utf-8 -*-
"""analysis.

API for analysing data.
"""


import re
import json
import numpy as np
from scipy.odr import Model, RealData, ODR
import pandas as pd
from pint import UnitRegistry
from uncertainties import unumpy as unp


ur = UnitRegistry()
ur.setup_matplotlib()
MODES = {'2p', '4p'}
NUM_FIBERS = 60
FIBER_RADIUS = (25 * ur.nanometer).plus_minus(3)


def load_data(path, order='vi'):
    """Read data from files."""
    # load data
    d = pd.read_excel(path)
    data = {}
    if order == "iv":
        data['current'] = np.array(d.loc[0]) * ur.A
        data['voltage'] = np.array(d.loc[1]) * ur.V
    elif order == "vi":
        data['voltage'] = np.array(d.loc[0]) * ur.V
        data['current'] = np.array(d.loc[1]) * ur.A
    else:
        raise ValueError(f"Unknown order '{order}'.")
    data['time'] = np.array(d.loc[2]) * ur.s
    data['temperature'] = np.array(d.loc[3]) * ur.K
    return data


def load_properties(path):
    """Load properties file."""
    raw_prop = json.load(open(path))
    quantity_regex = r"([0-9\.]*)\s*([a-zA-Z]*)"
    prop = {}
    for name in raw_prop:
        prop[name] = {}
        for k in ['input', 'output', 'temperature']:
            m = re.match(quantity_regex, raw_prop[name][k], re.M)
            unit = ur[m.group(2)]
            magnitude = float(m.group(1))
            if k == 'temperature':
                prop[name]['temperature'] = (magnitude * unit).to(ur.K)
            elif unit.is_compatible_with(ur.V):
                prop[name]['voltage'] = (magnitude * unit).to(ur.V)
                if k == 'input':
                    prop[name]['order'] = 'vi'
                else:
                    prop[name]['order'] = 'iv'
            elif unit.is_compatible_with(ur.A):
                prop[name]['current'] = (magnitude * unit).to(ur.A).magnitude
        prop[name]['pair'] = raw_prop[name]['pair']
        prop[name]['segment'] = ChipParameters.pair_to_segment(raw_prop[name]['pair'])
        prop[name]['comment'] = raw_prop[name]['comment']
    return prop


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


def qlist_to_array(x):
    """Turn a quantity list in a numpy array."""
    unit = x[0].units
    return np.array(strip_units(x)) * unit


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
        """Get distance between 2 contacts."""
        if isinstance(segment, str):
            segment = self.pair_to_segment(segment)
        pad_high = np.amin(segment)
        pad_low = np.amax(segment)

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

    @staticmethod
    def segment_to_pair(segment):
        return '-'.join([f"P{n}" for n in segment])

    @staticmethod
    def pair_to_segment(pair):
        return np.array(pair.replace('P', '').split('-'), int)
