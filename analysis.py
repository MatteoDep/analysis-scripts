# -*- coding: utf-8 -*-
"""analysis.

API for analysing data.
"""


import json
import os
import numpy as np
from scipy.odr import Model, RealData, ODR
import pandas as pd
from matplotlib import pyplot as plt
import pint
from uncertainties import unumpy as unp
from matplotlib import rcParams


rcParams.update({'figure.autolayout': True})

ur = pint.UnitRegistry()
ur.setup_matplotlib()
Q = ur.Quantity
ur.default_format = "~P"

MODES = {'2p', '4p'}
NUM_FIBERS = 60
FIBER_RADIUS = (25 * ur.nanometer).plus_minus(3)


DEFAULT_UNITS = {
    'voltage': ur.V,
    'current': ur.A,
    'time': ur.s,
    'temperature': ur.K,
}


# HANDLE DATA

class DataHandler:
    """
    Loads data and builds quantities arrays.
    """

    def __init__(self, data_dir, **props_kwargs):
        self.data_dir = data_dir
        self.props = self.load_properties(**props_kwargs)
        self.qd = {
            'conductance': ur.S,
            'tau': ur.s,
            'temperature': ur.K,
            'length': ur.um,
        }
        self.cp = {}
        pass

    def load_properties(self, path=None, names=None):
        """Load properties file."""
        if path is None:
            path = os.path.join(self.data_dir, 'properties.csv')
        df = pd.read_csv(path, sep='\t', index_col=0)
        if names is None:
            names = df.index
        self.props = {}
        oldcols = ['input', 'output']
        for name in names:
            self.props[name] = {}
            self.props[name]['order'] = ''
            self.props[name]['temperature'] = Q(df.loc[name, 'temperature'])
            for k in oldcols:
                q = Q(df.loc[name, k])
                if q.is_compatible_with(ur.V):
                    self.props[name]['order'] += 'v'
                    self.props[name]['voltage'] = q
                elif q.is_compatible_with(ur.A):
                    self.props[name]['order'] += 'i'
                    self.props[name]['current'] = q
                else:
                    raise ValueError(f"Unrecognized units {q.units} of name {name}")
        return self.props

    def load_data(self, name):
        """Read data from files."""
        # load data
        path = os.path.join(self.data_dir, name + '.xlsx')
        df = pd.read_excel(path, skiprows=[5, 6]).T
        self.data = {}
        keys = [o.replace('i', 'current').replace('v', 'voltage') for o in self.props[name]['order']] \
            + ['time', 'temperature']
        for i, key in enumerate(keys):
            self.data[key] = Q(df[i].to_numpy(), DEFAULT_UNITS[key])
        return self.data

    def get_conductance(self, method="auto", bias_window=None, only_return=False, debug=False):
        """Calculate conductance."""
        if method == "auto":
            if bias_window is None:
                method = "fit"
            else:
                if np.multiply(*bias_window) < 0:
                    method = "fit"
                else:
                    method = "average"

        # prepare data
        cond = self.data['voltage'] != 0
        if only_return:
            time_window = np.array([0.25, 0.75]) * self.data['time'][-1]
            cond *= is_between(self.data['time'], time_window)
        if bias_window is not None:
            if not hasattr(bias_window, 'units'):
                bias_window = bias_window * DEFAULT_UNITS['voltage']
            cond *= is_between(self.data['voltage'], bias_window)

        # calculate conductance
        if method == "fit":
            coeffs, model = fit_linear(self.data['voltage'][cond], self.data['current'][cond], debug=debug)
            conductance = coeffs[0]
        elif method == "average":
            conductance = get_mean_std(self.data['current'][cond]/self.data['voltage'][cond])
        else:
            raise ValueError(f"Unrecognized method {method}.")

        return conductance

    def get_temperature(self):
        self.temperature = get_mean_std(self.data['temperature'])
        return self.temperature

    def get(self, names, quantities, qkwargs=None):
        res = [[] * self.qd[q] for q in quantities]
        if qkwargs is None:
            qkwargs = [{} for q in quantities]
        for name in names:
            for i, q in enumerate(quantities):
                self.load_data(name)
                res[i] = np.append(res[i], getattr(self, f'get_{q}')(*qkwargs[i]))
        return res


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

        return (distance * ur.m).plus_minus(self.sigma)

    @staticmethod
    def segment_to_pair(segment):
        return '-'.join([f"P{n}" for n in segment])

    @staticmethod
    def pair_to_segment(pair):
        return [int(x) for x in pair.replace('P', '').split('-')]


# MEASUREMENTS HELP FUNCTIONS

def measurement_is_equal(x, y):
    """Compare 2 independent measurements."""
    x_, dx, _ = separate_measurement(x)
    y_, dy, _ = separate_measurement(y)
    if x_ == y_ and dx == dy:
        return True
    return False


def measurement_is_present(x, arr):
    """Check if x is present in arr."""
    for y in arr:
        if measurement_is_equal(x, y):
            return True
    return False


def separate_measurement(x):
    """Get parameters to feed ODR from Quantities."""
    x_ = unp.nominal_values(strip_units(x))
    dx = unp.std_devs(strip_units(x))
    if len(x_.shape) == 0:
        x_ = float(x_)
        dx = float(dx) if dx != 0 else None
    elif (dx == 0).all():
        dx = None
    return x_, dx, x.units


def nominal_values(x):
    x_, _, u = separate_measurement(x)
    return x_ * u


def std_devs(x):
    _, dx, u = separate_measurement(x)
    return dx * u


# TODO change to just x.magnitude
def strip_units(x):
    """Strip unit from Quantity x."""
    if hasattr(x, "magnitude"):
        s = x.magnitude
    elif isinstance(x, np.ndarray):
        s = np.array([strip_units(x_) for x_ in x])
    elif isinstance(x, list):
        s = [strip_units(x_) for x_ in x]
    else:
        s = x
    return s


# ANALYSIS

def get_mean_std(x):
    """Calculate average with std."""
    return np.mean(x).plus_minus(np.std(x))


def fit_exponential(x, y, offset=None, ignore_err=False, debug=False):
    """Calculate 1D exponential fit (y = exp(a*x + b) + offset)."""
    if offset is not None:
        y1 = y - offset
    else:
        y1 = y
    x_, dx, ux = separate_measurement(x)
    y1_, dy1, uy1 = separate_measurement(y1)
    coeffs, _ = fit_linear((x_, dx, ux), (np.log(y1_), dy1 / y1_, ur.dimensionless),
                           already_separated=True, ignore_err=ignore_err, debug=debug)

    def model_fcn(b, x, offset):
        return np.exp(b[0] * x + b[1]) + offset

    beta = [separate_measurement(c)[0] for c in coeffs]
    if offset is None:
        offset_ = 0
    else:
        offset_, _, _ = separate_measurement(offset)

    return coeffs, lambda x, b=beta, offset=offset_: model_fcn(b, x, offset)


def fit_linear(x, y, ignore_err=False, already_separated=False, debug=False):
    """Calculate 1D linear fit (y = a*x + b)."""
    if already_separated:
        x_, dx, ux = x
        y_, dy, uy = y
    else:
        x_, dx, ux = separate_measurement(x)
        y_, dy, uy = separate_measurement(y)
    if ignore_err:
        dx = None
        dy = None
    if dx is not None:
        dx[dx == 0] = np.nan
    if dy is not None:
        dy[dy == 0] = np.nan
    data = RealData(x_, y_, sx=dx, sy=dy)

    m = (y_[-1] - y_[0]) / (x_[-1] - x_[0])
    q = y_[0] - m * x_[0]

    def model_fcn(b, x):
        return b[0] * x + b[1]

    odr_model = Model(model_fcn)
    odr = ODR(data, odr_model, beta0=[m, q])
    res = odr.run()

    def are_res_dummy(res):
        return res.beta[0] == m or res.beta[1] == q

    if not ignore_err and are_res_dummy(res):
        print('Fit using errors failed... doing fit without errors.')
        ignore_err = True
        data = RealData(x_, y_)
        odr = ODR(data, odr_model, beta0=[m, q])
        res = odr.run()
    if ignore_err and are_res_dummy(res):
        raise RuntimeError("Fit Failed!")

    # build result as physical quantities
    units = [uy/ux, uy]
    coeffs = []
    for i in range(len(res.beta)):
        coeffs.append((res.beta[i] * units[i]).plus_minus(res.sd_beta[i]))

    if debug:
        res.pprint()
        plt.figure()
        plt.errorbar(x_, y_, xerr=dx, yerr=dy, fmt='o')
        plt.plot(x_, model_fcn(res.beta, x_))
        plt.show()

    return coeffs, lambda x, b=res.beta: model_fcn(b, x)


# OTHER HELP FUNCTIONS

def include_origin(ax, axis='xy'):
    """Fix limits to include origin."""
    for a in axis:
        lim = getattr(ax, f"get_{a}lim")()
        d = np.diff(lim)[0] / 20
        lim = [min(lim[0], -d), max(lim[1], d)]
        getattr(ax, f"set_{a}lim")(lim)
    return ax


def is_between(x, window):
    return np.logical_and(x > window[0], x < window[1])
