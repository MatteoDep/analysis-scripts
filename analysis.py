# -*- coding: utf-8 -*-
"""analysis.

API for analysing data.
"""


import json
import os
import numpy as np
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

NUM_FIBERS = 60
FIBER_RADIUS = (25 * ur.nanometer).plus_minus(3)


DEFAULT_UNITS = {
    'voltage': ur.V,
    'current': ur.A,
    'time': ur.s,
    'temperature': ur.K,
    'conductance': ur.S,
    'length': ur.m,
}


# HANDLE DATA

class DataHandler:
    """
    Loads data and builds quantities arrays.
    """

    def __init__(self, data_dir, chips_dir='chips', **props_kwargs):
        self.data_dir = data_dir
        self.chips_dir = chips_dir
        self.props = self.load_properties(**props_kwargs)
        self.chips = {}
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
            self.props[name]['pair'] = df.loc[name, 'pair']
            self.props[name]['injection'] = df.loc[name, 'injection']
            self.props[name]['temperature'] = Q(df.loc[name, 'temperature'])
            self.props[name]['order'] = ''
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

    def load(self, name):
        """Read data from files."""
        # reset quantities
        resetnot = ['data_dir', 'chips_dir', 'props', 'chips']
        attributes = [x for x in dir(self) if not callable(getattr(self, x)) and not x.startswith('__')]
        for x in attributes:
            if x not in resetnot:
                setattr(self, x, None)
        # load data
        self.name = name
        self.chip_name = self.name.split('_')[0]
        self.prop = self.props[name]
        path = os.path.join(self.data_dir, name + '.xlsx')
        df = pd.read_excel(path, skiprows=[5, 6]).T
        self.data = {}
        keys = [o.replace('i', 'current').replace('v', 'voltage') for o in self.prop['order']] \
            + ['time', 'temperature']
        for i, key in enumerate(keys):
            self.data[key] = Q(df[i].to_numpy(), DEFAULT_UNITS[key])
        return self.data

    def get_conductance(self, method="auto", bias_win=None, time_win=None, debug=False):
        """Calculate conductance."""
        if method == "auto":
            if bias_win is None:
                method = "fit"
            else:
                if np.multiply(*bias_win) < 0:
                    method = "fit"
                else:
                    method = "average"

        # prepare data
        cond = self.data['voltage'] != 0
        if time_win is not None:
            if not hasattr(time_win, 'units'):
                if not hasattr(time_win[0], 'units'):
                    time_win = np.array(time_win) * self.data['time'][-1]
            cond *= is_between(self.data['time'], time_win)
        if bias_win is not None:
            if not hasattr(bias_win, 'units'):
                if not hasattr(bias_win[0], 'units'):
                    bias_win = bias_win * self.prop['voltage']
            cond *= is_between(self.data['voltage'], bias_win)

        # calculate conductance
        if method == "fit":
            coeffs, model = fit_linear(self.data['voltage'][cond], self.data['current'][cond], debug=debug)
            conductance = coeffs[0]
        elif method == "average":
            conductance = get_mean_std(self.data['current'][cond]/self.data['voltage'][cond])
        else:
            raise ValueError(f"Unrecognized method {method}.")

        return conductance

    def get_resistance(self, **kwargs):
        return 1 / self.get_conductance(**kwargs)

    def get_temperature(self):
        return get_mean_std(self.data['temperature'])

    def get_length(self):
        if self.chip_name not in self.chips:
            self.chips[self.chip_name] = Chip(os.path.join(self.chips_dir, self.chip_name + '.json'))
        return self.chips[self.chip_name].get_distance(self.prop['pair'])

    def plot(self, ax, mode='i/v', correct_offset=False, x_win=None, y_win=None, label=r'{prop[temperature]}',
             color=None, markersize=5, set_xy_label=True):
        ykey, xkey = [
            c.replace('v', 'voltage').replace('i', 'current') if c != 't' else 'time' for c in mode.lower().split('/')
        ]
        x = self.data[xkey]
        y = self.data[ykey]
        label = label.replace('{', '{0.').format(self)

        # correct data
        if correct_offset:
            y -= np.mean(y)
        cond = np.ones(x.shape)
        if x_win is not None:
            cond *= is_between(x, x_win)
        if y_win is not None:
            cond *= is_between(y, y_win)
        ax.scatter(x, y, label=label, c=color, s=markersize, edgecolors=None)
        if set_xy_label:
            ysym, xsym = [
                c.upper() if c in 'iv' else c for c in mode.split('/')
            ]
            ax.set_xlabel(f"${xsym}$ [${x.units}$]")
            ax.set_ylabel(f"${ysym}$ [${y.units}$]")
        return ax

    def process(self, names, instruction_dict, per_data_args={}):
        res = [[] for key in instruction_dict]
        for key in instruction_dict:
            if key not in per_data_args:
                per_data_args[key] = {}
        for i, name in enumerate(names):
            self.load(name)
            for j, key in enumerate(instruction_dict):
                if i == 0:
                    if key.startswith('get_'):
                        res[j] *= DEFAULT_UNITS[key.split('get_')[1]]
                res0 = getattr(self, key)(**per_data_args[key], **instruction_dict[key])
                res[j] = np.append(res[j], res0)
        return res


class Chip():
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
        pad1 = np.amin(segment)
        pad2 = np.amax(segment)

        count = 0
        x = 0
        x1 = None
        x2 = None
        for item in self.layout:
            if item['type'] == "contact":
                count += 1
            # do not change the order of the if statements below
            if count == pad2:
                x2 = ((x + item['width'] / 2) * ur.m).plus_minus(item['width'] / np.sqrt(12))
                break
            if count == pad1:
                x1 = ((x + item['width'] / 2) * ur.m).plus_minus(item['width'] / np.sqrt(12))
            x += item['width']

        return x2 - x1

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
    return (x_ == y_) * (dx == dy)


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
    return np.nanmean(x).plus_minus(np.nanstd(x))


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
    """Calculate 1D linear fit (y = p[0] + p[1]*x)."""
    if already_separated:
        x_, dx, ux = x
        y_, dy, uy = y
    else:
        x_, dx, ux = separate_measurement(x)
        y_, dy, uy = separate_measurement(y)
    if ignore_err:
        dx = None
        dy = None
    if dy is not None:
        dy[dy == 0] = np.nan
    if dx is not None:
        dx[dx == 0] = np.nan

    p, pcov = np.polyfit(x_, y_, 1, w=dy, cov=True)

    if dx is not None and dy is not None:
        dy_prop = np.sqrt(dy**2 + (p[1]*dx)**2)
        p, pcov = np.polyfit(x_, y_, 1, w=dy_prop, cov=True)

    dp = np.sqrt(np.diag(pcov))

    # build result as physical quantities
    units = [uy/ux, uy]
    coeffs = [(p[i] * units[i]).plus_minus(dp[i]) for i in range(2)]

    if debug:
        plt.figure()
        plt.errorbar(x_, y_, xerr=dx, yerr=dy, fmt='o')
        plt.plot(x_, p[0]*x_ + p[1])
        plt.show()

    return coeffs, lambda x, p=p: p[0]*x + p[1]


# OTHER HELP FUNCTIONS

def include_origin(ax, axis='xy'):
    """Fix limits to include origin."""
    for a in axis:
        lim = getattr(ax, f"get_{a}lim")()
        d = np.diff(lim)[0] / 20
        lim = [min(lim[0], -d), max(lim[1], d)]
        getattr(ax, f"set_{a}lim")(lim)
    return ax


def is_between(x, win):
    return np.logical_and(x > win[0], x < win[1])
