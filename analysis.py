# -*- coding: utf-8 -*-
"""analysis.

API for analysing data.
"""


import json
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import rcParams
from uncertainties import unumpy as unp
import pint
import warnings


rcParams.update({'figure.autolayout': True})

ur = pint.UnitRegistry()
ur.setup_matplotlib()
ur.default_format = ".2f~P"

NUM_FIBERS = 60
FIBER_RADIUS = (25 * ur.nanometer).plus_minus(3)


DEFAULT_UNITS = {
    'bias': ur.V,
    'gate': ur.V,
    'current': ur.A,
    'time': ur.s,
    'temperature': ur.K,
    'conductance': ur.S,
    'resistance': ur['Mohm'],
    'length': ur.m,
}


# HANDLE DATA

class DataHandler:
    """
    Loads data and builds quantities arrays.
    """

    def __init__(self, data_dir='data', chips_dir='chips'):
        self.data_dir = data_dir
        self.chips_dir = chips_dir
        self.props = {}
        self.cps = {}
        self.name = None
        pass

    def load_chip(self, chip, names=None):
        """Load properties file."""
        # extend chip parameters dict
        cp_path = os.path.join(self.chips_dir, chip + '.json')
        self.cps[chip] = ChipParameters(cp_path)
        # extend properties dict
        props_path = os.path.join(self.data_dir, chip, 'properties.csv')
        df = pd.read_csv(props_path, sep='\t', index_col=0)
        if names is None:
            names = df.index
        oldcols = ['input', 'output']
        for name in names:
            self.props[name] = {}
            self.props[name]['pair'] = df.loc[name, 'pair']
            self.props[name]['injection'] = df.loc[name, 'injection']
            self.props[name]['temperature'] = ur.Quantity(df.loc[name, 'temperature'])
            self.props[name]['order'] = ''
            for k in oldcols:
                q = ur.Quantity(df.loc[name, k])
                if q.is_compatible_with(ur.V):
                    self.props[name]['order'] += 'v'
                    self.props[name]['bias'] = q
                elif q.is_compatible_with(ur.A):
                    self.props[name]['order'] += 'i'
                    self.props[name]['current'] = q
                else:
                    raise ValueError(f"Unrecognized units {q.units} of name {name}")

    def load(self, name):
        """Read data from files."""
        old_name = self.name
        if name == old_name:
            return self.data

        # load data
        self.name = name
        self.chip = self.name.split('_')[0]
        self.prop = self.props[name]
        self.data = {}
        path_name = os.path.join(self.data_dir, self.chip, name)
        if os.path.isfile(path_name + '.xlsx'):
            df = pd.read_excel(path_name + '.xlsx', skiprows=[5, 6]).T
            keys = [o.replace('i', 'current').replace('v', 'bias') for o in self.prop['order']] \
                + ['time', 'temperature']
            for i, key in enumerate(keys):
                self.data[key] = ur.Quantity(df[i].to_numpy(), DEFAULT_UNITS[key])
        elif os.path.isfile(path_name + '.csv'):
            df = pd.read_csv(path_name + '.csv')
            old_keys = ['x1', 'y', 'x2', 'temp', 't']
            keys = [o.replace('i', 'current').replace('v', 'bias') for o in self.prop['order']] \
                + ['gate', 'temperature', 'time']
            for old_key, key in zip(old_keys, keys):
                self.data[key] = ur.Quantity(df[old_key].to_numpy(), DEFAULT_UNITS[key])

        # reset quantities
        if old_name is None or self.props[old_name]['pair'] != self.prop['pair']:
            self.length = None
        self.temperature = None
        self.gate = None
        return self.data

    def get_conductance(self, method='all', time_win=None, bias_win=None, noise_level=None,
                        correct_offset=None, debug=False):
        """Calculate conductance."""
        if correct_offset is None:
            correct_offset = False if method == 'fit' else True

        # prepare data
        bias = self.data['bias']
        current = self.data['current']
        cond = np.ones(self.data['bias'].shape, dtype=bool)
        if time_win is not None:
            cond *= is_between(self.data['time'], time_win)
        if correct_offset:
            bias_cond = is_between(bias, [-0.05, 0.05] * ur['V/um'] * self.get_length())
            current -= np.mean(current[cond * bias_cond])
        if bias_win is not None:
            cond *= is_between(self.data['bias'], bias_win)
        if noise_level is not None:
            cond *= (np.abs(current) > noise_level)

        # calculate conductance
        if method == 'fit':
            coeffs, model = fit_linear(bias[cond], current[cond], debug=debug)
            conductance = coeffs[0]
        else:
            cond *= bias != 0 * ur.V
            conductance = current / np.where(cond, bias, np.nan)
            if method == 'average':
                conductance = average(conductance)
            elif method == 'all':
                pass
            else:
                raise ValueError(f"Unrecognized method {method}.")

        return conductance.to(DEFAULT_UNITS['conductance'])

    def get_resistance(self, **kwargs):
        return (1 / self.get_conductance(**kwargs)).to(DEFAULT_UNITS['resistance'])

    def get_temperature(self):
        if self.temperature is None:
            self.temperature = average(self.data['temperature'])
        return self.temperature

    def get_gate(self):
        if self.gate is None:
            self.gate = average(self.data['gate'])
        return self.gate

    def get_length(self):
        if self.length is None:
            self.length = self.cps[self.chip].get_distance(self.prop['pair'])
        return self.length

    def plot(self, ax, mode='i/v', correct_offset=False, x_win=None, y_win=None, label=r'{prop[temperature]}',
             color=None, markersize=5, set_xy_label=True):
        ykey, xkey = [
            c.replace('vg', 'gate').replace('v', 'bias').replace('i', 'current') if c != 't' else 'time'
            for c in mode.lower().split('/')
        ]
        x = self.data[xkey]
        y = self.data[ykey]
        label = label.replace('{', '{0.').format(self)

        # correct data
        if correct_offset:
            y -= np.mean(y)
        cond = np.ones(x.shape, dtype=bool)
        if x_win is not None:
            cond *= is_between(x, x_win)
        if y_win is not None:
            cond *= is_between(y, y_win)
        ax.scatter(x[cond], y[cond], label=label, c=color, s=markersize, edgecolors=None)
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
                x2 = ((x + item['width'] / 2) * ur.um).plus_minus(item['width'] / np.sqrt(12))
                break
            if count == pad1 and x1 is None:
                x1 = ((x + item['width'] / 2) * ur.um).plus_minus(item['width'] / np.sqrt(12))
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
    x_ = unp.nominal_values(x.magnitude)
    dx = unp.std_devs(x.magnitude)
    if len(x_.shape) == 0:
        x_ = float(x_)
        dx = float(dx) if dx != 0 else None
    elif (dx == 0).all():
        dx = None
    return x_, dx, x.units


def isnan(x):
    if hasattr(x, 'units'):
        x_, dx, ux = separate_measurement(x)
        if dx is not None:
            return np.isnan(x_ * dx)
        return np.isnan(x_)
    else:
        return np.isnan(x)


def strip_nan(*args):
    """Strip value if is NaN in any of the arguments."""
    indices = np.sum([isnan(x) for x in args if x is not None], axis=0) == 0
    return [x[indices] if x is not None else None for x in args]


# ANALYSIS

def average(x, ignore_err=False, already_separated=False, check_nan=True, debug=False):
    """Calculate average with standard error."""
    if already_separated:
        x_, dx, ux = x
    else:
        x_, dx, ux = separate_measurement(x)
    if ignore_err:
        dx = None
    if dx is not None:
        cond = dx == 0
        dx[cond] = np.nanmin(dx[np.where(cond, False, True)])/2
    if check_nan:
        x_, dx = strip_nan(x_, dx)

    w = 1 / dx**2 if dx is not None else None
    avg = np.average(x_, weights=w)
    n = len(x_)
    std_err = np.sqrt((n / (n-1)**2) * np.average((x_-avg)**2, weights=w))

    return (avg * ux).plus_minus(std_err)


def fit_powerlaw(x, y, offset=None, **kwargs):
    """Calculate 1D exponential fit (y = exp(a*x + b) + offset)."""
    if offset is not None:
        y1 = y - offset
    else:
        y1 = y
    x_, dx, ux = separate_measurement(x)
    y1_, dy1, uy1 = separate_measurement(y1)

    # apply log
    cond = (x_ > 0) * (y1_ > 0)
    dx = dx[cond] / x_[cond] if dx is not None else None
    x_ = np.log(x_[cond])
    ux = ur.dimensionless
    dy1 = dy1[cond] / y1_[cond] if dy1 is not None else None
    y1_ = np.log(y1_[cond])
    uy1 = ur.dimensionless

    coeffs, _ = fit_linear((x_, dx, ux), (y1_, dy1, uy1), already_separated=True, **kwargs)
    if coeffs is None:
        return None, None

    p = [separate_measurement(c)[0] for c in coeffs]
    if offset is None:
        offset = 0
    else:
        offset, _, _ = separate_measurement(offset)

    return coeffs, lambda x, p=p, offset=offset: np.exp(p[1]) * x ** p[0] + offset


def fit_exponential(x, y, offset=None, **kwargs):
    """Calculate 1D exponential fit (y = exp(a*x + b) + offset)."""
    if offset is not None:
        y1 = y - offset
    else:
        y1 = y
    x_, dx, ux = separate_measurement(x)
    y1_, dy1, uy1 = separate_measurement(y1)

    # apply log
    cond = (y1_ > 0)
    dx = dx[cond] if dx is not None else None
    x_ = x_[cond]
    dy1 = dy1[cond] / y1_[cond] if dy1 is not None else None
    y1_ = np.log(y1_[cond])
    uy1 = ur.dimensionless

    coeffs, _ = fit_linear((x_, dx, ux), (y1_, dy1, uy1), already_separated=True, **kwargs)
    if coeffs is None:
        return None, None

    p = [separate_measurement(c)[0] for c in coeffs]
    if offset is None:
        offset = 0
    else:
        offset, _, _ = separate_measurement(offset)

    return coeffs, lambda x, p=p, offset=offset: np.exp(p[0] * x + p[1]) + offset


def fit_linear(x, y, ignore_err=False, already_separated=False, check_nan=True, debug=False):
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
        cond = dy == 0
        dy[cond] = np.nanmin(dy[np.where(cond, False, True)])/2
    if check_nan:
        x_, dx, y_, dy = strip_nan(x_, dx, y_, dy)

    if len(x_) < 1:
        warnings.warn('Not enough points for fit.', RuntimeWarning)
        return None, None

    p, pcov = np.polyfit(x_, y_, 1, w=1/dy if dy is not None else None, cov=True)

    if dx is not None and dy is not None:
        dy_prop = np.sqrt(dy**2 + (p[1]*dx)**2)
        p, pcov = np.polyfit(x_, y_, 1, w=1/dy_prop, cov=True)

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
    if not hasattr(win[0], 'units'):
        win = np.array(win) * np.amax(x)
    return np.logical_and(x > win[0], x < win[1])
