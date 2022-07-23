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
ur.default_format = ".3g~P"

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


# HIGHER LEVEL FUNCTIONS

# HANDLE DATA

class DataHandler:
    """
    Loads data and builds quantities arrays.
    """
    DEFAULT_OFFSET_MASK_ARGS = {'field_win': [-0.05, 0.05]*ur['V/um'], 'time_win': [0.25, 0.75]}

    def __init__(self, data_dir='data', chips_dir='chips', offset_mask_args=None):
        self.data_dir = data_dir
        self.chips_dir = chips_dir
        self.set_offset_mask_args(offset_mask_args)

        self.props = {}
        self.cps = {}
        self.chip = None
        self.name = None
        self.raws = {}
        self.raw = None
        self.length = None
        self.gate = None
        self.temperature = None
        pass

    def load_chip(self, chip, names=None):
        """Load properties file."""
        # extend chip parameters dict
        cp_path = os.path.join(self.chips_dir, chip + '.json')
        self.cps[chip] = ChipParameters(cp_path)
        self.chip = chip
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
            return self.raw

        # load data
        self.name = name
        self.chip = self.name.split('_')[0]
        self.prop = self.props[name]
        if name in self.raws:
            self.raw = self.raws[name]
        else:
            self.raw = {}
            path_name = os.path.join(self.data_dir, self.chip, name)
            if os.path.isfile(path_name + '.xlsx'):
                df = pd.read_excel(path_name + '.xlsx', skiprows=[5, 6]).T
                keys = [o.replace('i', 'current').replace('v', 'bias') for o in self.prop['order']] \
                    + ['time', 'temperature']
                for i, key in enumerate(keys):
                    self.raw[key] = ur.Quantity(df[i].to_numpy(), DEFAULT_UNITS[key])
            elif os.path.isfile(path_name + '.csv'):
                df = pd.read_csv(path_name + '.csv')
                old_keys = ['x1', 'y', 'x2', 'temp', 't']
                keys = [o.replace('i', 'current').replace('v', 'bias') for o in self.prop['order']] \
                    + ['gate', 'temperature', 'time']
                for old_key, key in zip(old_keys, keys):
                    self.raw[key] = ur.Quantity(df[old_key].to_numpy(), DEFAULT_UNITS[key])
            else:
                raise FileNotFoundError(f"Could not load name '{name}'. File does not exist.")
            self.raws[name] = self.raw

        # reset quantities
        if old_name is None or self.props[old_name]['pair'] != self.prop['pair']:
            self.length = None
        self.temperature = None
        self.gate = None
        return self.raw

    def get_mask(self, field_win=None, time_win=None, bias_win=None):
        """Generate mask to be applied to data.
        Note: field_win will compute bias_win = length * field_win.
        """
        mask = np.ones(self.raw['bias'].shape, dtype=bool)
        if time_win is not None:
            mask *= is_between(self.raw['time'], time_win)
        if field_win is not None:
            if bias_win is not None:
                warnings.warn("argument 'field_win' ignored because 'bias_win' takes precedence.", UserWarning)
            else:
                if isinstance(field_win, list):
                    field_win = qlist2qarray(field_win)
                bias_win = field_win * self.get_length()
        if bias_win is not None:
            mask *= is_between(self.raw['bias'], bias_win)
        return mask

    def set_offset_mask_args(self, offset_mask_args=None):
        if offset_mask_args is None:
            self.offset_mask_args = self.DEFAULT_OFFSET_MASK_ARGS
        else:
            self.offset_mask_args = offset_mask_args

    def get_conductance(self, correct_offset=False, noise_level=None, **wins):
        """Calculate conductance."""
        bias = self.raw['bias']
        current = self.raw['current']
        mask = self.get_mask(**wins)
        if correct_offset:
            offset_mask = self.get_mask(**self.offset_mask_args)
            current -= np.mean(current[offset_mask])
        if noise_level is not None:
            mask *= np.abs(current) > noise_level
        mask *= bias != 0*ur.V
        conductance = current / np.where(mask, bias, np.nan)
        return conductance.to(DEFAULT_UNITS['conductance'])

    def get_resistance(self, **kwargs):
        return (1 / self.get_conductance(**kwargs)).to(DEFAULT_UNITS['resistance'])

    def get_temperature(self):
        if self.temperature is None:
            self.temperature = average(self.raw['temperature'])
        return self.temperature

    def get_gate(self):
        if self.gate is None:
            self.gate = average(self.raw['gate'])
        return self.gate

    def get_length(self):
        if self.length is None:
            self.length = self.cps[self.chip].get_distance(self.prop['pair'])
        return self.length

    def plot(self, ax, mode='i/v', correct_offset=False, label=None,
             color=None, markersize=5, set_xy_label=True, **wins):
        mode_chars = ['t', 'v', 'vg', 'i']
        to_key = ['time', 'bias', 'gate', 'current']
        to_sym = ['t', 'V', 'V_G', 'I']
        ykey, xkey = [to_key[mode_chars.index(c)] for c in mode.lower().split('/')]
        x = self.raw[xkey]
        y = self.raw[ykey]
        if label is not None:
            label = label.replace('{', '{0.').format(self)

        # correct data
        if correct_offset:
            y -= np.mean(y)
        mask = self.get_mask(**wins)
        ax.scatter(x[mask], y[mask], label=label, c=color, s=markersize, edgecolors=None)
        if set_xy_label:
            ysym, xsym = [to_sym[mode_chars.index(c)] for c in mode.lower().split('/')]
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

    def __init__(self, config_path=None):
        if config_path is not None:
            self.load(config_path)

    def load(self, config_path):
        """
        Set attributes from configuration dictionary.

        :param config_path: path to json configuration file.
        """
        config = json.load(open(config_path))

        self.positions = {}
        count = 0
        x = 0
        for item in config['layout']:
            if item['type'] == "contact":
                count += 1
                self.positions[f'P{count}'] = ((x + item['width'] / 2) * ur.um).plus_minus(item['width'] / np.sqrt(12))
            x += item['width']

    def get_distance(self, pair):
        """Get distance between 2 cojtacts."""
        k1, k2 = pair.split('-')
        return np.abs(self.positions[k2] - self.positions[k1])


# MEASUREMENTS HELP FUNCTIONS

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


def qlist2qarray(qlist):
    """Change a list of quantities to a quantity array."""
    ux = qlist[0].units
    mlist = []
    for x in qlist:
        assert x.units == ux, 'Inconsistent units in quantity list.'
        mlist.append(x.magnitude)
    return mlist * ux


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
    """Calculate 1D exponential fit (y = x^a * exp(b) + offset)."""
    x_, dx, ux = separate_measurement(x)
    y_, dy, uy = separate_measurement(y)
    if offset is None:
        offset = 0
    else:
        offset = separate_measurement(offset)[0]
    y_ -= offset

    # apply log
    cond = (x_ >= 0) * (y_ > 0)
    dx = dx[cond] / x_[cond] if dx is not None else None
    x_ = np.log(x_[cond])
    ux = ur.dimensionless
    dy = dy[cond] / y_[cond] if dy is not None else None
    y_ = np.log(y_[cond])
    uy = ur.dimensionless

    coeffs, _ = fit_linear((x_, dx, ux), (y_, dy, uy), already_separated=True, **kwargs)
    if coeffs is None:
        return None, None

    p = [separate_measurement(c)[0] for c in coeffs]

    return coeffs, lambda x, p=p, offset=offset: (np.where(x >= 0, x, np.nan) ** p[0]) * np.exp(p[1]) + offset


def fit_exponential(x, y, offset=None, **kwargs):
    """Calculate 1D exponential fit (y = exp(a*x + b) + offset)."""
    x_, dx, ux = separate_measurement(x)
    y_, dy, uy = separate_measurement(y)
    if offset is None:
        offset = 0
    else:
        offset = separate_measurement(offset)[0]
    y_ -= offset

    # apply log
    cond = (y_ > 0)
    dx = dx[cond] if dx is not None else None
    x_ = x_[cond]
    dy = dy[cond] / y_[cond] if dy is not None else None
    y_ = np.log(y_[cond])
    uy = ur.dimensionless

    coeffs, _ = fit_linear((x_, dx, ux), (y_, dy, uy), already_separated=True, **kwargs)
    if coeffs is None:
        return None, None

    p = [separate_measurement(c)[0] for c in coeffs]

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
    t_low, t_high = win
    mask = np.ones(x.shape, dtype=bool)
    if t_low is not None:
        if not hasattr(t_low, 'units'):
            t_low *= np.amax(x)
        mask *= x > t_low
    if t_high is not None:
        if not hasattr(t_high, 'units'):
            t_high *= np.amax(x)
        mask *= x < t_high
    return mask
