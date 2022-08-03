# -*- coding: utf-8 -*-
"""analysis.

API for analysing data.
"""


import json
import os
import numpy as np
import pandas as pd
# from scipy.optimize import curve_fit
from scipy.odr import Model, RealData, ODR
from matplotlib import pyplot as plt
from uncertainties import unumpy as unp
import pint
import warnings


ur = pint.UnitRegistry()
ur.setup_matplotlib(False)
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
                    raise ValueError(f"Unrecognized units {q.u} of name {name}")

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
            if os.path.isfile(path_name + '.csv'):
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
        return self.raw

    def get_mask(self, field_win=None, time_win=None, bias_win=None, current_win=None):
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
        if current_win is not None:
            mask *= is_between(self.raw['current'], current_win)
        return mask

    def set_offset_mask_args(self, offset_mask_args=None):
        if offset_mask_args is None:
            self.offset_mask_args = self.DEFAULT_OFFSET_MASK_ARGS
        else:
            self.offset_mask_args = offset_mask_args

    def get_conductance(self, correct_offset=False, noise_level=None, **wins):
        """Calculate conductance."""
        bias = self.raw['bias'].copy()
        current = self.raw['current'].copy()
        mask = self.get_mask(**wins)
        if correct_offset:
            offset_mask = self.get_mask(**self.offset_mask_args)
            current -= np.mean(current[offset_mask])
        if noise_level is not None:
            mask *= np.abs(current) > noise_level
        mask *= bias != 0*ur.V
        notmask = np.where(mask, False, True)
        current[notmask] = np.nan
        bias[notmask] = np.nan
        conductance = current / bias
        return conductance.to(DEFAULT_UNITS['conductance'])

    def get_resistance(self, **kwargs):
        return (1 / self.get_conductance(**kwargs)).to(DEFAULT_UNITS['resistance'])

    def get_temperature(self):
        return average(self.raw['temperature'])

    def get_gate(self):
        return average(self.raw['gate'])

    def get_length(self):
        if self.length is None:
            self.length = self.cps[self.chip].get_distance(self.prop['pair'])
        return self.length

    def plot(self, ax, mode='i/v', correct_offset=False, label=None,
             color=None, set_xy_label=True, **wins):
        key_short_list = ['t', 'v', 'vg', 'i']
        key_list = ['time', 'bias', 'gate', 'current']
        sym_list = ['t', 'V', 'V_G', 'I']
        ykeys, xkey = mode.split('/')
        xkey = key_list[key_short_list.index(xkey)]
        ykeys = [key_list[key_short_list.index(ykey)] for ykey in ykeys.split('-')]
        x = self.raw[xkey]
        if label is not None:
            label = label.replace('{', '{0.').format(self)

        for ykey in ykeys:
            y = self.raw[ykey]
            # correct data
            if correct_offset and ykey == 'current':
                offset_mask = self.get_mask(**self.offset_mask_args)
                y -= np.mean(y[offset_mask])
            mask = self.get_mask(**wins)
            ax.plot(x[mask].m, y[mask].m, '.', label=label, c=color)
            if set_xy_label:
                ysym = sym_list[key_list.index(ykey)]
                ax.set_ylabel(f"${ysym}$" + ulbl(y.u))
        if set_xy_label:
            xsym = sym_list[key_list.index(xkey)]
            ax.set_xlabel(f"${xsym}$" + ulbl(x.u))
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
    if hasattr(x, 'units'):
        m = x.m
        u = x.u
    else:
        m = np.asarray(x)
        u = ur.dimensionless
    x_ = unp.nominal_values(m)
    dx = unp.std_devs(m)
    if len(x_.shape) == 0:
        x_ = float(x_)
        dx = float(dx) if dx != 0 else None
    elif (dx == 0).all():
        dx = None
    return x_, dx, u


def strip_err(x):
    x_, _, ux = separate_measurement(x)
    return x_ * ux


def qlist2qarray(qlist):
    """Change a list of quantities to a quantity array."""
    ux = None
    mlist = []
    for x in qlist:
        if x is None:
            x = np.nan
        if hasattr(x, 'units'):
            if ux is not None:
                assert x.u == ux, 'Inconsistent units in quantity list.'
            ux = x.u
            mlist.append(x.m)
        else:
            mlist.append(x)
    assert ux is not None, 'No element of the list was a quantity!'
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


def is_between(x, win):
    t0, t1 = win
    mask = np.ones(x.shape, dtype=bool)
    if not hasattr(t0, 'units'):
        t0 *= np.amax(x)
    mask *= x > t0
    if not hasattr(t1, 'units'):
        t1 *= np.amax(x)
    mask *= x < t1
    return mask


def fmt(x, latex=False):
    """Format quantity."""
    if latex:
        uargs = '~L'
        larg = 'L'
    else:
        uargs = '~P'
        larg = ''

    def fmt_m(m):
        try:
            return ('{0:.1uS' + larg + '}').format(m)
        except ValueError:
            return '{0:.3g}'.format(m)

    def fmt_u(u):
        return ('{0:' + uargs + '}').format(u)

    if hasattr(x, 'units'):
        string = ' '.join([fmt_m(x.m), fmt_u(x.u)])
    elif isinstance(x, pint.Unit):
        string = fmt_u(x)
    else:
        string = fmt_m(x)
    return string


def ulbl(u):
    """make label for units."""
    return r' [$' + fmt(u, latex=True) + '$]'


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

    if len(x_) < 2:
        warnings.warn('Not enough points for fit.', RuntimeWarning)
        return None, None

    p, pcov = np.polyfit(x_, y_, 1, w=1/dy if dy is not None else None, cov=True)

    if dx is not None and dy is not None:
        dy_prop = np.sqrt(dy**2 + (p[1]*dx)**2)
        p, pcov = np.polyfit(x_, y_, 1, w=1/dy_prop, cov=True)

    dp = np.sqrt(np.diag(pcov))

    # build result as physical quantities
    units = [uy/ux, uy]
    coeffs = [(p_ * up).plus_minus(dp_) for p_, dp_, up in zip(p, dp, units)]

    if debug:
        plt.figure()
        plt.errorbar(x_, y_, xerr=dx, yerr=dy, fmt='o')
        plt.plot(x_, p[0]*x_ + p[1])
        plt.show()

    return coeffs, lambda x, p=p: p[0]*x + p[1]


def fit_generic(x, y, f, beta0, units, log='', already_separated=False, ignore_err=False, check_nan=True, debug=False, **kwargs):
    """Fit curve defined by `f`."""
    if already_separated:
        x_, dx, _ = x
        y_, dy, _ = y
    else:
        x_, dx, _ = separate_measurement(x)
        y_, dy, _ = separate_measurement(y)
    if ignore_err:
        dx = None
        dy = None
    if dy is not None:
        cond = dy == 0
        dy[cond] = np.nanmin(dy[np.where(cond, False, True)])/2
    if check_nan:
        x_, dx, y_, dy = strip_nan(x_, dx, y_, dy)

    # apply log
    cond = np.ones(x_.shape, bool)
    if 'x' in log:
        cond *= (x_ > 0)
    if 'y' in log:
        cond *= (x_ > 0)
    if 'x' in log:
        dx = dx[cond] / x_[cond] if dx is not None else None
        x_ = np.log(x_[cond])
    else:
        dx = dx[cond] if dx is not None else None
        x_ = x_[cond]
    if 'y' in log:
        dy = dy[cond] / y_[cond] if dy is not None else None
        y_ = np.log(y_[cond])
        model = Model(lambda beta, x: np.log(f(beta, x)))
    else:
        dy = dy[cond] if dy is not None else None
        y_ = y_[cond]
        model = Model(f)

    data = RealData(x_, y_, sx=dx, sy=dy)
    odr = ODR(data, model, beta0=beta0, **kwargs)
    out = odr.run()
    p = out.beta
    dp = out.sd_beta

    # build result as physical quantities
    coeffs = [(p_ * up).plus_minus(dp_) for p_, dp_, up in zip(p, dp, units)]

    if debug:
        plt.figure()
        plt.errorbar(x_, y_, xerr=dx, yerr=dy, fmt='o')
        plt.plot(x_, f(x_, *p))
        plt.show()

    return coeffs, lambda x, p=p: f(p, x)
