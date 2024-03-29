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


plt.style.use('./report.mplstyle')

ur = pint.UnitRegistry()
ur.setup_matplotlib(False)
ur.default_format = "~P"

NUM_FIBERS = 60
FIBER_RADIUS = (25 * ur.nanometer).plus_minus(3)


# HIGHER LEVEL FUNCTIONS

# HANDLE DATA

class DataHandler:
    """
    Loads data and builds quantities arrays.
    """
    _PARAMS = {
        'only_return': True,
        'correct_offset': True,
        'offset_mask_args': {'field_win': [-0.01, 0.01]*ur['V/um'], 'only_return': True},
    }
    _UNITS = {
        'bias': ur.V,
        'gate': ur.V,
        'current': ur.A,
        'time': ur.s,
        'temperature': ur.K,
        'conductance': ur.nS,
        'resistance': ur.Mohm,
        'length': ur.m,
    }

    def __init__(self, data_dir='data', chips_dir='chips', **params):
        self.data_dir = data_dir
        self.chips_dir = chips_dir
        self.set_default_params(**params)
        self.props = {}
        self.cp_cache = {}
        self.cp = None
        self.chip = None
        self.clear()
        pass

    def clear(self):
        self.name = None
        self.cache = {}
        self.raw = None
        self.prop = None
        pass

    def load_chip(self, chip, names=None):
        """Load properties file.
        Note: assumes that names are not shared between different chips."""
        # extend chip parameters dict
        cp_path = os.path.join(self.chips_dir, chip + '.json')
        if chip not in self.cp_cache:
            self.cp_cache[chip] = ChipParameters(cp_path)
        self.cp = self.cp_cache[chip]
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
        if not isinstance(name, str):
            names = name
            for i, name in enumerate(names):
                self.load(name)
        else:
            self.name = name
            self.prop = self.props[name]
            if name not in self.cache:
                self.cache[name] = {}
                path_name = os.path.join(self.data_dir, self.chip, name)
                if os.path.isfile(path_name + '.csv'):
                    df = pd.read_csv(path_name + '.csv')
                    old_keys = ['x1', 'y', 'x2', 'temp', 't']
                    keys = [o.replace('i', 'current').replace('v', 'bias') for o in self.prop['order']] \
                        + ['gate', 'temperature', 'time']
                    for old_key, key in zip(old_keys, keys):
                        self.cache[name][key] = ur.Quantity(df[old_key].to_numpy(), self._UNITS[key])
                else:
                    raise FileNotFoundError(f"Could not load name '{name}'. File does not exist.")
            self.raw = self.cache[name]
        return self.raw

    def set_default_params(self, **params):
        self._PARAMS.update(params)

    def set_default_units(self, **units):
        self._UNITS.update(units)

    def get_mask(self, field_win=None, time_win=None, bias_win=None, current_win=None, **params):
        """Generate mask to be applied to data."""
        params = self._get_tmp_params(**params)
        n = self.raw['time'].shape[0]
        mask = np.ones((n,), dtype=bool)
        if time_win is not None:
            mask *= is_between(self.get_time(), time_win)
        if field_win is not None:
            if bias_win is not None:
                warnings.warn("argument 'field_win' ignored because 'bias_win' takes precedence.", UserWarning)
            else:
                mask *= is_between(self.get_field(), field_win)
        if bias_win is not None:
            mask *= is_between(self.get_bias(), bias_win)
        if current_win is not None:
            mask *= is_between(self.get_current(**params), current_win)
        if params['only_return']:
            inp = self.raw['bias'].m if self.prop['injection'].lower() == '2p' else self.raw['current'].m
            if inp[int(n / 6)] > inp[0]:
                first = np.argmax(inp)
                last = np.argmin(inp)
            else:
                first = np.argmin(inp)
                last = np.argmax(inp)
            mask_tmp = np.zeros((n,), dtype=bool)
            mask_tmp[first:last] = True
            mask *= mask_tmp
        return mask

    def get_field(self, **kwargs):
        return self.get_bias(**kwargs) / self.get_length()

    def get_current(self, method='all', mask=None, **params):
        params = self._get_tmp_params(**params)
        if self.prop['injection'].lower() == '2p' and params['correct_offset']:
            offset_mask = self.get_mask(**params['offset_mask_args'])
            current = self.raw['current'] - np.mean(self.raw['current'][offset_mask])
        else:
            current = self.raw['current']
        return self._process_quantity(current, method=method, mask=mask).to(self._UNITS['current'])

    def get_conductance(self, method='all', mask=None, **params):
        """Calculate conductance."""
        bias = self.get_bias()
        if method == 'fit':
            current = self.get_current(mask=mask, correct_offset=False)
            coeffs, _ = fit_linear(bias, current)
            conductance = coeffs[0]
        else:
            bias = self.apply_mask(bias, bias != 0*ur.V)
            current = self.get_current(**params)
            conductance = current / bias
            conductance = self._process_quantity(conductance, method=method, mask=mask)
        return conductance.to(self._UNITS['conductance'])

    def get_resistance(self, **kwargs):
        return (1 / self.get_conductance(**kwargs)).to(self._UNITS['resistance'])

    def get_temperature(self, **kwargs):
        return self._get_simple('temperature', **kwargs)

    def get_bias(self, **kwargs):
        return self._get_simple('bias', **kwargs)

    def get_time(self, **kwargs):
        return self._get_simple('time', **kwargs)

    def get_gate(self, **kwargs):
        return self._get_simple('gate', **kwargs)

    @staticmethod
    def apply_mask(q, mask):
        q = np.where(mask, q, np.nan)
        return q

    @staticmethod
    def _process_quantity(q, method='all', mask=None):
        if method == 'average':
            if mask is not None:
                q = q[mask]
            q = average(q)
        elif method == 'all':
            if mask is not None:
                q = __class__.apply_mask(q, mask)
        else:
            raise ValueError(f'Unknown method {method}.')
        return q

    def _get_simple(self, key, method='all', mask=None):
        q = self.raw[key]
        return self._process_quantity(q, method=method, mask=mask).to(self._UNITS[key])

    def _get_tmp_params(self, **params):
        for key in self._PARAMS:
            if key not in params:
                params[key] = self._PARAMS[key]
        return params

    def get_length(self):
        return self.cp.get_distance(self.prop['pair'])

    def plot(self, ax, mode='i/v', mask=None, notraw=False, label=None,
             color=None, markersize=10, set_xy_label=True, to_compact=['current']):
        key_short_list = ['t', 'v', 'vg', 'i']
        key_list = ['time', 'bias', 'gate', 'current']
        sym_list = ['t', 'V', 'V_G', 'I']
        ykeys, xkey = mode.split('/')
        xkey = key_list[key_short_list.index(xkey)]
        ykeys = [key_list[key_short_list.index(ykey)] for ykey in ykeys.split('-')]
        if notraw:
            x = getattr(self, f'get_{xkey}')()
        else:
            x = self.raw[xkey]
        if xkey in to_compact:
            x = q_to_compact(x)
        if mask is not None:
            x = x[mask]
        if label is not None:
            label = label.replace('{', '{0.').format(self)

        for ykey in ykeys:
            if notraw:
                y = getattr(self, f'get_{ykey}')()
            else:
                y = self.raw[ykey]
            if ykey in to_compact:
                y = q_to_compact(y)
            if mask is not None:
                y = y[mask]
            # correct data
            ax.plot(x.m, y.m, 'o', label=label, c=color, markersize=markersize)
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
                        res[j] *= self._UNITS[key.split('get_')[1]]
                res0 = getattr(self, key)(**per_data_args[key], **instruction_dict[key])
                res[j] = np.append(res[j], res0)
        return res


class ChipParameters():
    """Define chip parameters."""

    def __init__(self, config_path=None):
        if config_path is not None:
            self.load(config_path)
        self.cache = {}

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
        if pair not in self.cache:
            k1, k2 = pair.split('-')
            self.cache[pair] = np.abs(self.positions[k2] - self.positions[k1])
        return self.cache[pair]


# MEASUREMENTS HELP FUNCTIONS

def separate_measurement(x):
    """Get parameters to feed ODR from Quantities."""
    if hasattr(x, 'units'):
        m = x.m
        u = x.u
    else:
        m = np.asarray(x)
        u = ur['']
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


def qlist_to_qarray(qlist):
    """Change a list of quantities to a quantity array."""
    ux = None
    mlist = []
    for x in qlist:
        if x is None or isnan(x):
            mlist.append(np.nan)
        elif hasattr(x, 'units'):
            if ux is not None:
                assert x.u == ux, 'Inconsistent units in quantity list.'
            ux = x.u
            mlist.append(x.m)
        else:
            raise ValueError('Quantity list contains non quantity items.')
    assert ux is not None, 'No element of the list was a quantity!'
    return mlist * ux


def q_from_df(df, key, units=ur['']):
    x = []
    for i, index in enumerate(df.index):
        x.append(df.loc[index, key] * units)
        if f'd_{key}' in df:
            dx = df.loc[index, f'd_{key}']
            x[i] = x[i].plus_minus(dx)
    if len(x) == 1:
        x = x[0]
    else:
        x = np.concatenate(x)
    return x


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
        t0 *= np.amax(np.abs(x))
    mask *= x > t0
    if not hasattr(t1, 'units'):
        t1 *= np.amax(np.abs(x))
    mask *= x < t1
    return mask


def q_to_compact(x):
    return x.to(np.amax(x).to_compact().units)


def fmt(x, latex=False, sep=' '):
    """Format quantity."""
    if latex:
        uargs = 'Lx'
        larg = 'L'
        sep = sep.replace(' ', r'\>')
    else:
        uargs = '~P'
        larg = ''

    def fmt_m(m):
        try:
            res = ('{0:.1uS' + larg + '}').format(m)
        except (ValueError, TypeError):
            res = '{0:.3g}'.format(m)
            if latex:
                res = res.split('e')
                if len(res) > 1:
                    res = r'\times 10^{'.join(res) + '}'
                else:
                    res = res[0]
        return res

    def fmt_u(u):
        if u == ur['']:
            return ''
        res = ('{0:' + uargs + '}').format(u).replace('_', '')
        return res

    if hasattr(x, 'units'):
        string = sep.join([fmt_m(x.m), fmt_u(x.u)])
    elif isinstance(x, pint.Unit):
        string = fmt_u(x)
    else:
        string = fmt_m(x)
    return string


def ulbl(u):
    """make label for units."""
    if u is None or u.is_compatible_with(ur['']):
        return ''
    return r' [' + fmt(u, latex=True) + ']'


# ANALYSIS

def average(x, ignore_err=False, check_nan=True, debug=False):
    """Calculate average with standard error."""
    x_, dx, ux = ensure_separated(x)
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
    x_, dx, ux = ensure_separated(x)
    y_, dy, uy = ensure_separated(y)

    if offset is None:
        offset = 0
    else:
        offset = ensure_separated(offset)[0]
    y_ -= offset

    # apply log
    cond = (x_ >= 0) * (y_ > 0)
    dx = dx[cond] / x_[cond] if dx is not None else None
    x_ = np.log(x_[cond])
    ux = ur['']
    dy = dy[cond] / y_[cond] if dy is not None else None
    y_ = np.log(y_[cond])
    uy = ur['']

    coeffs, _ = fit_linear((x_, dx, ux), (y_, dy, uy), **kwargs)
    if coeffs is None:
        return None, None

    p = [separate_measurement(c)[0] for c in coeffs]

    return coeffs, lambda x, p=p, offset=offset: (np.where(x >= 0, x, np.nan) ** p[0]) * np.exp(p[1]) + offset


def fit_exponential(x, y, offset=None, **kwargs):
    """Calculate 1D exponential fit (y = exp(a*x + b) + offset)."""
    x_, dx, ux = ensure_separated(x)
    y_, dy, uy = ensure_separated(y)

    if offset is None:
        offset = 0
    else:
        offset = ensure_separated(offset)[0]
    y_ -= offset

    # apply log
    cond = (y_ > 0)
    dx = dx[cond] if dx is not None else None
    x_ = x_[cond]
    dy = dy[cond] / y_[cond] if dy is not None else None
    y_ = np.log(y_[cond])
    uy = ur['']

    coeffs, _ = fit_linear((x_, dx, ux), (y_, dy, uy), **kwargs)
    if coeffs is None:
        return None, None

    p = [separate_measurement(c)[0] for c in coeffs]

    return coeffs, lambda x, p=p, offset=offset: np.exp(p[0] * x + p[1]) + offset


def fit_linear(x, y, ignore_err=False, check_nan=True, debug=False):
    """Calculate 1D linear fit (y = p[0] + p[1]*x)."""
    x_, dx, ux = ensure_separated(x)
    y_, dy, uy = ensure_separated(y)

    if ignore_err:
        dx = None
        dy = None

    # error cannot be zero, change to small value (only on y is  ok)
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
        plt.errorbar(x_, y_, xerr=dx, yerr=dy, fmt='.', zorder=0)
        plt.plot(x_, p[0]*x_ + p[1], zorder=1)
        plt.show()

    return coeffs, lambda x, p=p: p[0]*x + p[1]


def fit_generic(x, y, f, beta0, units, log='', ignore_err=False, check_nan=True, debug=False, **kwargs):
    """Fit curve defined by `f`."""
    x_, dx, _ = ensure_separated(x)
    y_, dy, _ = ensure_separated(y)

    if ignore_err:
        dx = None
        dy = None

    # error cannot be zero, change to small value (only on y is  ok)
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
        cond *= (y_ > 0)
        cond *= (f(beta0, x_) > 0)
    if 'x' in log:
        dx = dx[cond] / x_[cond] if dx is not None else None
        x_ = np.log(x_[cond])
    else:
        dx = dx[cond] if dx is not None else None
        x_ = x_[cond]
    if 'y' in log:
        dy = dy[cond] / y_[cond] if dy is not None else None
        y_ = np.log(y_[cond])
    else:
        dy = dy[cond] if dy is not None else None
        y_ = y_[cond]

    # get appropriate model
    if log == '':
        model = Model(f)
    elif log == 'x':
        model = Model(lambda beta, x: f(beta, np.exp(x)))
    elif log == 'y':
        model = Model(lambda beta, x: np.log(f(beta, x)))
    elif log == 'xy':
        model = Model(lambda beta, x: np.log(f(beta, np.exp(x))))
    else:
        raise ValueError(f"Parameter 'log' can be '', 'x', 'y' or 'xy', not '{log}'.")

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


def ensure_separated(x):
    if isinstance(x, tuple) or (isinstance(x, list) and len(x) == 3):
        return x
    else:
        return separate_measurement(x)
