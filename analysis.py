# -*- coding: utf-8 -*-
"""analysis.

API for analysing data.
"""


import json
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


def load_data(path, order='vi'):
    """Read data from files."""
    # load data
    df = pd.read_excel(path, skiprows=[5, 6]).T
    data = {}
    keys = [o.replace('i', 'current').replace('v', 'voltage') for o in order.lower()] \
        + ['time', 'temperature']
    for i, key in enumerate(keys):
        data[key] = Q(df[i].to_numpy(), DEFAULT_UNITS[key])
    return data


def load_properties(path, names=None, sep='\t'):
    """Load properties file."""
    df = pd.read_csv(path, sep=sep, index_col=0)
    if names is None:
        names = df.index
    props = {}
    oldcols = ['input', 'output']
    for name in names:
        props[name] = {}
        props[name]['order'] = ''
        props[name]['temperature'] = Q(df.loc[name, 'temperature'])
        for k in oldcols:
            q = Q(df.loc[name, k])
            if q.is_compatible_with(ur.V):
                props[name]['order'] += 'v'
                props[name]['voltage'] = q
            elif q.is_compatible_with(ur.A):
                props[name]['order'] += 'i'
                props[name]['current'] = q
            else:
                raise ValueError(f"Unrecognized units {q.units} of name {name}")
    return props


# COMPUTE QUANTITIES FROM DATA

def get_conductance(data, method="auto", bias_window=None):
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
    cond = data['voltage'] != 0
    if bias_window is not None:
        if not hasattr(bias_window, 'units'):
            bias_window = bias_window * DEFAULT_UNITS['voltage']
        cond *= is_between(data['voltage'], bias_window)

    # calculate conductance
    if method == "fit":
        coeffs, model = fit_linear(data['voltage'][cond], data['current'][cond])
        conductance = coeffs[0]
    elif method == "average":
        conductance = get_mean_std(data['current'][cond]/data['voltage'][cond])
    else:
        raise ValueError(f"Unrecognized method {method}.")

    return conductance


def get_tau(data, time_window, const_estimate_time=5*ur.s):
    """Calculate time constant tau."""
    t = data['time']
    i = data['current']
    if not hasattr(time_window, 'units'):
        time_window *= DEFAULT_UNITS['time']
    t_start = time_window[0]
    t_end = time_window[1]
    i_start = np.mean(data['current'][is_between(data['time'], [t_start - const_estimate_time, t_start])])
    i_end = np.mean(data['current'][is_between(data['time'], [t_end - const_estimate_time, t_end])])
    amp = i_start - i_end
    offset = i_end

    cond = is_between(t, [t_start, t_end])
    t_cut = t[cond]
    i_cut = i[cond]
    sign = (-1) ** int(amp < 0)
    t_end_decay = t_cut[np.nonzero(sign * i_cut < sign * (amp * np.exp(-3) + offset))[0][0]]

    cond = is_between(t, [t_start, t_end_decay])
    x = t[cond]
    y = np.log(sign * (i[cond] - offset).magnitude) * ur.dimensionless

    coeffs, model = fit_linear(x, y)
    tau = - 1 / coeffs[0]

    return tau


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


def strip_units(a):
    """Strip unit from Quantity x."""
    if hasattr(a, "magnitude"):
        s = a.magnitude
    elif isinstance(a, np.ndarray):
        s = np.array([strip_units(x) for x in a])
    elif isinstance(a, list):
        s = [strip_units(x) for x in a]
    else:
        s = a
    return s


def qlist_to_array(x):
    """Turn a quantity list in a numpy array."""
    unit = x[0].units
    return np.array(strip_units(x)) * unit


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


def strip_nan(*args):
    """Strip value if is NaN in any of the arguments."""
    args = [x for x in args if x is not None]
    prod = True
    for x in args:
        prod *= x
    indices = np.isnan(prod) == 0
    for x in args:
        x = x[indices]


def is_between(x, window):
    return np.logical_and(x > window[0], x < window[1])
