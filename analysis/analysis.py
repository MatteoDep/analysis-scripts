# -*- coding: utf-8 -*-
"""analysis.

Analysis module for cable bacteria experiments.
"""


import os
import re
from glob import glob
import json
import numpy as np
from scipy.odr import Model, RealData, ODR
import uncertainties as u
from uncertainties import unumpy as unp
from matplotlib import pyplot as plt
import pandas as pd

from . import Visualise as v
from . import Quantity


def linear_fit(qx, qy):
    """Calculate 1D llinear fit."""

    def linear(b, x):
        """Linear model."""
        return b[0]*x + b[1]

    def estimate(data):
        """Parameters estimation."""
        beta0 = [0, data.y[0]/data.x[0]]
        return np.array(beta0)

    # separate data and uncertainties
    x, sigma_x = qx.get_raw()
    y, sigma_y = qy.get_raw()

    model = Model(linear, estimate=estimate)
    data = RealData(x, y, sx=sigma_x, sy=sigma_y)
    odr = ODR(data, model)
    res = odr.run()
    coeffs = [u.ufloat(res.beta[i], res.sd_beta[i]) for i in range(len(res.beta))]

    return coeffs, lambda x: linear(res.beta, x)


class Chip:
    """Define chip parameters."""

    def __init__(self, config=None):
        """
        Create object and load configuration.

        :param config: configuration toml file or dictionary.
        """
        # default values
        self.sigma = 0
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

        :param config: json configuration file or dictionary.
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

        return u.ufloat(distance, self.sigma)


class DataHandler:
    """Functions to compute various quantities."""

    QUANTITIES = ["resistance", "length", "resistivity", "conductivity"]
    SIGMA = 0

    def __init__(self, chip_parameters, path=None, **kwargs):
        """Declare quantities."""
        self.cp = chip_parameters

        for q in self.QUANTITIES:
            setattr(self, q, None)

        if path is not None:
            self.load(path, **kwargs)

    def inspect(self):
        """Plot input/output over time and output over input."""
        current_label = r"current [A]"
        voltage_label = r"voltage [V]"
        time_label = r"time [s]"

        fig, axs = plt.subplots(3, 1)
        plt.tight_layout()
        fig.suptitle(self.name)

        time = unp.nominal_values(self.time)
        voltage = unp.nominal_values(self.voltage)
        current = unp.nominal_values(self.current)

        axs[0].plot(time, current, 'o')
        axs[0].set_xlabel(time_label)
        axs[0].set_ylabel(current_label)

        axs[1].plot(time, voltage, 'o')
        axs[1].set_xlabel(time_label)
        axs[1].set_ylabel(voltage_label)

        axs[2].plot(current, voltage, 'o')
        axs[2].set_xlabel(current_label)
        axs[2].set_ylabel(voltage_label)

        if self.resistance is not None:
            # resistance = self.resistance.nominal_value
            # offset = self.offset.nominal_value
            y = self.resistance_model(current)
            axs[2].plot(current, y, '-')

        plt.show()

    def get(self, qname):
        """Call function to compute a certain quantity."""
        try:
            q = getattr(self, qname)
            if q is None:
                q = getattr(self, "_compute_" + qname)()
        except AttributeError:
            err = f"Computing {qname} is not implemented.\navailable quantities are: {self.QUANTITIES}"
            raise ValueError(err)
        return q

    def _compute_length(self):
        """Compute cable length between electrodes."""
        self.length = self.cp.get_distance(self.segment)
        return self.length

    def _compute_resistance(self):
        """Compute resistance from current voltage curve."""
        coeffs, self.resistance_model = linear_fit(self.current, self.voltage)
        self.resistance = coeffs[0]
        self.offset = coeffs[1]
        return self.resistance


class Analysis:
    """Analyse data from an experiment."""

    def __init__(self, data_dir, chip, verbosity=1):
        """Initialize parameters."""
        self.data_dir = data_dir
        self.chip = chip
        self.verbosity = verbosity

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
            self.current = unp.uarray(d.loc[0], self.SIGMA)
            self.voltage = unp.uarray(d.loc[1], self.SIGMA)
        elif order == "vi":
            self.voltage = unp.uarray(d.loc[0], self.SIGMA)
            self.current = unp.uarray(d.loc[1], self.SIGMA)
        else:
            raise ValueError(f"Unknown order '{order}'.")
        self.time = unp.uarray(d.loc[2], self.SIGMA)
        self.temperature = unp.uarray(d.loc[3], self.SIGMA)

    def compute_quantities(self, quantities):
        """Start analysis."""
        qd = {mode: {} for mode in MODES}
        for mode in MODES:
            qd[mode] = {q: [] for q in quantities}
        segments = []

        pattern = os.path.join(self.data_dir, '*.xlsx')
        for path in np.sort(glob(pattern)):
            # initialize data handler
            dh = DataHandler(self.chip, path)

            # get properties
            name = dh.name
            mode = dh.mode
            segments.append(dh.segment)

            self.print("---", name, "---")
            for q in quantities:
                value = dh.get(q)
                qd[mode][q].append(value)
                if self.verbosity > 0:
                    v.print(f"{q}:", value)

                # set also for other modes but to np.nan
                # if the file exist it will be overwritten
                for mode_ in (set(MODES)-set([mode])):
                    if q == "length":
                        qd[mode_][q].append(qd[mode][q][-1])
                    else:
                        qd[mode_][q].append(u.ufloat(np.nan, 0))

            if self.verbosity > 1:
                dh.inspect()

        # create arrays from lists
        segments = np.array(segments)
        indeces = np.argsort(segments[:, 1])
        segments.sort(axis=0)
        for mode in MODES:
            for q in quantities:
                qd[mode][q] = np.array(qd[mode][q])[indeces]

        return qd, segments


def compute_resistivity(length, resistance):
    """Compute resistivity from lengths and resistances."""
    coeffs, model = linear_fit(length, resistance)
    resistivity = coeffs[0] * AREA
    return resistivity, model
