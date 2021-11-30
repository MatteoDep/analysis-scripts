# -*- coding: utf-8 -*-
"""res_vs_length.

analyse data to plot conductivity over length.
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


NUM_FIBERS = 60
FIBER_RADIUS = u.ufloat(25e-9, 1e-10)


def p(*args, inner=False):
    """Print in scientific notation."""
    out = ""
    for arg in args:
        if arg is None:
            out += " None"
        elif isinstance(arg, str):
            out += " " + arg
        elif isinstance(arg, list):
            out += " " + p(*arg, inner=True)
        elif isinstance(arg, np.ndarray):
            out += " {}".format(arg)
        elif isinstance(arg, u.UFloat):
            out += " {:.2u}".format(arg)
        else:
            out += " {:.3f}".format(arg)
    if inner:
        return out
    else:
        print(out)


def linear_fit(x, y):
    """Calculate 1D llinear fit."""
    for d in (x, y):
        if isinstance(d, list):
            if len(d) < 1:
                return None, None
            if not isinstance(d[0], u.UFloat):
                d = unp.uarray(d, np.zeros(len(d)))
            else:
                d = np.array(d)

    def linear(b, x):
        """Linear model."""
        return b[0]*x + b[1]

    def estimate(data):
        """Parameters estimation."""
        beta0 = [0, data.y[0]/data.x[0]]
        return np.array(beta0)

    # separate data and uncertainties
    x_ = unp.nominal_values(x)
    y_ = unp.nominal_values(y)
    sigma_x = unp.std_devs(x)
    sigma_y = unp.std_devs(y)
    if not sigma_x.all():
        sigma_x = None
    if not sigma_y.all():
        sigma_y = None

    model = Model(linear, estimate=estimate)
    data = RealData(x_, y_, sx=sigma_x, sy=sigma_y)
    odr = ODR(data, model)
    res = odr.run()
    coeffs = [u.ufloat(res.beta[i], res.sd_beta[i]) for i in range(len(res.beta))]

    return coeffs


class ChipParameters():
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

        return u.ufloat(distance, self.sigma)


class DataHandler():
    """Functions to compute various quantities."""

    QUANTITIES = ["resistance", "length", "resistivity", "conductivity"]
    SIGMA = 0
    MODES = ['2p', '4p']

    def __init__(self, chip_parameters, path=None, **kwargs):
        """Declare quantities."""
        self.cp = chip_parameters

        for q in self.QUANTITIES:
            setattr(self, q, None)

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
        if self.mode not in self.MODES:
            raise ValueError(f"Unknown mode '{mode}'.")
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
            resistance = self.resistance.nominal_value
            offset = self.offset.nominal_value
            y = resistance * current + offset
            axs[2].plot(current, y, '-')

        plt.show()

    def get(self, quantity):
        """Call function to compute a certain quantity."""
        try:
            q = getattr(self, quantity)
            if q is None:
                q = getattr(self, "_compute_" + quantity)()
        except AttributeError:
            err = f"Computing {quantity} is not implemented.\navailable quantities are: {self.QUANTITIES}"
            raise ValueError(err)
        return q

    def _compute_resistivity(self):
        """Compute resistivity from resistance and length."""
        length = self.get("length")
        resistance = self.get("resistance")
        surface = NUM_FIBERS * np.pi * FIBER_RADIUS**2
        self.resistivity = resistance * surface / length
        return self.resistivity

    def _compute_conductivity(self):
        """Compute conductivity from resistivity."""
        resistivity = self.get("resistivity")
        self.conductivity = 1 / resistivity
        return self.conductivity

    def _compute_length(self):
        """Compute cable length between electrodes."""
        self.length = self.cp.get_distance(self.segment)
        return self.length

    def _compute_resistance(self):
        """Compute resistance from current voltage curve."""
        coeffs = linear_fit(self.current, self.voltage)
        self.resistance = coeffs[0]
        self.offset = coeffs[1]
        return self.resistance


def compute_quantities(data_dir, quantities, verbosity=1):
    """Start analysis."""
    modes = DataHandler.MODES
    qd = {mode: {} for mode in modes}
    for mode in modes:
        qd[mode] = {q: [] for q in quantities}
    segments = []

    # for mode in modes:
    #     p(f"### {mode} ###\n")
    pattern = os.path.join(data_dir, '*.xlsx')
    for path in np.sort(glob(pattern)):
        # initialize data handler
        dh = DataHandler(cp, path)

        # get properties
        name = dh.name
        mode = dh.mode
        segments.append(dh.segment)

        p("---", name, "---")
        for q in quantities:
            value = dh.get(q)
            qd[mode][q].append(value)
            if verbosity > 0:
                p(f"{q}:", value)

            # set also for other modes but to np.nan
            # if the file exist it will be overwritten
            for mode_ in (set(modes)-set([mode])):
                if q == "length":
                    qd[mode_][q].append(qd[mode][q][-1])
                else:
                    qd[mode_][q].append(u.ufloat(np.nan, 0))

        if verbosity > 1:
            dh.inspect()

    # create arrays from lists
    segments = np.array(segments)
    indeces = np.argsort(segments[:, 1])
    segments.sort(axis=0)
    for mode in modes:
        for q in quantities:
            qd[mode][q] = np.array(qd[mode][q])[indeces]

    return qd, segments


if __name__ == "__main__":
    # chip = "SLC7"
    # chip = "SKC7"
    chip = "SIC1x"
    experiment = "4p_room-temp"
    chip_dir = os.path.join("data", chip)
    data_dir = os.path.join(chip_dir, experiment)
    res_dir = os.path.join("results", chip, experiment)
    verbosity = 1   # 0, 1 or 2

    # load chip parameters
    cp = ChipParameters(os.path.join(chip_dir, chip + ".json"))
    # create results dir if it doesn't exist
    os.makedirs(res_dir, exist_ok=True)
    # define plots to produce
    couples = [
        ["length", "resistance"],
        ["length", "conductivity"],
    ]
    all_modes = DataHandler.MODES
    quantities = np.unique(np.array(couples).flatten())

    qd, segments = compute_quantities(data_dir, quantities, verbosity=verbosity)
    print(qd)

    coeffs = linear_fit(qd['4p']["length"], qd['4p']["resistance"])
    resistivity = coeffs[0] * NUM_FIBERS * np.pi * FIBER_RADIUS**2
    conductivity = 1 / resistivity

    if verbosity > 0:
        p("\nresistivity:", resistivity)
        p("conductivity:", conductivity)

    factor = {
        'length': 1e6,
        'resistance': 1e-6,
        'conductivity': 1e-2,
    }
    label = {
        'length': r'length [$\mu m$]',
        'resistance': r'resistance [$M\Omega$]',
        'conductivity': r'conductivity [$S/cm$]',
    }

    def get_lim(*arrays):
        """Get limits for array a."""
        max_a = 0
        min_a = 0
        for array in arrays:
            a = unp.nominal_values(array)
            max_a = np.nanmax([max_a, np.nanmax(a)])
            min_a = np.nanmin([min_a, np.nanmin(a)])
        padding = 0.1 * (max_a - min_a)
        lim = [min_a - padding, max_a + padding]
        return np.array(lim)

    # create dictionary with available modes for each quantity
    exclude = {'2p': ['conductivity'],
               '4p': []}
    modes = {}
    for q in quantities:
        modes[q] = []
        for mode in all_modes:
            if q in exclude[mode]:
                continue
            elif np.isnan(unp.nominal_values(qd[mode][q])).all():
                print(f"Warning: Missing data. Skipping {mode}-{q}.")
            else:
                modes[q].append(mode)

    for qx, qy in couples:
        # decide which modes to iterate over
        modes_ = set(modes[qx])-set(modes[qy])

        # generate axis
        fig, axs = plt.subplots(len(modes_), 1)
        if len(modes_) == 1:
            axs = [axs]

        # plot quantities
        for i, mode in enumerate(modes_):
            axs[i].set_title(f"{qy} vs {qx} ({mode})")
            x = unp.nominal_values(qd[mode][qx])
            y = unp.nominal_values(qd[mode][qy])
            dx = unp.std_devs(qd[mode][qx])
            dy = unp.std_devs(qd[mode][qy])

            axs[i].errorbar(
                x * factor[qx],
                y * factor[qy],
                xerr=dx * factor[qx],
                yerr=dy * factor[qy],
                fmt='o',
                label=f'{mode} data')
            axs[i].set_xlim(factor[qx]*get_lim(x))
            print(qx, qy)
            if qx == 'length' and qy == 'resistance' and mode == '4p':
                print(qx, qy)
                x = np.insert(x, 0, 0)
                y1 = coeffs[0].nominal_value * x + coeffs[1].nominal_value
                axs[i].plot(x*factor[qx], y1*factor[qy], '-', label=f'{mode} fit')
            if qx == 'length' and qy == 'conductivity':
                print(qx, qy)
                y1 = conductivity.nominal_value*np.ones(x.shape)
                axs[i].plot(x*factor[qx], y1*factor[qy], '-', label=f'{mode} fit')
            axs[i].set_ylim(factor[qy]*get_lim(y, y1))

            axs[i].set_xlabel(label[qx])
            axs[i].set_ylabel(label[qy])
            axs[i].legend()
        res_image = os.path.join(res_dir, f"{qy}_vs_{qx}.png")
        plt.tight_layout()
        plt.savefig(res_image)

    if '2p' in modes and '4p' in modes:
        # contact_resistance = []
        # for i, segment in enumerate(segments['2p']):
        #     contact_resistance.append(qd['2p']['resistance'][i] - qd['4p']['resistance'][i])
        # contact_resistance = np.array(contact_resistance)
        contact_resistance = qd['2p']['resistance'] - qd['4p']['resistance']

        fig, ax = plt.subplots()
        ax.set_title("2p-4p resistance vs length")
        x = unp.nominal_values(qd['2p']['length'])
        y = unp.nominal_values(contact_resistance)
        dx = unp.std_devs(qd['2p']['length'])
        dy = unp.std_devs(contact_resistance)
        ax.errorbar(
            x * factor['length'],
            y * factor['resistance'],
            xerr=dx * factor['length'],
            yerr=dy * factor['resistance'],
            fmt='o')
        ax.set_xlim(factor['length']*get_lim(qd['2p']['length']))
        ax.set_ylim(factor['resistance']*get_lim(qd['2p']['resistance']))
        ax.set_xlabel(label['length'])
        ax.set_ylabel(label['resistance'])
        res_image = os.path.join(res_dir, f"{'4p-2p-resistance'}_vs_{'length'}.png")
        plt.tight_layout()
        plt.savefig(res_image)
