# -*- coding: utf-8 -*-
"""analysis/visualise.

Define functions to plot results and data.
"""


import configparser
import numpy as np
from uncertainties import unumpy as unp
import uncertainties as u


class Quantity:
    """Define quantity properties."""

    CONFIG = {}

    def __init__(self, data, qtype=None, sigma=None):
        """Create object of a certain quantity type."""
        # Convert to appropriate data type
        if isinstance(data, u.UFloat):
            self.data = data
        elif isinstance(data, (np.ndarray, list)):
            try:
                if isinstance(data[0], u.UFloat):
                    if isinstance(data, list):
                        self.data = np.array(data)
                    else:
                        self.data = data
                else:
                    if sigma is None:
                        self.data = np.array(data)
                        self.has_sigma = False
                    else:
                        self.data = unp.uarray(data, sigma)
            except IndexError:
                print("Warning: Creating quantity with empty data.")
                self.data = None
        else:
            if sigma is None:
                self.data = data
                self.has_sigma = False
            else:
                self.data = u.ufloat(data, sigma)

        self.qtype = "generic_quantity" if qtype is None else qtype
        if qtype is None:
            self.unit = None
            self.conv_factor = None
        elif qtype not in self.CONFIG:
            print(f"Warning: quantity type {qtype} not defined. Use {__name__}.load(config) to define quantities.")
        else:
            self.unit = self.CONFIG[qtype]['unit']
            self.conv_factor = self.CONFIG[qtype]['conv_factor']
            if 'latex_unit' in self.CONFIG[qtype]:
                self.latex_unit = self.CONFIG[qtype]['latex_unit']
            else:
                self.latex_unit = self.unit

    @staticmethod
    def load_config(config):
        """
        Set attributes from configuration dictionary.

        :param config: ini configuration file or dictionary.
        """
        if isinstance(config, str):
            config = configparser.read(open(config))
        for attr_key, attr_value in config.items():
            __class__.CONFIG[attr_key] = attr_value

    @staticmethod
    def dump_config(path=None):
        """Dump configs to a dict."""
        if path is not None:
            with open(path, 'w') as f:
                configparser.write(__class__.CONFIG, f)
        return __class__.CONFIG

    @staticmethod
    def display_config():
        """Display Configuration values."""
        print("\nConfigured quantities:")
        for key, val in __class__.dump_config().items():
            print(f"{key:20} {val}")
        print("\n")

    def get_nominal(self):
        """Get nominal value of quantity."""
        if isinstance(self.data, np.ndarray):
            nominal = unp.nominal_values(self.data)
        else:
            nominal = self.data.nominal_value()
        return nominal

    def get_sigma(self):
        """Get standard deviation of quantity."""
        if isinstance(self.data, np.ndarray):
            nominal = unp.std_devs(self.data)
        else:
            nominal = self.data.std_dev()
        return nominal

    def get_raw(self):
        """Get both nominal value and standard deviation."""
        return self.get_nominal, self.get_sigma

    def converted(self):
        """Get data in its units."""
        return __class__(self.data * self.conv_factor, self.qtype)

    def __str__(self):
        """Format string."""
        fmt = r"{} {}"
        return fmt.format(self.converted().data, self.unit)

    def __add__(self, other):
        """Addition operator."""
        qtype = self.qtype
        if isinstance(other, __class__):
            data = self.data + other.data
            if qtype != other.qtype:
                qtype = None
        else:
            data = self.data + other
        return __class__(data, qtype)

    def __radd__(self, other):
        """Addition operator reflected."""
        data = other + self.data
        return __class__(data, self.qtype)

    def __iadd__(self, other):
        """Addition operator reflected."""
        if isinstance(other, __class__):
            self.data += other.data
            if self.qtype != other.qtype:
                print("Warning: mixing {other.qtype} quantity to a {self.qtype} quantity.")
        else:
            self.data += other
        return self

    def __sub__(self, other):
        """Subtraction operator."""
        qtype = self.qtype
        if isinstance(other, __class__):
            data = self.data - other.data
            if self.qtype != other.qtype:
                qtype = None
        else:
            data = self.data - other
        return __class__(data, qtype)

    def __rsub__(self, other):
        """Subtraction operator reflected."""
        data = other - self.data
        return __class__(data, self.qtype)

    def __isub__(self, other):
        """Subtraction operator reflected."""
        if isinstance(other, __class__):
            self.data -= other.data
            if self.qtype != other.qtype:
                print("Warning: mixing {other.qtype} quantity to a {self.qtype} quantity.")
        else:
            self.data -= other
        return self

    def __mul__(self, other):
        """Addition operator."""
        qtype = self.qtype
        if isinstance(other, __class__):
            data = self.data * other.data
            if qtype != other.qtype:
                qtype = None
        else:
            data = self.data * other
        return __class__(data, qtype)

    def __rmul__(self, other):
        """Addition operator reflected."""
        data = other * self.data
        return __class__(data, self.qtype)

    def __imul__(self, other):
        """Addition operator reflected."""
        if isinstance(other, __class__):
            self.data *= other.data
            if self.qtype != other.qtype:
                print("Warning: mixing {other.qtype} quantity to a {self.qtype} quantity.")
        else:
            self.data *= other
        return self

    def __truediv__(self, other):
        """Addition operator reflected."""
        qtype = self.qtype
        if isinstance(other, __class__):
            data = self.data / other.data
            if qtype != other.qtype:
                qtype = None
        else:
            data = self.data / other
        return __class__(data, qtype)

    def __rtruediv__(self, other):
        """Addition operator."""
        data = other / self.data
        return __class__(data, self.qtype)

    def __itruediv__(self, other):
        """Addition operator reflected."""
        if isinstance(other, __class__):
            self.data /= other.data
            if self.qtype != other.qtype:
                print("Warning: mixing {other.qtype} quantity to a {self.qtype} quantity.")
        else:
            self.data /= other
        return self

    def __pow__(self, other):
        """Addition operator."""
        qtype = self.qtype
        if isinstance(other, __class__):
            data = self.data ** other.data
            if qtype != other.qtype:
                qtype = None
        else:
            data = self.data ** other
        return __class__(data, qtype)

    def __rpow__(self, other):
        """Addition operator reflected."""
        data = other ** self.data
        return __class__(data, self.qtype)

    def __ipow__(self, other):
        """Addition operator reflected."""
        if isinstance(other, __class__):
            self.data **= other.data
            if self.qtype != other.qtype:
                print("Warning: mixing {other.qtype} quantity to a {self.qtype} quantity.")
        else:
            self.data **= other
        return self

    def __getitem__(self, index):
        """Get item operator."""
        return __class__(self.data[index], self.qtype)

    def __setitem__(self, index, value):
        """Set item operator."""
        return NotImplemented

    def __len__(self):
        """Get length of data."""
        return len(self.data)


if __name__ == "__main__":
    config = {
            'length': {
                'unit': 'μm',
                'conv_factor': 1e6
                },
            'resistance': {
                'unit': 'MΩ',
                'conv_factor': 1e-6,
                },
            'conductivity': {
                'unit': 'S/cm',
                'conv_factor': 1e-2,
                }
            }

    Quantity.load_config(config)
    Quantity.display_config()

    data_variants = [
                [[10, 5], [0.1, 0.05]],
                [unp.uarray([10, 5], [0.1, 0.05]), None],
                [np.array([10, 5]), np.array([0.1, 0.05])],
                [np.array([10, 5]), None],
                [[u.ufloat(10, 0.1), u.ufloat(5, 0.05)], None],
                [10, 0.1],
                [10, None],
                [u.ufloat(10, 0.1), None],
            ]

    lengths = []
    for data, sigma in data_variants:
        length = Quantity(data, 'length', sigma=sigma)
        lengths.append(length)
        print(length)

    print("lengths[0] + lengths[1] = ", lengths[0] + lengths[1])
    print("lengths[0] + 1 = ", lengths[0] + 1)
    print("1 + lengths[0] = ", 1 + lengths[0])
    lengths[0] += lengths[1]
    print("lengths[0] += lengths[1]; lengths[0] = ", lengths[0])
