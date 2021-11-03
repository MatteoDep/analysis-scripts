# -*- coding: utf-8 -*-
"""res_vs_length.

analyse data to plot conductivity over length.
"""


import os
import re
from glob import glob
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

data_dir = 'data'


def p(*args, inner=False):
    """Print in scientific notation."""
    out = ""
    for arg in args:
        if isinstance(arg, str):
            out += " " + arg
        elif isinstance(arg, list):
            out += " " + p(*arg, inner=True)
        else:
            out += " {:.3e}".format(arg)
    if inner:
        return out
    else:
        print(out)


def read_data(path):
    """Read data from files."""
    d = pd.read_excel(path)
    data = {}
    data['input'] = d.loc[0]
    data['output'] = d.loc[1]
    data['temp'] = d.loc[2]
    data['time'] = d.loc[4]

    return data


def plot_data(data, input_label=r'current [A]', output_label=r'voltage [V]'):
    """Plot input/output over time and output over input."""
    time_label = r'time [s]'
    fig, axs = plt.subplots(3, 1)
    plt.tight_layout()

    axs[0].plot(data['time'], data['input'], 'o-')
    axs[0].set_xlabel(time_label)
    axs[0].set_ylabel(input_label)

    axs[1].plot(data['time'], data['output'], 'o-')
    axs[1].set_xlabel(time_label)
    axs[1].set_ylabel(output_label)

    axs[2].plot(data['input'], data['output'], 'o-')
    axs[2].set_xlabel(input_label)
    axs[2].set_ylabel(output_label)

    return fig, axs


def main():
    """Start analysis."""
    resistances = []
    lengths = []
    unit_length = 5e-6
    pattern = os.path.join(data_dir, '4p*.xlsx')
    for path in glob(pattern):
        name = os.path.splitext(os.path.basename(path))[0]
        data = read_data(path)

        # fig, axs = plot_data(data)
        # fig.suptitle(name)
        # plt.show()

        A = np.vstack([data['input'], np.ones(data['input'].shape)]).T
        resistance, offset = np.linalg.lstsq(A, data['output'], rcond=None)[0]
        resistances.append(resistance)

        m = re.match(r'4p_.*_([0-9]*)-([0-9]*)', name, re.M)
        length = (int(m.group(2)) - int(m.group(1))) * unit_length
        lengths.append(length)

        p("---", name, "---")
        p("length:", length, "m")
        p("resistance:", resistance, "")
        p("offset:", offset, "V\n")

    fig, ax = plt.subplots()
    ax.set_title('resistance over length')
    ax.plot(lengths, resistances, 'o')
    ax.set_xlabel(r'length [m]')
    ax.set_ylabel(r'resistance [\omega]')
    plt.show()

    return 0


if __name__ == "__main__":
    exit(main())
