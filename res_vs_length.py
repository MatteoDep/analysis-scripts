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
    data['time'] = d.loc[2]
    data['temp'] = d.loc[3]

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

def compute_length(electrode1, electrode2, contact_width=1e-5, contact5_width=1e-4, spacing=4e-5):
    """Compute cable length between electrodes."""
    spacing_num = electrode2 - electrode1
    contact5_num = int((electrode2-1) / 5) - int((electrode1-1) / 5)
    contact_num = spacing_num - 1 - contact5_num
    print("spacing_num:", spacing_num)
    print("contact_num:", contact_num)
    print("contact5_num:", contact5_num)

    length = (spacing_num * spacing) + \
        (contact_num * contact_width) + \
        (contact5_num * contact5_width)
    return length


def main():
    """Start analysis."""
    contact_width=1e-5
    contact5_width=1e-4
    spacing=4e-5

    resistances = []
    lengths = []
    pattern = os.path.join(data_dir, '4p*.xlsx')
    for path in np.sort(glob(pattern)):
        name = os.path.splitext(os.path.basename(path))[0]
        data = read_data(path)

        # fig, axs = plot_data(data)
        # fig.suptitle(name)
        # plt.show()

        A = np.vstack([data['input'], np.ones(data['input'].shape)]).T
        resistance, offset = np.linalg.lstsq(A, data['output'], rcond=None)[0]
        resistances.append(resistance)

        m = re.match(r'4p_.*_([0-9]*)-([0-9]*)', name, re.M)
        length = compute_length(int(m.group(1)), int(m.group(2)), contact_width=contact_width,
                                                    contact5_width=contact5_width, spacing=spacing)
        lengths.append(length)

        p("---", name, "---")
        p("length:", length, "m")
        p("resistance:", resistance, "")
        p("offset:", offset, "V\n")

    fig, ax = plt.subplots()
    ax.set_title("resistance over length")
    ax.plot(lengths, resistances, 'o')
    ax.set_xlabel(r"length [m]")
    ax.set_ylabel(r"resistance [$\Omega$]")
    res_image = os.path.join(res_dir, "res_vs_length.png")
    plt.savefig(res_image)

    return 0


if __name__ == "__main__":
    data_dir = 'data/SIC1x'
    res_dir = 'results'

    os.makedirs(res_dir, exist_ok=True)

    exit(main())
