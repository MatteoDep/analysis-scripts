# -*- coding: utf-8 -*-
"""light_calibration.

Perform light calibration.
"""


import os
import pandas as pd
from matplotlib import pyplot as plt
from pint import UnitRegistry


ur = UnitRegistry()
ur.setup_matplotlib()


if __name__ == "__main__":
    data_dir = os.path.join("data", "light-calibration")
    res_dir = os.path.join("results", "light-calibration")
    freq = 532

    os.makedirs(res_dir, exist_ok=True)
    data_file = os.path.join(data_dir, f"{freq}.csv")
    data = pd.read_csv(data_file)

    fig, ax = plt.subplots()
    ax.set_title("Knot Ticks Dependence")
    ax.set_xlabel("Tick")
    ax.set_ylabel("Power [mW]")
    for i, dist in enumerate(data['distance']):
        y = data.values[i, 1:]
        x = range(2, 2 + len(y))
        ax.plot(x, y, label=f"{dist*ur['cm']:~P}")
    plt.legend()
    path = os.path.join(res_dir, f"{freq}_tick_dependence.png")
    plt.savefig(path)

    fig, ax = plt.subplots()
    ax.set_title("Distance Dependence")
    ax.set_xlabel("Distance [cm]")
    ax.set_ylabel("Power [mW]")
    for key in data:
        if key != 'distance':
            ax.plot(data['distance'], data[key], label=f"tick {key}")
    plt.legend()
    path = os.path.join(res_dir, f"{freq}_distance_dependence.png")
    plt.savefig(path)
