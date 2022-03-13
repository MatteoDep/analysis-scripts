# -*- coding: utf-8 -*-
"""plot_together.

Plot data together.
"""


import os
import numpy as np
from matplotlib import pyplot as plt
import analysis as a


if __name__ == "__main__":
    data_dir = os.path.join('data', 'light_effect', 'SIC1x')
    res_dir = os.path.join('results', 'light_effect', 'SIC1x')
    os.makedirs(res_dir, exist_ok=True)
    correct_offset = True
    names_list = [
        ['SIC1x_711',  'SIC1x_712'],
        ['SIC1x_713',  'SIC1x_714'],
        ['SIC1x_715',  'SIC1x_716'],
        ['SIC1x_717',  'SIC1x_718'],
        ['SIC1x_719',  'SIC1x_720'],
        ['SIC1x_721',  'SIC1x_722'],
        ['SIC1x_723',  'SIC1x_724'],
        ['SIC1x_725',  'SIC1x_726'],
        ['SIC1x_727',  'SIC1x_728'],
        ['SIC1x_729',  'SIC1x_730'],
        ['SIC1x_731',  'SIC1x_732'],
        ['SIC1x_733',  'SIC1x_734'],
        ['SIC1x_735',  'SIC1x_736'],
        ['SIC1x_737',  'SIC1x_738'],
        ['SIC1x_739',  'SIC1x_740'],
        ['SIC1x_741',  'SIC1x_742'],
    ]
    orders_list = [['vi' for name in names] for names in names_list]

    for names, orders in zip(names_list, orders_list):
        res_path = os.path.join(res_dir, '-'.join(names) + '.png')
        fig, ax = plt.subplots()
        for name, order in zip(names, orders):
            path = os.path.join(data_dir, name + '.xlsx')
            data = a.load_data(path, order=order)
            if correct_offset:
                data['current'] += np.mean(data['current'])
            ax.scatter(data['voltage'], data['current'])
        ax.set_xlabel("Voltage [V]")
        ax.set_ylabel("Current [A]")
        plt.show()
        fig.savefig(res_path)
