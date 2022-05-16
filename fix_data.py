# -*- coding: utf-8 -*-
"""fix_data.

Created on 16.05.2022
Author: Matteo De Pellegrin
Email: matteo.dep97@gmail.com
"""


import os
import numpy as np
import pandas as pd


def load(name):
    global data_dir
    path = os.path.join(data_dir + '_copy', name + '.xlsx')
    df = pd.read_excel(path)
    return df


def save(df, name):
    global data_dir
    path = os.path.join(data_dir, name + '.xlsx')
    df.to_excel(path, index=False)


def modify_gain(df, key, fact):
    i = ['input', 'output', 'time', 'temperature', 'temperature1', 'temperature2'].index(key)
    df.loc[i] *= fact
    return df


if __name__ == "__main__":
    chip = "SPC2"
    data_dir = os.path.join('data', chip)
    if not os.path.exists(data_dir + '_copy'):
        os.popen(f'cp {data_dir} {data_dir}_copy')

    nums = np.arange(45, 74)
    names = [f"{chip}_{i}" for i in nums]

    for name in names:
        df = load(name)
        df = modify_gain(df, 'input', 2)
        save(df, name)
