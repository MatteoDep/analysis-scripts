# -*- coding: utf-8 -*-
"""fix_data.

Created on 16.05.2022
Author: Matteo De Pellegrin
Email: matteo.dep97@gmail.com
"""


import os
import numpy as np
import pandas as pd
import shutil


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
    bkp_data_dir = data_dir + '_bkp'
    os.makedirs(bkp_data_dir, exist_ok=True)

    nums = np.arange(148, 149)
    names = [f"{chip}_{i}" for i in nums]

    for name in names:
        bkp_path = os.path.join(bkp_data_dir, name + '.xlsx')
        if not os.path.exists(bkp_path):
            shutil.copyfile(os.path.join(data_dir, name + '.xlsx'), bkp_path)
        df = load(name)
        df = modify_gain(df, 'output', 10)
        save(df, name)
