# -*- coding: utf-8 -*-
"""analysis/visualise.

Define functions to plot results and data.
"""


import numpy as np
from uncertainties import unumpy as unp
import uncertainties as u
from matplotlib import pyplot as plt


class Visualise:
    """Plot data and results with ease."""

    @staticmethod
    def adjust_lim(lim, array):
        """Get limits for array a."""
        oldlim = lim
        min_a = np.nanmin(array)
        max_a = np.nanmax(array)
        lim[0] = np.nanmin([min_a, lim[0]])
        lim[1] = np.nanmax([max_a, lim[1]])
        padding = 0.1 * (max_a - min_a)
        lim = [lim[i] if lim[i] == oldlim[i] else lim[i] + (-1)**(i+1) * padding for i in range(2)]
        return np.array(lim)

    def plot(ax, qx, qy, set_labels=True, **kwargs):
        """Plot quantity qy vs quantity qx."""
        ax.set_title(f"{qy.qtype} vs {qx.qtype}")
        x, dx = qx.converted().get_raw()
        y, dy = qy.converted().get_raw()

        ax.errorbar(x, y, xerr=dx, yerr=dy, fmt='o', label='data', **kwargs)
        ax.set_xlim(__class__.adjust_lim(ax.get_xlim(), x))
        ax.set_ylim(__class__.adjust_lim(ax.get_ylim(), y))

        if set_labels:
            ax.set_xlabel(f"{qx.qtype} [${qx.latex_unit}$]")
            ax.set_ylabel(f"{qy.qtype} [${qy.latex_unit}$]")
        ax.legend()
