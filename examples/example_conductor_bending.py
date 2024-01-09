import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import strdcable.cable
from strdcable.config import cfg, __PATH_STRDCABLE__

"""
    This example makes a comparison of the bending behavior of a conductor cable (Cardinal) 
    as calculated by the stranded-cable package and experimental results from the literature.
    """

path = __PATH_STRDCABLE__


def get_experimental_results(experimental_df):
    """Plots experimental data from a dataframe and returns the figure handle with plotted data

    Parameters
    ----------
    experimental_df: str
        dataframe with the experimental results

    Returns
    -------
    fig: figure
        figure object where the experimental data is already plotted

    """

    fig = plt.figure()
    ax = fig.gca()

    kappa = np.append([0.], experimental_df.kappa.values)
    M = np.append([0.], experimental_df.M.values)
    ax.plot(kappa, M, '-ok', label='Experimental data')

    kappa_max = np.amax(kappa)
    if kappa_max < 0.0:
        kappa_max = 0.0

    return fig, kappa_max


if __name__ == "__main__":
    # path to csv containing data for cables
    cables_table_path = os.path.join(path, cfg['example_files']['cables_dataset'])
    # dataframe obtained from the csv dataset
    df_cabledata = pd.read_csv(cables_table_path)

    # path to csv with experimental results
    experimental_dataframe = pd.read_csv(os.path.join(path, cfg['example_files']['experimental_bending_data']))

    # The Cardinal conductor is picked
    conductor_name = "CARDINAL"
    cable_indices = df_cabledata['cable'] == conductor_name

    # The Conductor object is created, a definition dataframe is obtained
    cable_type = 'conductor'
    headers = ['material', 'dwires', 'nbwires']
    dict_rename = {}
    sortcols = ['layer']
    cable, df_cable = strdcable.cable.get_cable(cable_type, df_cabledata, cable_indices,
                                                headers, dict_rename, sortcols)

    # Create a plot with experimental values for the bending test
    dict_columns = {'curvature [1/m]': 'kappa', 'bending_moment [N.m]': 'M'}
    experimental_dataframe = experimental_dataframe.rename(columns=dict_columns)
    fig1, kappamax = get_experimental_results(experimental_dataframe)

    ####
    # Numerical estimation of the bending behavior of the Cardinal conductor

    # friction coefficient used in the calculation
    mu = 0.7

    comp = strdcable.cable.SimplifiedBending(df_cable, sortcols=sortcols, drename=dict_rename)
    # setting standard lay length
    comp.set_laylengths(0.5)
    comp.set_friction(mu)
    comp.cable.compute_all()

    # setting tension for the bending test corresponding to experimental conditions compared to here
    H = 40e3

    # calculation and plots
    ax1 = fig1.gca()

    formulation = 'FOTI'
    comp.evaluate(H, formul=formulation)

        # calculation of bending moment as curvature increases
    kappa_num = np.append([0.], np.append(comp.kappa, [kappamax]))
    Mnum = np.zeros(kappa_num.shape)
    for i in range(1, len(kappa_num)):
        Mnum[i] = Mnum[i - 1] + (kappa_num[i] - kappa_num[i - 1]) * comp.EI[i - 1]

        # plot using Matplotlib
    ax1.plot(kappa_num, Mnum, 'x-', label='stranded-cable')

    plt.figure(fig1)
    plt.xlabel("Curvature [1/m]")
    plt.ylabel("Bending moment [N.m]")
    plt.legend()
    plt.show()
    plt.savefig('./example_conductor_bending.png')
