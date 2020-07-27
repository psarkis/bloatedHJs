'''
This module takes samples from MCMC as input
and plots several diagnostic plots:
- trace plots 
- corner plots 
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import warnings

import corner

def trace_plot(samples, labels, identifier=None, save_plots=False, show_plots=True):
    ii = samples.shape[2]

    fig, axes = plt.subplots(ii, 1, sharex=True, figsize=(8, 9))
    plt.suptitle('{0}'.format(identifier))
    
    for i in range(ii):
        axes[i].plot(samples[:, :, i].T, color="k", alpha=0.4)
        axes[i].yaxis.set_major_locator(MaxNLocator(5))
        axes[i].set_ylabel(labels[i])

    if save_plots:
        if identifier is not None:
            plt.savefig('figures/{0}_traces.pdf'.format(identifier))
        else:
            warnings.warning('Name of the file is not specified ... the plot was not saved')

    if show_plots:
        plt.show()

def corner_plot(samples, data, derived_params=False, selected_columns=None, identifier=None, save_plots=True, show_plots=True, **kwargs):
    '''
    This function makes the cornet plot of:
        - the fitted parameters
        - the derived parameters 
        OR
        - columns of choice entered by the user 
    '''
    if selected_columns is None:
        if not derived_params:
            columns    = ['Lumi', 'Epsilon', 'Zhomo', 'Teq', 'Mp', 'Rp']
            truths     = [-1, -1, -1, data['Teq'], data['Mp'], data['Rp']]
            identifier = identifier + '_fittedparams'
        if derived_params:
            columns    = ['Tint', 'Prcb', 'Trcb']
            truths     = [-1, -1, -1]
            identifier = identifier + '_derivedparams'
    else:
        columns    = selected_columns
        identifier = identifier

    X = samples[columns]

    if selected_columns is None:
        corner.corner(X, labels=columns, quantiles=[0.16,0.5,0.84], show_titles=True, truths=truths, **kwargs)
    else:
        corner.corner(X, labels=columns, quantiles=[0.16,0.5,0.84], show_titles=True, **kwargs)

    if save_plots is True:
        plt.savefig('figures/{0}_corner.pdf'.format(identifier), bbox_inches='tight', dpi=400)

    if show_plots:
        plt.show()







