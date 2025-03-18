import sys
import os

current_script_path = os.path.dirname(os.path.realpath(__file__))
module_path = os.path.join(current_script_path, '..')
sys.path.append(os.path.abspath(module_path))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import random

from config.lateral_connection_paras import lateral_cnct_params_dict
from config.ring_model_paras import *
from model.ActiveNeuronsIndices import ActiveNeuronsIndices
from model.LateralConnections import LateralKernels
from model.RingModel import PerturbedRingModel

# Config neuron_num, lateral connections and active state


def gain_curves_of_different_shape(lateral_cnct_param_keys, plot_freq, fig_title, method='h_inv'):
    """
    method can be choose as singualr_value or h_inv
    """
    nrows = len(lateral_cnct_param_keys)
    fig, ax = plt.subplots(nrows=nrows, figsize = (4, nrows*3), dpi=300)
    ax = np.atleast_1d(ax)
    title_prefixes = ['I. ', 'II. ', 'III. ']
    for i, lateral_cnct_param_key in enumerate(lateral_cnct_param_keys):
        lateral_cnct_params = lateral_cnct_params_dict[lateral_cnct_param_key]
        alphaEE = lateral_cnct_params['kEE']['para']['amplitude']
        alphaIE = lateral_cnct_params['kEI']['para']['amplitude']
        lateral_kernels = LateralKernels(num_neurons, lateral_cnct_params)
        if method == 'h_inv':
            hat_h_inv = lateral_kernels.get_hat_h_inv()
            ax[i].plot(range(plot_freq), hat_h_inv[:plot_freq], marker='o')
        elif method == 'singular_value':
            perturbed_model = PerturbedRingModel(num_neurons=num_neurons, lateral_kernels=lateral_kernels)
            act_exc_idx = ActiveNeuronsIndices.create(orientation_prefs=perturbed_model.get_neurons_orient_prefs(),
                                                    angle_range=act_exc_range)
            act_inh_idx = ActiveNeuronsIndices.create(orientation_prefs=perturbed_model.get_neurons_orient_prefs(),
                                                    angle_range=act_inh_range)
            perturbed_model.set_act_idx(act_exc_idx, act_inh_idx)
            df = perturbed_model.USV_df()
            filtered_df = df[df['dom_freq'] < plot_freq]
            sorted_df = filtered_df.sort_values(by='dom_freq')
            ax[i].plot(sorted_df['dom_freq'], sorted_df['s'], marker='o')
        ax[i].set_xlabel(f'$\\xi$')
        ax[i].set_title(fr'{title_prefixes[i]}$\alpha_{{EE}}={alphaEE},\ \alpha_{{IE}}=\alpha_{{EI}}={alphaIE}$')
        ax[i].xaxis.set_major_locator(MaxNLocator(integer=True))
        ax[i].set_ylim(0, 2.8)
        # Remove x-ticks except for the last plot
        if i != 2:
            ax[i].set_xticks([])
        else:
            ax[i].set_xlabel(f'$\\xi$')
    plt.tight_layout()
    plt.savefig(f'{save_path}/{fig_title}_layout2.pdf')
    return

lateral_cnct_param_keys = ['09', '10', '11']
act_exc_range = act_exc_ranges['all']
act_inh_range = act_inh_ranges['all']
save_path = f'{current_script_path}/../../Data/RM'
os.makedirs(f'{save_path}', exist_ok=True)
gain_curves_of_different_shape(lateral_cnct_param_keys, plot_freq=18, fig_title = 'different_shape_gain_curve_test_unify_y_axis_scale', method='h_inv')
    


