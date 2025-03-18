import sys
import os

current_script_path = os.path.dirname(os.path.realpath(__file__))
module_path = os.path.join(current_script_path, '..')
sys.path.append(os.path.abspath(module_path))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

from config.lateral_connection_paras import lateral_cnct_params_dict
from config.ring_model_paras import *
from model.ActiveNeuronsIndices import ActiveNeuronsIndices
from model.LateralConnections import LateralKernels
from model.RingModel import PerturbedRingModel
from utilities.VecOps import VectorsProperty
from utilities.Ploter import draw_sinusoids, draw_sinusoids_singularvectors, draw_sinusoids_amp, draw_singularvectors, draw_singularvalues, circulant_matrix_visualization

# Config neuron_num, lateral connections and active state


def plot_gain_curves(lateral_cnct_param_key, act_range_keys, special_act_exc, throw_probability, gain_over='singular_val', plot_freq=10):
    fig, ax = plt.subplots()
    for act_range_key in act_range_keys:
        act_exc_range = act_exc_ranges[act_range_key]
        lateral_kernels = LateralKernels(num_neurons, lateral_cnct_params)
        perturbed_model = PerturbedRingModel(num_neurons=num_neurons, lateral_kernels=lateral_kernels)
        act_exc_idx = ActiveNeuronsIndices.create(orientation_prefs=perturbed_model.get_neurons_orient_prefs(),
                                                angle_range=act_exc_range)
        act_exc_idx.special_keep(special_act_exc, throw_probability)
        act_inh_idx = ActiveNeuronsIndices.create(orientation_prefs=perturbed_model.get_neurons_orient_prefs(),
                                                angle_range=act_inh_range)
        perturbed_model.set_act_idx(act_exc_idx, act_inh_idx)
        
        if gain_over == 'sin':
            best_sinusoids = perturbed_model.best_sinusoids()
            ax.plot(best_sinusoids['freq'], best_sinusoids['amp'], marker='o', label=act_range_key)
        elif gain_over == 'singular_val':
            df = perturbed_model.USV_df()
            filtered_df = df[df['dom_freq'] < plot_freq]
            sorted_df = filtered_df.sort_values(by='dom_freq')
            ax.plot(sorted_df['dom_freq'], sorted_df['s'], marker='o', label=act_range_key)
        ax.legend()
        ax.set_title('Gain curve')
    # plt.savefig(f'{save_path}/gain_curves_over_{gain_over}_{special_act_exc}_tp_{throw_probability}.pdf')
    return ax

act_range_keys = ['all', 'third forths', 'half', 'one forth']
# special_act_exc = 'keep_even'
special_act_exc = ''
throw_probability = 0
# gain_over = 'singular_val'
gain_over = 'sin'
axs = []
for lateral_cnct_param_key in ['09', '10', '11']:
# for lateral_cnct_param_key in ['07']:
    lateral_cnct_params = lateral_cnct_params_dict[lateral_cnct_param_key]
    act_inh_range = act_inh_ranges['all']
    save_path = f'{lateral_cnct_param_key}'
    os.makedirs(f'{save_path}', exist_ok=True)
    ax = plot_gain_curves(lateral_cnct_param_key, act_range_keys, special_act_exc, throw_probability, gain_over=gain_over)
    axs.append(ax)

fig, axes = plt.subplots(ncols=3, figsize=(12, 3), dpi=300)
i = 0
for ax, original_ax in zip(axes, axs):
    colors = []
    for line in original_ax.get_lines():
        ax.plot(line.get_xdata(), line.get_ydata(), label=line.get_label(), color=line.get_color(), marker='o', markersize=5)
        colors.append(line.get_color())
        ax.set_xlabel(f'$\\xi$')
        ax.set_ylim(0, 2.8)
        ax.set_ylabel(f'Hz')
        # ax.set_ylabel(f'gain')


        # Only show the y-ticks for the first subplot
        if i > 0:
            ax.tick_params(labelleft=False)
    i+=1
    # ax.legend()

labels = ['all', 'third forths', 'half', 'one forth']
# colors = ['blue', 'orange', 'green', 'red']
fig.legend(labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=4, frameon=False)
plt.tight_layout()
# plt.subplots_adjust(top=0.90)  
plt.savefig(f'{current_script_path}/../../Data/RM/gain_curves_compare_2_{special_act_exc}_legend_adjust_3.pdf')
    


