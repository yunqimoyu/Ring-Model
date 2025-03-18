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
from utilities.Ploter import draw_sinusoids, draw_sinusoids_singularvectors, draw_sinusoids_amp, draw_singularvectors, draw_singularvalues, circulant_matrix_visualization


# Config neuron_num, lateral connections and active state

lateral_cnct_param_key = '07'
act_range_key = 'all'
lateral_cnct_params = lateral_cnct_params_dict[lateral_cnct_param_key]
act_exc_range = act_exc_ranges[act_range_key]
act_inh_range = act_inh_ranges['all']

special_act_exc = '' # Choose from the ['', 'keep_even', 'keep_odd', 'random_keep']
throw_probability = 0
save_path = f'{current_script_path}/../../Data/RM/{lateral_cnct_param_key}/{act_range_key}'

if special_act_exc:
    save_path = f'{save_path}/{special_act_exc}'
    if special_act_exc == 'random_keep':
        save_path = f'{save_path}/{throw_probability}/{random.randint(1, 10000)}'
else:
    save_path = f'{save_path}/all_keep'

os.makedirs(f'{save_path}', exist_ok=True)

# Construct the model

lateral_kernels = LateralKernels(num_neurons, lateral_cnct_params)
perturbed_model = PerturbedRingModel(num_neurons=num_neurons, lateral_kernels=lateral_kernels)
act_exc_idx = ActiveNeuronsIndices.create(orientation_prefs=perturbed_model.get_neurons_orient_prefs(),
                                          angle_range=act_exc_range
                                          )
## Adjust the active state, create save path
act_exc_idx.special_keep(special_act_exc)

act_inh_idx = ActiveNeuronsIndices.create(orientation_prefs=perturbed_model.get_neurons_orient_prefs(),
                                          angle_range=act_inh_range
                                          )
perturbed_model.set_act_idx(act_exc_idx, act_inh_idx)

# Get singular vectors
H = perturbed_model.get_effect_parts()
H_inv = np.linalg.inv(H)
df = perturbed_model.USV_df()
df.to_pickle(f'{save_path}/USV_df')

# Search truncated sinusoids with maximal output
# best_sinusoids = perturbed_model.best_sinusoids()

# Visualization
x = perturbed_model.get_neurons_orient_prefs()

# circulant_matrix_visualization(H, H_inv, save_path)

# draw_sinusoids(x, best_sinusoids, save_path)
# draw_sinusoids_amp(best_sinusoids, plot_freq=10, save_path = save_path)

draw_sinusoids_singularvectors(df, x, range(10), save_path=save_path) # singularvectors or eigenvectors?
# draw_singularvectors(perturbed_model.get_neurons_orient_prefs(), df, save_path) # draw singular vectors seperately and integrate into an animation
# draw_singularvalues(df, plot_freq=10, save_path=save_path)