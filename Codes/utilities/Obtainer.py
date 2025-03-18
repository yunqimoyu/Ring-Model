import sys
import os

current_script_path = os.path.dirname(os.path.realpath(__file__))
module_path = os.path.join(current_script_path, '..')
sys.path.append(os.path.abspath(module_path))

from model.RotationalFilters import rotational_filters
from model.LateralConnections import LateralKernels
from model.RingModel import RingModel, PerturbedRingModel
from model.ActiveNeuronsIndices import ActiveNeuronsIndices
from utilities.FuncGenerators import gabor_generator, gaussian_generator
from utilities.FuncOps import counterclockwise_rotate_2d_func, translate_2d_func,  discretize_2d_func
from utilities.Ploter import draw_kernel_diagram, draw_varing_hat_h_inv, draw_signals_and_images
from utilities.VecOps import VectorTrans, VectorsProperty
from config.rotational_filter_paras import *
from config.lateral_connection_paras import *
from config.ring_model_paras import *

import numpy as np


act_exc_state_key = 'all'
act_inh_state_key = 'all'
lateral_cnct_param_key = '07'

class ImageDiscreteParaObtainer:
    @staticmethod
    def para():
        image_discrete_para_1 = {'x_range': (-1, 1), 'y_range': (-1, 1), 'resolution': (64, 64)}
        return image_discrete_para_1

class ImagePieceObtainer:
    @staticmethod
    def gabor(rad=0):
        discretize_paras = ImageDiscreteParaObtainer.para()
        gabor = RotationalFiltersObtainer.gabor(gabor_para=gabor_para)
        gabor = discretize_2d_func(gabor, **discretize_paras)
        return gabor
    
    @staticmethod
    def GaussPoints(centers=[(0.5, 0.5)], sizes=None):
        gaussian = gaussian_generator(gaussian_var=0.1)
        
        discretize_paras = ImageDiscreteParaObtainer.para()
        if centers:
            gaussians = 0
            for center in centers:
                transed_gaussian = translate_2d_func(gaussian, center)
                transed_gaussian = discretize_2d_func(transed_gaussian, **discretize_paras)
                gaussians += transed_gaussian
            gaussian = gaussians
        else:

            gaussian = discretize_2d_func(gaussian, **discretize_paras)
        return gaussian
    
    # @staticmethod
    # def image()

class PerturbationInfluenceObtainer:
    @staticmethod
    def results():
        original_image = ImagePieceObtainer.gabor()
        perturb_image = ImagePieceObtainer.GaussPoints() * 0.01
        perturbed_image = original_image + perturb_image

        gabor_filter = RotationalFiltersObtainer.gabor_filters()

        input_signal = gabor_filter.image2signal(original_image)
        input_signal_p = gabor_filter.image2signal(perturbed_image)
        inhib_input = InputsObtainer.inhib_inputs()

        RM = RingModelObtainer.ring_model()
        RM_output = RM.compute_rates([input_signal, inhib_input])
        RM_output_p = RM.compute_rates([input_signal_p, inhib_input])
        return original_image, perturb_image, perturbed_image, input_signal, input_signal_p, RM_output, RM_output_p
    
    

# get 2d gabor function
class RotationalFiltersObtainer:
    @staticmethod
    def gabor(gabor_para=gabor_para, rad=0):
        '''
        return a ?? (continuous 2d function)??
        '''
        gabor = gabor_generator(**gabor_para)
        rotated_gabor = counterclockwise_rotate_2d_func(gabor, rad=-np.pi/2)
        return rotated_gabor
    
    @staticmethod
    def gabor_filters(gabor_para=gabor_para, rotate_resolution=rotate_resolution, discretize_paras = discretize_paras):
        gabor = RotationalFiltersObtainer.gabor(gabor_para=gabor_para)
        gabor_filters = rotational_filters(rotation_resolution=rotate_resolution, rotation_kernel=gabor, **discretize_paras)
        return gabor_filters

    @staticmethod
    def gabors_in_filters(gabor_para=gabor_para, rotate_resolution=rotate_resolution, discretize_paras = discretize_paras):
        gabor_filters=RotationalFiltersObtainer.gabor_filters(gabor_para=gabor_para, rotate_resolution=rotate_resolution, discretize_paras = discretize_paras)
        return gabor_filters.obtain_filters()

    @staticmethod
    def gabor_filters_matrix(gabor_para=gabor_para, rotate_resolution=rotate_resolution, discretize_paras = discretize_paras):
        gabor_filters=RotationalFiltersObtainer.gabor_filters(gabor_para=gabor_para, rotate_resolution=rotate_resolution, discretize_paras = discretize_paras)
        return gabor_filters.obtain_filters_matrix()

    @staticmethod
    def sig_and_imgs_from_full_gabor_filt_by_svd(gabor_para=gabor_para, rotate_resolution=rotate_resolution, discretize_paras = discretize_paras, take_num=9):
        gabor_filters=RotationalFiltersObtainer.gabor_filters(gabor_para=gabor_para, rotate_resolution=rotate_resolution, discretize_paras = discretize_paras)
        signals, images =  gabor_filters.signals_and_images_from_svd()
        signals = signals[:take_num]
        images = images[:take_num]
        return signals, images
    
    @staticmethod
    def signals_and_images_all_active():
        gabor_filters_matrix = RotationalFiltersObtainer.gabor_filters_matrix()
        print(gabor_filters_matrix.shape)
        U, S, VT = np.linalg.svd(gabor_filters_matrix)
        neuron_orient_prefs=np.linspace(0, 180, 128, endpoint=False)
        fig = draw_signals_and_images(U, VT, neuron_orient_prefs)
        return fig

    @staticmethod
    def signals_and_images_half_active():
        gabor_filters_matrix = RotationalFiltersObtainer.gabor_filters_matrix()
        # gabor_filters_matrix_truncate = gabor_filters_matrix[:64]
        VT = np.load('/root/yanjing/data/subjects/Robustness_RingModel_v2/Data/GF/gabor_filters_part_active/64*64/v_optimal_6.npy')
        # U, S, VT = np.linalg.svd(gabor_filters_matrix_truncate)
        print('VT is of shape', VT.shape)
        VT = VT[:6]
        U = np.matmul(gabor_filters_matrix, VT.T)
        U = VectorTrans.normalize_columns(U)
        neuron_orient_prefs=np.linspace(0, 180, 128, endpoint=False)
        fig = draw_signals_and_images(U, VT, neuron_orient_prefs)
        return fig

class LateralKernelsObtainer:
    @staticmethod
    def get_lateral_kernels(lateral_cnct_param_key = lateral_cnct_param_key):
        lateral_cnct_params = lateral_cnct_params_dict[lateral_cnct_param_key]
        lateral_kernels = LateralKernels(num_neurons=num_neurons, lateral_cnct_params=lateral_cnct_params)
        return lateral_kernels
    
    @staticmethod
    def obtain_hat_h_inv(lateral_cnct_param_key = lateral_cnct_param_key, truncate=8):
        lateral_cnct_params = lateral_cnct_params_dict[lateral_cnct_param_key]
        lateral_kernels = LateralKernels(num_neurons, lateral_cnct_params)
        hat_h_inv = lateral_kernels.get_hat_h_inv() 
        hat_h_inv = hat_h_inv[:truncate]
        return hat_h_inv
    
    @staticmethod
    def varing_hat_h_inv(lateral_cnct_param_keys = ['EN0_v2', 'EN1_v2', 'EN2_v2'], truncate=18):
        h_hat_inv_list = []
        for key in lateral_cnct_param_keys:
            lateral_cnct_params = lateral_cnct_params_dict[key]
            lateral_kernels = LateralKernels(num_neurons, lateral_cnct_params)
            hat_h_inv = lateral_kernels.get_hat_h_inv()
            hat_h_inv = hat_h_inv[:truncate]
            h_hat_inv_list.append(hat_h_inv)
        fig = draw_varing_hat_h_inv(h_hat_inv_list)
        return fig

    @staticmethod
    def compare_exc_and_inh_parts(lateral_cnct_param_key = '08', truncate = 18):
        lateral_cnct_params = lateral_cnct_params_dict[lateral_cnct_param_key]
        lateral_kernels = LateralKernels(num_neurons, lateral_cnct_params)

        kEE = lateral_kernels.excitatory_connection()
        kEIIE = lateral_kernels.recurrent_inhibitory_connection()
        neuron_orient_prefs=NeuronOrientationPrefsObtainer.get_neurons_orients()
        neuron_orient_prefs = neuron_orient_prefs - neuron_orient_prefs[num_neurons//2 - 1]

        hat_kEE = lateral_kernels.hat_excitatory_connection()
        hat_kEIIE = lateral_kernels.hat_recurrent_inhibitory_connection()
        freqs = np.arange(num_neurons)
        freqs = freqs - freqs[num_neurons//2  - 1]

        hat_h_inv_with_EE_only = lateral_kernels.get_hat_h_inv_with_EE_only()
        hat_h_inv_with_EIIE_only = lateral_kernels.get_hat_h_inv_with_EIIE_only()
        hat_h_inv_with_EE_only = hat_h_inv_with_EE_only[:truncate]
        hat_h_inv_with_EIIE_only = hat_h_inv_with_EIIE_only[:truncate]

        fig = draw_kernel_diagram(kEE, kEIIE, neuron_orient_prefs,
                                  hat_kEE, hat_kEIIE, freqs,
                                  hat_h_inv_with_EE_only, hat_h_inv_with_EIIE_only)
      
        return fig

class InputsObtainer:
    @staticmethod
    def test_inputs(input_index=3):
        x = np.linspace(0, 2*np.pi, num_neurons)
        sin = np.sin(x)
        ones = np.ones(num_neurons)
        inputs_set = [[ones, 0*ones],
                      [ones*0.1, ones],
                      [ones, ones],
                      [sin*0.1 + ones*0.9, ones]]
        return inputs_set[input_index]

    @staticmethod
    def inhib_inputs(input_index=2):
        x = np.linspace(0, 2*np.pi, num_neurons)
        ones = np.ones(num_neurons)
        inputs_set = [0*ones,
                      0.5*ones,
                      ones]
        return inputs_set[input_index]

    @staticmethod
    def sin_inputs():
        return
