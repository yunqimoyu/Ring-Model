import sys
import os

current_script_path = os.path.dirname(os.path.realpath(__file__))
module_path = os.path.join(current_script_path, '..')
sys.path.append(os.path.abspath(module_path))

import numpy as np
import pandas as pd
from scipy.optimize import fsolve

from model.OrientPrefs import NeuronOrientationPrefs
from model.ActiveNeuronsIndices import ActiveNeuronsIndices
from utilities.SignalGenerator import sinSignal
from utilities.VecOps import VectorsProperty
from utilities.FuncOps import counterclockwise_rotate_2d_func


class RingModel:
    def __init__(self, num_neurons, lateral_kernels):
        self._num_neurons = num_neurons
        self._lateral_kernels = lateral_kernels
        
        return

    
    def compute_rates_all_neurons_active(self, inputs):
        I_E, I_I = inputs

        KEE = self._lateral_kernels.EE_kernel.matrix_form
        KEI = self._lateral_kernels.EI_kernel.matrix_form
        KIE = self._lateral_kernels.IE_kernel.matrix_form
        H = np.eye(self._num_neurons)  -  KEE  + KEI @ KIE
        print('H is', H)
        H_inv = np.linalg.inv(H) 
        print('H has inverse H inv', H)

        I_rec = I_E - KIE @ I_I
        print('I_rec is', I_rec)
        r_E = H_inv @ I_rec
        print('r_E is', r_E)
        r_I = I_I + KIE @ r_E
        r_solution = [r_E, r_I]
        return r_solution

    def equations(self, r_E, r_I, I_E, I_I, KEE, KEI, KIE):
        conv_E = I_E + KEE @ r_E  - KEI @ r_I
        conv_I = I_I  + KIE @ r_E

        new_r_E = np.maximum(0, conv_E)
        new_r_I = np.maximum(0, conv_I)
        return new_r_E, new_r_I
    
    def compute_rates(self, inputs, init_scale=1e-4, max_iter=1000, tol=1e-8, lr = 1):
        I_E, I_I = inputs

        KEE = self._lateral_kernels.EE_kernel.matrix_form
        KEI = self._lateral_kernels.EI_kernel.matrix_form
        KIE = self._lateral_kernels.IE_kernel.matrix_form
        print('kEE is', KEE)

        r_E = np.ones(self._num_neurons) * init_scale
        r_I = np.ones(self._num_neurons) * init_scale

        # r_E = np.random.rand(self._num_neurons) * init_scale
        # r_I = np.random.rand(self._num_neurons) * init_scale

        for i in range(max_iter):
            print(r_E[0])
            new_r_E, new_r_I = self.equations(r_E, r_I, I_E, I_I, KEE, KEI, KIE)
            if np.max(np.abs(new_r_E - r_E)) < tol and np.max(np.abs(new_r_I - r_I)) < tol:
                print(f'Converged after {i+1} iterations.')
                return [new_r_E, new_r_I]
            r_E, r_I = (1-lr) * r_E + lr * new_r_E, (1-lr) * r_I + lr * new_r_I
        print('Did not converge.')
        return [new_r_E, new_r_I]
    
    def equations_2(self, r, I_E, I_I, KEE, KEI, KIE):
        print(r.shape)
        n = len(r)//2
        r_E = r[:n]
        r_I = r[n:]

        conv_E = I_E + KEE @ r_E - KEI @ r_I
        conv_I = I_I + KIE @ r_E

        new_r_E = np.maximum(0, conv_E)
        new_r_I = np.maximum(0, conv_I)

        residual_E = new_r_E - r_E
        residual_I = new_r_I - r_I
        return np.concatenate((residual_E, residual_I))

    
    def compute_rates_2(self, inputs, init_scale = 1e-3):
        I_E, I_I = inputs

        KEE = self._lateral_kernels.EE_kernel.matrix_form
        KEI = self._lateral_kernels.EI_kernel.matrix_form
        KIE = self._lateral_kernels.IE_kernel.matrix_form

        E_neurons_rates = np.random.rand(self._num_neurons) * init_scale
        I_neurons_rates = np.random.rand(self._num_neurons) * init_scale
        print(E_neurons_rates.shape)
        print(I_neurons_rates.shape)
        r_initial = np.array([E_neurons_rates, I_neurons_rates])

        r_solution = fsolve(lambda r: self.equations(r, I_E, I_I, KEE, KEI, KIE), r_initial.flatten())
        n = len(r_solution)//2
        r_E = r_solution[:n]
        r_I = r_solution[n:]
        return [r_E, r_I] 


    

class PerturbedRingModel:
    """
    The perturbed system of Ring Model with ReLU as activation function.
    """
    def __init__(self, num_neurons, lateral_kernels, 
                 act_exc_idx=None, act_inh_idx=None):
        """
        act_exc_idx: index vector over which neurons are active
        """

        self._num_neurons = num_neurons

        self._lateral_kernels = lateral_kernels

        self._orient_prefs = NeuronOrientationPrefs(num_neurons).neuron_orientation_prefs

        self._act_exc_idx = act_exc_idx
        self._act_inh_idx = act_exc_idx

        return
    
    def get_adversarial_perturbation():
        return

    def compute_H(self):
        GI = self._act_inh_idx.get_matrix_form(self._num_neurons)
        KEE = self._lateral_kernels.EE_kernel.matrix_form
        KEI = self._lateral_kernels.EI_kernel.matrix_form
        KIE = self._lateral_kernels.IE_kernel.matrix_form
        self.H = np.eye(self._num_neurons) - KEE  + KEI @ GI @ KIE
        return self.H
    
    def get_lateral_kernels(self):
        return self._later_kernels

    def get_effect_parts(self):
        if not hasattr(self, 'H'):
            self.compute_H()
        self.effect_H = self.H[self._act_exc_idx.get_indices(), :][:, self._act_exc_idx.get_indices()] 
        return self.effect_H

    def get_effect_H_inv(self):
        if not hasattr(self, 'effect_H'):
            self.get_effect_parts()
        self.effect_H_inv = np.linalg.inv(self.effect_H)
        return self.effect_H_inv
    
    def USV_df(self):
        if not hasattr(self, 'effect_H_inv'):
            self.get_effect_H_inv()
        H_inv = self.effect_H_inv
        U, S, VT = np.linalg.svd(H_inv)
        # Attach frequencies and save data
        singular_vector_dict = {}
        VT = self.restore_vectors(VT.T).T

        for i in range(U.shape[1]):
            fft_results = np.fft.fft(VT, axis=1)
            dom_freq = VectorsProperty.dominant_frequencies(fft_results.T)
            data = {'ut': list(U.T), 's': list(S), 'vt': list(VT), 'dom_freq': dom_freq, 'fft_result': list(np.abs(fft_results))}
            df = pd.DataFrame(data)
        self.USV_df = df
        return df
    
    def set_act_idx(self, act_exc_idx, act_inh_idx):
        self._act_exc_idx = act_exc_idx
        self._act_inh_idx = act_inh_idx
    
    def truncate_signal(self, signal):
        truncated_signal = signal[self._act_exc_idx.get_indices()]
        return truncated_signal
    
    def truncated_signal_pass(self, truncated_signal):
        if not hasattr(self, 'effect_H_inv'):
            self.get_effect_H_inv()
        output = self.effect_H_inv @ truncated_signal
        return output
    
    def signal_pass(self, signal):
        truncated_signal = self.truncate_signal(signal)
        output = self.truncated_signal_pass(truncated_signal)
        output = self.restore_vector(output)
        return output

    def restore_vector(self, vector):
        """
        Complete the vectors (which are given as columns in a 2d numpy array)
        """
        restored_vector = np.zeros(self._num_neurons)
        vector = np.array(vector)
        filtered_indices = np.array(self._act_exc_idx.get_indices())
        print(filtered_indices.shape, vector.shape)
        restored_vector[filtered_indices] = vector
        return restored_vector

    def restore_vectors(self, vectors):
        """
        Complete the vectors (which are given as columns in a 2d numpy array)
        """
        restored_vectors = np.zeros([self._num_neurons, vectors.shape[1]])
        filtered_indices = self._act_exc_idx.get_indices()
        for index, row in zip(filtered_indices, list(vectors)):
            # print(index)
            # print(row)
            restored_vectors[index] = row
        return restored_vectors
    
    def get_act_exc_idx(self):
        return self._act_exc_idx.get_indices()

    def get_freq(self):
        """
        Only applies to 
        """
        act_exc_idx = self._act_exc_idx.get_indices()
        print("Get the frequency. Not that this only apply to the case when neurons are continuously activated.")
        freq = np.arange((len(act_exc_idx) + 2)//2)
        return freq

    def get_modified_freq(self):
        """
        Only applies to 
        """
        act_exc_idx = self._act_exc_idx.get_indices()
        freq = self.get_freq()
        modified_freq = freq * self._num_neurons / len(act_exc_idx)
        return modified_freq

    def eigenvalues():
        return

    def get_neurons_orient_prefs(self):
        return self._orient_prefs
        
    def best_sinusoid(self, freq, phase_grid = 100):
        x = self.get_neurons_orient_prefs()
        amp_opt = 0
        phase_opt = 0
        sinusoid_opt = self.truncate_signal(np.zeros(x.shape[0]))
        output_opt = self.truncate_signal(np.zeros(x.shape[0]))
        for phase in np.linspace(0, 2*np.pi, 100, endpoint=False):
            sinusoid = sinSignal(x, 180, freq, phase=phase)
            truncated_sinusoid = self.truncate_signal(sinusoid)
            output = self.truncated_signal_pass(truncated_sinusoid)
            amp = np.linalg.norm(output)/np.linalg.norm(truncated_sinusoid)
            if amp > amp_opt:
                sinusoid_opt = truncated_sinusoid
                phase_opt = phase
                amp_opt = amp
                output_opt = output
        return amp_opt, sinusoid_opt, phase_opt, output_opt

    # truncated-sinusoids pass the system
    def best_sinusoids(self, freq_range=range(0, 10)):
        """
        return a dataframe with columns 'sin', 'amp', 'freq' and 'output'
        """
        sinusoids = []
        amps = []
        freqs = []
        outputs = []
        for freq in freq_range:
            amp_opt, sinusoid_opt, phase_opt, output_opt = self.best_sinusoid(freq)
            sinusoid_opt = self.restore_vector(sinusoid_opt)
            output_opt = self.restore_vector(output_opt)
            sinusoids.append(sinusoid_opt)
            amps.append(amp_opt)
            freqs.append(freq)
            outputs.append(output_opt)
        data = {'sin': sinusoids, 'amp': amps, 'freq': freqs, 'output': outputs}
        best_sinusoids = pd.DataFrame(data)
        return best_sinusoids
    
