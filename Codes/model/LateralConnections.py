import numpy as np
from scipy.fft import fft, ifft

from utilities.SignalGenerator import signalGenerator
from utilities.VecOps import VectorTrans

class LateralCnctKernel:
    def __init__(self, num_neurons, param):
        self.num_neurons = num_neurons
        self.param = param

        self.generate_theta_k()
        self.generate_lateral_connection()
        self.matrix_form()
        self.get_FFT_kernel()
        return

    def generate_theta_k(self):
        '''
        num_neurons is even: 
        num_neurons = 2:
        np.linspace(...) --> 0, 90 
        translate --> 0, 90 
        num_neurons is odd:
        num_neurons = 3:
        np.linspace(...) --> 0, 60, 120 
        translate --> -60, 0, 60
        '''
        self.theta_k = np.linspace(0, 180, self.num_neurons, endpoint=False)
        self.theta_k -= self.theta_k[(self.num_neurons-1)//2]
        return

    def generate_lateral_connection(self):
        kernel = 180*signalGenerator[self.param['type']](self.theta_k, **self.param['para'])/self.num_neurons
        self.kernel = VectorTrans.trans_kernel(kernel)
        return
    
    def matrix_form(self):
        self.matrix_form = VectorTrans.circulant_matrix_from_vector(self.kernel)
        return
    
    def get_FFT_kernel(self):
        self.FFT_kernel = np.fft.fft(self.kernel)
        return

class LateralKernels():
    def __init__(self, num_neurons, lateral_cnct_params, kernels=[]):
        if kernels and lateral_cnct_params:
            raise ValueError("Please provide either kernels or lateral_cnct_params, not both.")
        elif kernels:
            self.EE_kernel, self.EI_kernel, self.IE_kernel = kernels
        elif lateral_cnct_params:                
            self.set_by_params(num_neurons, lateral_cnct_params)
        else:
            raise ValueError("Either kernels or lateral_cnct_params must be provided")
        return
        
    def set_by_params(self, num_neurons, lateral_cnct_params):
        self.EE_kernel = LateralCnctKernel(num_neurons, lateral_cnct_params['kEE'])
        self.EI_kernel = LateralCnctKernel(num_neurons, lateral_cnct_params['kEI'])
        self.IE_kernel = LateralCnctKernel(num_neurons, lateral_cnct_params['kIE'])
        return
    
    def excitatory_connection(self):
        EE_kernel  = VectorTrans.itrans_kernel(self.EE_kernel.kernel)
        return EE_kernel

    def recurrent_inhibitory_connection(self):
        hat_recurrent_inhibitory_connection = self.EI_kernel.FFT_kernel*self.IE_kernel.FFT_kernel
        recurrent_inhibitory_connection = np.fft.ifft(hat_recurrent_inhibitory_connection)
        recurrent_inhibitory_connection = VectorTrans.itrans_kernel(recurrent_inhibitory_connection)
        return recurrent_inhibitory_connection
    
    def hat_excitatory_connection(self):
        FFT_kernel = self.EE_kernel.FFT_kernel
        FFT_kernel = VectorTrans.itrans_kernel(FFT_kernel)
        return FFT_kernel

    def hat_recurrent_inhibitory_connection(self):
        hat_recurrent_inhibitory_connection = self.EI_kernel.FFT_kernel*self.IE_kernel.FFT_kernel
        hat_recurrent_inhibitory_connection = VectorTrans.itrans_kernel(hat_recurrent_inhibitory_connection)
        return hat_recurrent_inhibitory_connection

    def get_hat_h_inv_with_EE_only(self):
        # self.h = self.EE_kernel.FFT_kernel
        if np.max(self.EE_kernel.FFT_kernel) >=1:
            print("WARNING: EE connection seems to be too strong!!!!!")
        hat_h_inv = 1/(1 - self.EE_kernel.FFT_kernel)
        return hat_h_inv

    def get_hat_h_inv_with_EIIE_only(self):
        # self.h = self.EE_kernel.FFT_kernel
        hat_h_inv = 1/(1 + self.EI_kernel.FFT_kernel*self.IE_kernel.FFT_kernel)
        return hat_h_inv
    
    def get_hat_h_inv(self):
        # self.h = self.EE_kernel.FFT_kernel
        if np.max(self.EE_kernel.FFT_kernel) >=1:
            print("WARNING: EE connection seems to be too strong!!!!!")
        self.hat_h_inv = 1/(1 - self.EE_kernel.FFT_kernel + self.EI_kernel.FFT_kernel*self.IE_kernel.FFT_kernel)
        return self.hat_h_inv