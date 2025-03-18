import sys
import os

current_script_path = os.path.dirname(os.path.realpath(__file__))
module_path = os.path.join(current_script_path, '..')
sys.path.append(os.path.abspath(module_path))

import numpy as np
from utilities.FuncOps import discretize_2d_func, counterclockwise_rotate_2d_func


class rotational_filters:
    def __init__(self, rotation_resolution, 
                 rotation_kernel, 
                 x_range=(-1, 1), y_range=(-1, 1), resolution=(16, 16)):
        self.rotation_resolution = rotation_resolution
        self.rotation_kernel = rotation_kernel
        self.img_resolution = resolution
        self.filter_discrete_paras = {'x_range': x_range, 'y_range': y_range, 'resolution': resolution}
        self.generate_filters()
        return
    
    def generate_filters(self):
        filters = []
        for rad in np.linspace(0, np.pi, self.rotation_resolution, endpoint=False):
            oriented_filter = counterclockwise_rotate_2d_func(func=self.rotation_kernel, rad=rad)
            discretized_oriented_filter = discretize_2d_func(oriented_filter, **self.filter_discrete_paras)
            filters.append(discretized_oriented_filter)
        self._filters = np.array(filters)
        return

    def obtain_filters(self):
        return self._filters

    def obtain_filters_matrix(self):
        filters = self._filters
        filters_flattened = filters.reshape(filters.shape[0], -1)
        return filters_flattened

    def image2signal(self, image):
        result = (self._filters * image[None, :, :]).sum(axis=(1, 2))
        return result

    def images2signals(self, images):
        filters_reshaped = self._filters[np.newaxis, :, :, :]
        images_reshaped = images[:, np.newaxis, :, :]
        result = (filters_reshaped * images_reshaped).sum(axis=(2, 3))
        return result

    def svd_decomp(self):
        matrix = self.obtain_filters_matrix()
        u, s, vt = np.linalg.svd(matrix)
        return u, s, vt

    def signals_and_images_from_svd(self):
        u, s, vt = self.svd_decomp()
        signals = u.T
        images = vt.reshape(vt.shape[0], *self.img_resolution)
        return signals, images


    