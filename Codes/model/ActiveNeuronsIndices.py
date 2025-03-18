# File: ./model/active_neuron_index.py

import numpy as np
from utilities.VecOps import NumberListOps

class ActiveNeuronsIndices:
    def __init__(self, indices=[], angle_range=[]):
        """Initialize with a list of indices."""
        self._indices = indices

    @classmethod
    def from_indices(cls, indices):
        """Initialize directly from provided indices."""
        return cls(indices)

    @classmethod
    def from_angle_range(cls, orientation_prefs, angle_range):
        """Initialize based on an angle range and neuron preferred angles."""
        indices = [i for i, angle in enumerate(orientation_prefs) 
                   if angle_range[0] <= angle < angle_range[1]]
        active_neurons_indices = cls(indices)
        active_neurons_indices.orientation_prefs = orientation_prefs[indices]
        return active_neurons_indices

    @classmethod
    def from_activity_array(cls, activity_array):
        """Initialize from an activity array (positive activity indicates active neurons)."""
        indices = [i for i, activity in enumerate(activity_array) if activity > 0]
        return cls(indices)

    @staticmethod
    def create(indices=None, orientation_prefs=None, angle_range=None, activity_array=None):
        if indices:
            return ActiveNeuronsIndices.from_indices(indices)
        # elif activity_array.any():
        #     return ActiveNeuronsIndices.from_activity_array(activity_array)
        # else:
        elif angle_range and orientation_prefs.any():
            return ActiveNeuronsIndices.from_angle_range(orientation_prefs, angle_range)
        # else:
        #     raise ValueError('''No information is provided!''')
    
    def keep_odd(self):
        self._indices = NumberListOps.keep_odd(self._indices)
        return

    def keep_even(self):
        self._indices = NumberListOps.keep_even(self._indices)
        return
        
    def randomly_keep(self, throw_probability=0.1):
        self._indices = NumberListOps.randomly_keep(self._indices, throw_probablity)
        return


    def special_keep(self, special_keep, throw_probability=0.1):
        if special_keep == 'keep_even':
            self.keep_even()
        elif special_keep == 'keep_odd':
            self.keep_odd()
        elif special_keep == 'random_keep':
            self.randomly_keep(throw_probability)
        return

    def get_indices(self):
        return self._indices
    
    def get_active_orients(self):
        if hasattr(self, 'orientation_prefs'):
            return self.orientation_prefs

    def get_active_num(self):
        return len(self._indices)

    def get_matrix_form(self, num_neurons):
        diagonal_vector = np.zeros(num_neurons)
        diagonal_vector[self._indices] = 1
        return np.diag(diagonal_vector)
