import numpy as np
class NeuronOrientationPrefs:
    def __init__(self, num_neurons):
        self.neuron_orientation_prefs = np.linspace(0, 180, num_neurons, endpoint=False) 
        return
