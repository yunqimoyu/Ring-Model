num_neurons=128

# left end is contained and the right is not
act_exc_ranges = {
                  'all': (0, 180),
                  'half': (0, 90),
                  'one third': (0, 60),
                  'one forth': (0, 45),
                  'third forths': (0, 135),
                  'two thirds': (0, 120),
                  'one tenth': (0, 18),
                  'nine tenths': (0, 168)
                  }

act_inh_ranges = {
                  'all': (0, 180),
                  'first half': (0, 90)
                  }

dict_eigenvector_width_ratios = {
                                 'all': [1.4, 1, 1], 
                                 'half': [1, 1, 1],
                                 'one third': [1, 1, 1],
                                 'two thirds': [1, 1, 1],
                                 }
