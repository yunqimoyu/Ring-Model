import random
import numpy as np

class VectorTrans:
    @staticmethod
    def itrans_kernel(kernel):
        """
        Example:
            Input: [2, 1, 0, 0, 1]
            Output: [0, 1, 2, 1, 0]
            Input: [2, 1, 0, 1]
            Output: [1, 2, 1, 0]
        """
        size = kernel.shape[0]
        itrans_kernel = np.concatenate((kernel[size//2 + 1:], kernel[:size//2  + 1]))
        return itrans_kernel

    @staticmethod
    def trans_kernel(kernel):
        """
        Example:
            Input: [0, 1, 2, 1, 0]
            Output: [2, 1, 0, 0, 1]
            Input: [1, 2, 1, 0]
            Output: [2, 1, 0, 1]
        """
        size = kernel.shape[0]
        trans_kernel = np.concatenate((kernel[(size-1)//2:], kernel[:(size-1)//2]))
        return trans_kernel

    @staticmethod
    def circulant_matrix_from_vector(vector):
        """
        Example:
            Input: [0, 1, 2]
            Output: [[0, 1, 2],
                    [2, 0, 1],
                    [1, 2, 0]]
        """
        n = len(vector)
        H = np.zeros((n, n))
        for i in range(n):
            H[i, :] = np.roll(vector, i)
        return H
   
    @staticmethod 
    def unique_averaged(a, b):
        # Get unique elements in `b` and the indices of the first occurrence
        unique_b, indices = np.unique(b, return_inverse=True)

        # Initialize a list to store the averaged values for unique elements in `b`
        averaged_a = []

        # Iterate over the unique elements in `b`
        for unique in unique_b:
            # Find the indices of all occurrences of `unique` in `b`
            occurrence_indices = np.where(b == unique)[0]

            # Compute the average of the corresponding elements in `a`
            average_value = np.mean(a[occurrence_indices])

            # Append the average value to `averaged_a`
            averaged_a.append(average_value)

        # Convert `averaged_a` to a numpy array
        averaged_a = np.array(averaged_a)

        return averaged_a, unique_b

    @staticmethod
    def filter_by_threshold(a, b, threshold):
        # Find indices where elements in `b` are less than the given threshold
        indices = np.where(b <= threshold)[0]

        # Use these indices to create filtered arrays
        filtered_a = a[indices]
        filtered_b = b[indices]

        return filtered_a, filtered_b


    @staticmethod
    def normalize_columns(matrix):
        """
        Normalize the columns of the given matrix such that each column has a 2-norm of 1.

        Parameters:
            matrix (numpy.ndarray): The input 2D numpy array where each column is a vector.

        Returns:
            numpy.ndarray: A new matrix with the same shape, with each column normalized.
        """
        # Calculate the 2-norm of each column
        column_norms = np.linalg.norm(matrix, axis=0)
        
        # Avoid division by zero by setting any zero norms to 1 temporarily
        column_norms[column_norms == 0] = 1
        
        # Normalize each column by its 2-norm
        normalized_matrix = matrix / column_norms
        
        return normalized_matrix

    def vectors_freq(vectors):
        fft_results = np.fft.fft(vectors, axis=0)
        return fft_results

    @staticmethod
    def vectors_order_by_freq(vectors):
        dom_freq = VectorsProperty.dominant_frequencies(vectors)
        order = np.argsort(dom_freq)
        return dom_freq, order

    @staticmethod
    def col_vectors_padding(vectors, padding_num):
        if padding_num < 1:
            return vectors
        num_rows, num_columns = vectors.shape
        padding_rows = np.zeros((padding_num, num_columns))
        return np.vstack((vectors, padding_rows))

    @staticmethod
    def index_to_frequency(dominant_index, signal_length, sampling_rate):
        """
        Parameters:
        - dominant_index: int or a list of int
        - signal_length: int, the length of the signal data.
        - sampling_rate: int or float, the rate at which the signal was sampled.

        Returns:
        - float, the dominant frequency in the signal in Hz.
        """
        frequency = dominant_index * sampling_rate / signal_length
        return frequency

    @staticmethod
    def take_part(v, x_range_0, x_range_1):
        """
        Parameters:
        - v: The original vector
        - x_range_0: Original x range interval, represented by a list containing two elements.
        - x_range_1: The new x range interval, represented by a list containing two elements.
        
        Returns
        - a new vector, which is a sub-vector(?) of the original one

        Example
        take_part([0, 1, 2, 3], [0, 1], [0, 1/2]) returns [0, 1]

        (close and end properties may be added, currently closed by default)
        """
        x0_start, x0_end = x_range_0
        x1_start, x1_end = x_range_1

        N = v.shape[0]
        T = x0_end - x0_start

        d = T/N

        start_index = int((x1_start - x0_start) // d)
        end_index = int((x1_end - x0_start) // d + 1)

        new_vector = v[start_index:end_index]

        return new_vector

class NumberListOps:
    @staticmethod
    def keep_odd(numbers):
        odd_numbers = [num for num in numbers if num % 2 != 0]
        return odd_numbers
    
    @staticmethod
    def keep_even(numbers):
        even_numbers = [num for num in numbers if num % 2 == 0]
        return even_numbers

    @staticmethod
    def randomly_keep(numbers_list, throw_probability=0.5):
        return [num for num in numbers_list if random.random() > throw_probability]


class MatTrans:
    @staticmethod
    def take_part(A, x_range_0, x_range_1, y_range_0, y_range_1):
        x0_start, x0_end = x_range_0
        x1_start, x1_end = x_range_1

        y0_start, y0_end = y_range_0
        y1_start, y1_end = y_range_1

        Nx = A.shape[1]
        Tx = x0_end - x0_start
        dx = Tx/Nx

        Ny = A.shape[0]
        Ty = y0_end - y0_start
        dy = Ty/Ny

        x_start_index = int((x1_start - x0_start) // dx)
        x_end_index = int((x1_end - x0_start) // dx + 1)

        y_start_index = int((y1_start - y0_start) // dy)
        y_end_index = int((y1_end - y0_start) // dy + 1)
        print(y_start_index)

        new_vector = A[y_start_index:y_end_index, x_start_index:x_end_index]
        return new_vector

class NumTrans:
    @staticmethod
    def max_rise(max, percent = 0.1):
        if max > 0:
            return max * (1 + percent)
        else:
            return max * (1 - percent)

    @staticmethod
    def min_decrease(min, percent = 0.1):
        if min > 0:
            return min * (1 - percent)
        else:
            return min * (1 + percent)

class VectorProperty:
    @staticmethod
    def dominant_frequency(fft_result):
        """
        Calculate the dominant frequency of a real-valued signal, considering only
        up to the Nyquist frequency for real signals.
        
        Parameters:
        - fft_result
        
        Returns:
        - int, the domiant index.
        """
        n = len(fft_result)
        half_n = n // 2

        magnitudes = np.abs(fft_result[:half_n])
        dominant_index = np.argmax(magnitudes)
        return dominant_index


class VectorsProperty:
    @staticmethod
    def dominant_frequencies(fft_results):
        """
        Calculate the dominant frequencies for real-valued vectors provided 
        as columns in a matrix.

        Parameters:
        - fft_results

        """
        n = fft_results.shape[0]
        half_n = n // 2

        magnitudes = np.abs(fft_results[:half_n, :])
        dominant_indices = np.argmax(magnitudes, axis=0)
        return dominant_indices
        
