import numpy as np

def gaussian_generator(gaussian_var, gaussian_aspect_ratio=1, coordinate_system='cartesian'):
    '''
    generator 2d gaussian function
    '''

    def gaussian_cartesian(x, y):
        return np.exp(-(x**2 + gaussian_aspect_ratio**2 * y**2) / (2 * gaussian_var**2))

    def gaussian_polar(r, theta_):
        x = r * np.cos(theta_)
        y = r * np.sin(theta_)
        return gaussian_cartesian(x, y)

    if coordinate_system == 'cartesian':
        return gaussian_cartesian
    elif coordinate_system == 'polar':
        return gaussian_polar
    else:
        raise ValueError("Invalid coordinate system. Choose 'cartesian' or 'polar'.")

def point_pairs_generator(gaussian_var, gaussian_aspect_ratio=1, distance=1, coordinate_system='cartesian'):
    '''
    generator 2d gaussian function
    '''

    def point_pairs_cartesian(x, y) :
        def gaussian_cartesian(x, y):
            return np.exp(-(x**2 + gaussian_aspect_ratio**2 * y**2) / (2 * gaussian_var**2))
        return gaussian_cartesian(x-distance/2, y) + gaussian_cartesian(x+distance/2, y)

    def point_pairs_polar(r, theta_):
        x = r * np.cos(theta_)
        y = r * np.sin(theta_)
        return point_pairs_cartesian(x, y)

    if coordinate_system == 'cartesian':
        return point_pairs_cartesian
    elif coordinate_system == 'polar':
        return point_pairs_polar
    else:
        raise ValueError("Invalid coordinate system. Choose 'cartesian' or 'polar'.")

def gabor_generator(cos_freq, cos_phase, gaussian_var, elong_fact=1, gaussian_aspect_ratio=1, coordinate_system='cartesian'):

    def gabor_cartesian(x, y):
        return np.exp(-(x**2 + gaussian_aspect_ratio**2 * (y/elong_fact)**2) / (2 * gaussian_var**2)) * np.cos(2 * np.pi * x * cos_freq  + cos_phase)

    def gabor_polar(r, theta_):
        x = r * np.cos(theta_)
        y = r * np.sin(theta_)
        return gabor_cartesian(x, y)

    if coordinate_system == 'cartesian':
        return gabor_cartesian
    elif coordinate_system == 'polar':
        return gabor_polar
    else:
        raise ValueError("Invalid coordinate system. Choose 'cartesian' or 'polar'.")

def scale_invar_gabor_generator(lambda_, theta, psi, sigma, gamma, coordinate_system='cartesian'):
    def scale_invar_gabor_polar(r, theta_):
        '''
        theta_ takes value in the range (-pi, pi)
        '''
        if theta_ > np.pi or theta_ < -np.pi:
            raise ValueError("Invalid angle value for the polar function.")
        return np.exp(-(theta_) / (2 * sigma**2)) * np.cos(2 * theta_ / lambda_ + psi)


    def scale_invar_gabor_cartesian(x, y):
        xp = x * np.cos(theta) + y * np.sin(theta)
        yp = -x * np.sin(theta) + y * np.cos(theta)
        return np.exp(-(xp**2 + gamma**2 * yp**2) / (2 * sigma**2)) * np.cos(2 * np.pi * xp / lambda_ + psi)

    if coordinate_system == 'cartesian':
        return scale_invar_gabor_cartesian
    elif coordinate_system == 'polar':
        return scale_invar_gabor_polar
    else:
        raise ValueError("Invalid coordinate system. Choose 'cartesian' or 'polar'.")