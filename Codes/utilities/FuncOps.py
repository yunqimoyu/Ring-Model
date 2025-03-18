import numpy as np

def counterclockwise_rotate_2d_func(func, rad):
    def rotated_func(x, y):
        xp = x * np.cos(rad) + y * np.sin(rad)
        yp = -x * np.sin(rad) + y * np.cos(rad)
        return func(xp, yp)
    return rotated_func

def translate_2d_func(func, trans_coord):
    def translated_func(x, y):
        xp = x - trans_coord[0]
        yp = y - trans_coord[1]
        return func(xp, yp)
    return translated_func

def translate_1d_func(func, trans_coord):
    def translated_func(x):
        xp = x - trans_coord
        return func(xp)
    return translated_func

def discretize_2d_func(func, x_range, y_range, resolution):
    """
    Discretize a 2D function over a specified domain.

    Parameters:
    - func: Callable[[float, float], float]
        The 2D function to discretize. It should take two float arguments (x, y) and return a float.
    - x_range: tuple[float, float]
        The range of x values over which to evaluate the function (start, end).
    - y_range: tuple[float, float]
        The range of y values over which to evaluate the function (start, end).
    - resolution: tuple[int, int]
        The number of points to sample along each dimension (x_resolution, y_resolution).

    Returns:
    - np.ndarray
        A 2D NumPy array representing the discretized function.
    """
    x = np.linspace(x_range[0], x_range[1], resolution[0])
    y = np.linspace(y_range[0], y_range[1], resolution[1])
    xx, yy = np.meshgrid(x, y, indexing='xy')
    values = func(xx, yy)

    return values


def discretize_1d_func(func, x_range, resolution):
    """
    Discretize a 1D function over a specified domain.
    """
    x = np.linspace(x_range[0], x_range[1], resolution)
    values = func(x)
    return values

def circle_function_from_2d(func, radius):
    """
    Creates a 1D function from a given 2D function `func` sampled along a circle of radius `radius`.

    Parameters:
    - func: A 2D function to sample, which takes two arguments (x, y) and returns a single value.
    - radius: The radius of the circle along which to sample `func`.

    Returns:
    - A 1D function that takes an angle theta (in radians, from 0 to 2*pi) and returns the value of `func`
      at the corresponding point on the circle.
    """
    def circle_function(theta):
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        return func(x, y)

    return circle_function


