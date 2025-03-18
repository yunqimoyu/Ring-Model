import numpy as np

### 1d, single signal
def gaussianSignal(x, mu, sigma, amplitude=1):
    # print("Generate a discritized Gaussian signal.")
    return amplitude * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

def sinSignal(t, period, freq=2, amp=1, phase=0):
    # print("Generate a discritized sinosoidal signal.")
    x = amp * np.sin(2 * np.pi * freq * t / period + phase)
    return x

def cosSignal(t, period, freq=2, amp=1, phase=0):
    # print("Generate a discritized sinosoidal signal.")
    x = amp * np.cos(2 * np.pi * freq * t / period + phase)
    return x

def constSignal(x, value):
    print("Generate a discritized constant signal.")
    return np.ones(x.shape[0]) * value

def image2Signal(x, img, ISTrans):
    signal = ISTrans.image2signal(img, x)
    return signal

signalGenerator = {'gaussian': gaussianSignal, 'sin': sinSignal, 'cos': cosSignal, 'const': constSignal,
                   'from_img': image2Signal}

### 1d, multiple signals


### 2d, single signals
def gabor2D(img_grid, amplitude, var, theta, phase, frequency, normalize = False):
    x, y = img_grid
    theta = np.deg2rad(theta)
    phase = np.deg2rad(phase)

    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)
    sin_part = np.cos(2 * np.pi * frequency * x_theta + phase)
    gaussian_part = gabor = amplitude * np.exp(-0.5 * ((x_theta ** 2) / var ** 2 + (y_theta ** 2) / var ** 2)) 
    gabor = gaussian_part * sin_part
    if normalize:
        shape = img_grid[0].shape
        const_img = np.ones(shape)
        P1 = np.sum(gabor*const_img)
        gabor = gabor - P1/(shape[0]*shape[1])
        # P2 = np.sum(gabor*gaussian_part)
        # print(P1, P2)
        # c = P1/P2
        # sin_part_1 = sin_part - c
        # gabor = gaussian_part * sin_part_1
    return gabor

# def func2Signal_2D(img_grid, func):
#     """
#     It seems that this function need a for loop. 
#     """
#     x, y = img_grid
#     return


### 2d, multiple signals
