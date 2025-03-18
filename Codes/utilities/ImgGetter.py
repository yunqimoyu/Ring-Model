import numpy as np

from utilities.FuncOps import counterclockwise_rotate_2d_func, discretize_2d_func
from utilities.FuncGenerators import gabor_generator
from config.rotational_filter_paras import *

def pick_natural_image(idx=5):
    data = np.load('/root/yanjing/data/projects/Ring_Model/Robustness_RingModel_v2/Data/Interim/natural_images/testCutImages.npy')
    img = data[idx]
    resized_img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_NEAREST)
    return resized_img

def get_GF_singular_vector(idx=4):
    data = np.load('/root/yanjing/data/projects/Ring_Model/Robustness_RingModel_v2/Data/GF/gabor_filters_part_active/64*64/v_optimal_6.npy')
    img = data[idx]
    img = img.reshape((64, 64))
    return img

def random_generated_img():
    img = np.random.rand(64, 64) + np.ones((64, 64))
    return img

def pick_one_gabor(gabor_paras, rad=-np.pi/4):
    gabor = gabor_generator(**gabor_para)
    rotated_gabor = counterclockwise_rotate_2d_func(gabor, rad=rad)
    img = discretize_2d_func(rotated_gabor, **discretize_paras)
    return img