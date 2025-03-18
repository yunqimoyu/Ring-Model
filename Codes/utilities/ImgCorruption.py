import sys
import os

current_script_path= os.path.dirname(os.path.realpath(__file__))
module_path = os.path.join(current_script_path, '..')
sys.path.append(os.path.abspath(module_path))

import cv2
import numpy as np

from utilities.FuncGenerators import point_pairs_generator
from utilities.FuncOps import counterclockwise_rotate_2d_func, discretize_2d_func

discretize_paras = {'x_range': (-1, 1), 'y_range':(-1, 1), 'resolution':(64, 64)} # parameters used to discrete 2d function to matrix

# Dictionary to store functions
corruption_functions = {}
def register_corruption(func):
    corruption_functions[func.__name__] = func
    return func

@register_corruption
def change_contrast(image_array, contrast_factor=1.2):
    if image_array.ndim != 2:
        raise ValueError('Input image should be a 2D array representing a grayscale image.')
    
    # Calculate the mean of the image
    mean_val = np.mean(image_array)
    
    # Adjust the contrast: (pixel - mean) * contrast_factor + mean
    adjusted_image = (image_array - mean_val) * contrast_factor + mean_val
    
    # Clip values to maintain them within [0, 1]
    # adjusted_image = np.clip(adjusted_image, 0, 1)
    
    return adjusted_image

@register_corruption
def resize_image(image_array, rad=3 * np.pi/4, x_scale=1.1, y_scale=1):
    # Get the original image dimensions
    original_height, original_width = image_array.shape[:2]
    
    # Calculate the center of the image
    center = ((original_width - 1) / 2, (original_height - 1) / 2)
    
    # Rotate the image using the radian angle converted to degrees
    angle_deg = np.degrees(rad)
    rotation_matrix = cv2.getRotationMatrix2D(center, -angle_deg, 1)
    rotated_image = cv2.warpAffine(image_array, rotation_matrix, (original_width, original_height))
    
    # Scale the rotated image
    resized_rotated_image = cv2.resize(rotated_image, None, fx=x_scale, fy=y_scale, interpolation=cv2.INTER_LINEAR)
    
    # Calculate the new dimensions after scaling
    scaled_height, scaled_width = resized_rotated_image.shape[:2]
    
    # Get the new center
    new_center = (scaled_width / 2, scaled_height / 2)
    
    # Rotate back the image to the original orientation
    revert_rotation_matrix = cv2.getRotationMatrix2D(new_center, angle_deg, 1)
    transformed_image = cv2.warpAffine(resized_rotated_image, revert_rotation_matrix, (scaled_width, scaled_height))
    
    # Crop or pad to original dimensions
    def crop_or_pad(image, target_height, target_width):
        current_height, current_width = image.shape[:2]
        
        # Determine cropping/padding requirements
        top_padding = (target_height - current_height) // 2
        bottom_padding = target_height - current_height - top_padding
        left_padding = (target_width - current_width) // 2
        right_padding = target_width - current_width - left_padding
        
        # Crop or pad
        cropped_padded_image = cv2.copyMakeBorder(
            image,
            max(0, top_padding), max(0, bottom_padding),
            max(0, left_padding), max(0, right_padding),
            cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )

        # Crop if needed
        start_y = max(0, -top_padding)
        start_x = max(0, -left_padding)
        cropped_padded_image = cropped_padded_image[start_y:start_y + target_height, start_x:start_x + target_width]
        
        return cropped_padded_image

    final_image = crop_or_pad(transformed_image, original_height, original_width)
    
    return final_image

@register_corruption
def add_ones(img):
    img = img + 0.1 * np.ones((64, 64))
    return img

@register_corruption
def add_point_pairs(img, distance = 0.5, rotate_rad=0, amp=1):
    point_pairs_gen = counterclockwise_rotate_2d_func(func=point_pairs_generator(gaussian_var=0.1, distance=distance), rad=rotate_rad)
    rotated_point_pairs = discretize_2d_func(point_pairs_gen, **discretize_paras)
    perturbed_img = img + rotated_point_pairs * amp
    return perturbed_img

@register_corruption
def apply_gaussian_noise(image):
    mean = 0
    stddev = 25
    gauss = np.random.normal(mean, stddev, image.shape)
    noisy = cv2.add(image, gauss)
    return noisy

# @register_corruption
# def apply_shot_noise(image):
#     shot_noise = np.random.poisson(image / 255.0 * 30) / 30 * 255
#     noisy_image = np.clip(image + shot_noise, 0, 255)
#     return noisy_image

@register_corruption
def apply_impulse_noise(image):
    prob = 0.01
    noise = np.random.choice((0, 255), image.shape, p=[1-prob, prob])
    noisy_image = np.where(noise == 255, noise, image)
    return noisy_image

@register_corruption
def apply_defocus_blur(image):
    return cv2.GaussianBlur(image, (15, 15), 0)

@register_corruption
def apply_motion_blur(image, size=15):
    kernel = np.zeros((size, size))
    kernel[int((size - 1) / 2), :] = np.ones(size)
    kernel = kernel / size
    return cv2.filter2D(image, -1, kernel)

@register_corruption
def apply_zoom_blur(image):
    h, w = image.shape
    zoom_factor = 0.9
    center = np.array((w, h)) / 2
    offset = (1 - zoom_factor) * center
    transformation_matrix = np.array([[zoom_factor, 0, offset[0]], [0, zoom_factor, offset[1]]])
    zoomed_image = cv2.warpAffine(image, transformation_matrix, (w, h), borderMode=cv2.BORDER_REFLECT101)
    return cv2.addWeighted(image, 0.5, zoomed_image, 0.5, 0)

@register_corruption
def apply_snow(image):
    snow = np.random.normal(130, 30, image.shape).astype('uint8')
    snowy_image = cv2.add(image, snow)
    return snowy_image

@register_corruption
def apply_frost(image):
    frost_mask = np.full(image.shape, 200, dtype='uint8')
    faked_frost_image = cv2.addWeighted(image, 0.7, frost_mask, 0.3, 0)
    return faked_frost_image

@register_corruption
def apply_fog(image):
    fog_layer = np.full(image.shape, 220, dtype='uint8')
    fogged_image = cv2.addWeighted(image, 0.7, fog_layer, 0.3, 0)
    return fogged_image

@register_corruption
def adjust_brightness(image, factor=1.5):
    return np.clip(image * factor, 0, 255).astype('uint8')

@register_corruption
def adjust_contrast(image, factor=1.5):
    mean = np.mean(image)
    return np.clip((image - mean) * factor + mean, 0, 255).astype('uint8')

# @register_corruption
# def apply_pixelate(image, pixel_size=16):
#     h, w = image.shape
#     image = cv2.resize(image, (pixel_size, pixel_size), interpolation=cv2.INTER_LINEAR)
#     pixelated_image = cv2.resize(image, (w, h), interpolation=cv2.INTER_NEAREST)
#     return pixelated_image

@register_corruption
def apply_jpeg_compression(image, quality=25):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', image, encode_param)
    jpeg_image = cv2.imdecode(encimg, 0)
    return jpeg_image