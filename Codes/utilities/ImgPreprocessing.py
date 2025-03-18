from PIL import Image
import numpy as np

def resize_images_from_numpy(image_array, new_size):
    """
    Resizes all images in the input NumPy array to the new size.
    
    Parameters:
    - image_array: np.ndarray, 3D array with shape (number of images, image width, image height).
    - new_size: tuple, new size for the images (width, height).
    
    Returns:
    - resized_array: np.ndarray, 3D array with resized images.
    """
    num_images = image_array.shape[0]
    resized_images = []
    
    for i in range(num_images):
        # Convert each sub-array to a Pillow image
        img = Image.fromarray(image_array[i])
        
        # Resize the image
        img_resized = img.resize(new_size, Image.LANCZOS)
        
        # Convert the resized image back to a NumPy array
        img_resized_array = np.asarray(img_resized)
        
        # Append to the list of resized images
        resized_images.append(img_resized_array)
    
    # Convert the list of resized images back to a 3D NumPy array
    resized_array = np.stack(resized_images, axis=0)
    
    return resized_array
