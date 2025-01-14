import numpy as np

class PreprocessConfig:

    # Image Rescaling Parameters
    ZOOM_FACTOR = 3.0
    RESIZED_FACOTR = 1/3.0

    # Shading
    shades = 5
    
    # Gaussian Blur
    kernel_dimension = 3
    gaussian_sigma = 0

    # Masking
    lower = np.array([0, 0, 100])
    upper_mask = np.array([157, 40, 255])
    
    # Horizontal projection for text lines
    RATIO = (5, 3)
    DILATE_ITER = 4

    


preprocess_config = PreprocessConfig()