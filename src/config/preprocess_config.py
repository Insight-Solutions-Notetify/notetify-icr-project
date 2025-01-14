import numpy as np

class PreprocessConfig:

    # Image Rescaling Parameters
    ZOOM_FACTOR = 3.0
    RESIZED_FACOTR = 1/3.0

    # Shading
    SHADES = 5
    
    # Gaussian Blur
    KERNEL_DIMS = 3
    GAUSSIAN_SIGMA = 0

    # Masking
    LOWER_MASK = np.array([0, 0, 100])
    UPPER_MASK = np.array([157, 40, 255])
    
    # Horizontal projection for text lines
    KERNEL_RATIO = (5, 3)
    DILATE_ITER = 4

    


preprocess_config = PreprocessConfig()