import numpy as np

class PreprocessConfig:

    # Image Adjustment
    BRIGHTNESS = 0.0
    CONTRAST = 1

    # Image Rescaling Parameters
    ZOOM_FACTOR = 3.0
    RESIZED_FACTOR = 1/3.0
    MAX_WIDTH = 1000


    # Shading
    SHADES = 5
    

    # Gaussian Blur
    KERNEL_DIMS = 3
    GAUSSIAN_SIGMA = 0


    # Masking
    LOWER_MASK = np.array([0, 0, 100]) # TODO - This should dynamically change based on the histogram of the image
    UPPER_MASK = np.array([0, 0, 255]) # TODO - The second most common grouping of color is most likely text and not the background
    

    # Horizontal projection for text lines
    KERNEL_RATIO = (5, 3) # TODO - This is probably ideal for text to be detected show try not to edit this if possible
    DILATE_ITER = 4
    ASPECT_RATIO = 7.0
    

preprocess_config = PreprocessConfig()