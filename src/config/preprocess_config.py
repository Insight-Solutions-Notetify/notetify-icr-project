import numpy as np

class PreprocessConfig:

    # Image Adjustment
    BRIGHTNESS = -10
    CONTRAST = 1.2

    # Skewing Adjustments
    ANGLE_DELTA = 4
    ANGLE_LIMIT = 45

    # Image Rescaling Parameters
    IMG_WIDTH = 800

    # Color Range
    MIN_RANGE = -15
    MAX_RANGE = 10

    # Text Detection
    LOWER_RANGE = 0
    UPPER_RANGE = 255

    # Shading
    SHADES = 8

    # Gaussian Blur
    KERNEL_DIMS = 3
    GAUSSIAN_SIGMA = 0

    # Horizontal projection for text lines
    KERNEL_RATIO = (3, 6)
    DILATE_ITER = 4
    ASPECT_RATIO = 1.1
    

preprocess_config = PreprocessConfig()