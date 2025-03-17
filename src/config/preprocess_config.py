import numpy as np

class PreprocessConfig:

    # Image Adjustment
    BRIGHTNESS = -70
    CONTRAST = 1.8

    # Skewing Adjustments
    ANGLE_DELTA = 1
    ANGLE_LIMIT = 70

    # Image Rescaling Parameters
    IMG_WIDTH = 800

    # Contour Detection
    BOUND_KERNEL = (5, 5)
    DILATE_ITER = 2
    ERODE_ITER = 9

    MIN_COUNTOUR_FACTOR = 0.5
    MAX_COUNTOUR_FACTOR = 0.9
    VALID_RATIO = 0.5

    LARGEST_CONTOUR_THRESHOLD = 0.5

    MIN_WIDTH_RATIO = 0.001
    MAX_WIDTH_RATIO = 0.92

    # Color Range
    MIN_RANGE = -15
    MAX_RANGE = 10

    # Text Detection
    LOWER_RANGE = 0
    UPPER_RANGE = 255
    SHADES = 10

    # Gaussian Blur
    KERNEL_DIMS = 3
    GAUSSIAN_SIGMA = 0

    # NEED to change this variable names for highlightText
    # Horizontal projection for text lines
    KERNEL_RATIO = (3, 6)
    # DILATE_ITER = 4
    ASPECT_RATIO = 1.1
    

preprocess_config = PreprocessConfig()