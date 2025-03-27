import numpy as np

class PreprocessConfig:

    # Image Adjustment
    BRIGHTNESS = -50
    CONTRAST = 1

    # Skewing Adjustments
    ANGLE_DELTA = 1
    ANGLE_LIMIT = 45 

    # Image Rescaling Parameters
    IMG_WIDTH = 800

    # Contour Detection
    BOUND_KERNEL = (5, 5)
    DILATE_ITER = 2
    ERODE_ITER = 9

    MIN_COUNTOUR_FACTOR = 0.5
    MAX_COUNTOUR_FACTOR = 0.9
    VALID_RATIO = 0.5

    # Color Range
    MIN_RANGE = -10
    MAX_RANGE = 10

    # Dynamic Image Adjustments
    BRIGHTNESS_DELTA = -150
    CONTRAST_DELTA = 2.9

    # Text Detection
    SHADES = 8
    LOWER_RANGE = 0
    UPPER_RANGE = 255

    # Gaussian Blur
    KERNEL_DIMS = 3
    GAUSSIAN_SIGMA = 0

    # NEED to change this variable names for highlightText
    # Horizontal projection for text lines
    HORIZONTAL_KERNEL = (20, 1)
    HORIZONTAL_ITER = 1
    KERNEL_RATIO = (1, 3)
    HIGH_DILATE_ITER = 1
    MAX_AR = 10
    MAX_AREA = 2000.0
    MIN_AREA = 20.0
    MAX_HEIGHT = 100
    

preprocess_config = PreprocessConfig()