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

    # Color Range
    MIN_RANGE = -15
    MAX_RANGE = 10

    # Text Detection
    SHADES = 8
    LOWER_RANGE = 0
    UPPER_RANGE = 255

    # Gaussian Blur
    KERNEL_DIMS = 3
    GAUSSIAN_SIGMA = 0

    # NEED to change this variable names for highlightText
    # Horizontal projection for text lines
    KERNEL_RATIO = (3, 5)
    HIGH_DILATE_ITER = 1
    MAX_AR = 10
    MIN_AREA = 4
    MIN_HEIGHT = 10
    

preprocess_config = PreprocessConfig()