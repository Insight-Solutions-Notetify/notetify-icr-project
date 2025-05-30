# Import the necessary libraries
import cv2
from cv2.typing import MatLike
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from typing import Set
from collections import Counter
from scipy.ndimage import rotate
from scipy.interpolate import interp1d
import subprocess
import re

# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
# sys.path.insert(0, project_root)
# os.chdir(project_root)

from .config.preprocess_config import preprocess_config
from .logger import logger, log_execution_time

### PREPROCESS MODULE ###

# Preprocess Module will process type .jpg images to prepare for segmentation
# Requirements:
# - Input is the MatLike surface since the main driver already retrieved image and convert it to cv2 ready format
# - non-text regions are excluded from the image
# - binary result of image either being white (1) or black(0)
# - Resizing/ De-blurring for varying image inputs and quality.
# - Range of desired shades of handwriting to be included
# - Mask on desired range

# Should only use BGRToShades for finding the color range
def BGRToShades(input: MatLike, shades = preprocess_config.SHADES ) -> MatLike:
    ''' Convert BGR to specialized Shades of GRAY'''
    try:
        gray_image = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
        gray_image = gray_image.astype(np.float32)/255

        logger.debug(f"Shading by {shades} colors")
        result = 255 * np.floor(gray_image * shades + 0.5) / shades
        result = result.clip(0 ,255).astype(np.uint8)
    except cv2.error as e:
        logger.error(f"Error: {e}")
        return input
    
    return result

def BGRToGRAY(input: MatLike) -> MatLike:
    ''' Convert BGR to GRAY '''
    try:
        gray = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
    except cv2.error as e:
        logger.error(f"Error: {e}")
        return input
    
    return gray
    
def GRAYToBGR(input: MatLike) -> MatLike:
    ''' Convert GRAY to BGR '''
    try:
        reverted = cv2.cvtColor(input, cv2.COLOR_GRAY2BGR)
    except cv2.error as e:
        logger.error(f"Error: {e}")
        return input
    
    return reverted

def BGRToHSV(input: MatLike) -> MatLike:
    ''' Convert BGR to HSV '''
    try:
        hsv = cv2.cvtColor(input, cv2.COLOR_BGR2HSV)
    except cv2.error as e:
        logger.error(f"Error: {e}")
        return input
    
    return hsv

def BGRToRGB(input: MatLike) -> MatLike:
    ''' Convert BGR to RGB '''
    try:
        rgb = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
    except cv2.error as e:
        logger.error(f"Error: {e}")
        return input
    
    return rgb

def flipImage(input: MatLike) -> MatLike:
    ''' Inverse of inputted image '''
    try:
        flipped = cv2.bitwise_not(input)
    except cv2.error as e:
        logger.error(f"Error: {e}")
        return input
    
    return flipped

def getThreshold(input: MatLike) -> MatLike:
    ''' Get the threshold of the image '''
    try:
        threshold = cv2.threshold(input, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    except cv2.error as e:
        logger.error(f"Error: {e}")
        return input
    
    return threshold

def contrastImage(input: MatLike, bg_range=None, contrast=preprocess_config.CONTRAST, brightness=preprocess_config.BRIGHTNESS):
    ''' Apply a contrast and brightness adjustment to the image '''
    if bg_range is not None:
        logger.debug(f"Adjusting image to background range: {bg_range}")
        #adjust image manually
        lower, upper = bg_range #lower gets assigned, but upper never gets assigned, might be how we change the images
        m = interp1d([0, 255], [0, 1])#change to max or min
        value = m(lower)#change to lower_value
        value_upper = m(upper)
        brightness_offset = value * preprocess_config.BRIGHTNESS_DELTA
        contrast_factor = 1 + (value * preprocess_config.CONTRAST_DELTA)
        factor = 0.2
        if value_upper > factor :
            scale_factor = 1.0 - (value_upper - factor)  # e.g. 0.8 => scale_factor=1, 1.0 => scale_factor=0.8
            scale_factor = max(scale_factor, 0.2)   # clamp so it doesn't go below 0.5
            brightness_offset *= scale_factor
            contrast_factor *= scale_factor

            logger.debug(f"Interpolating lower value: {lower} => {value}  upper value: {upper} => {value_upper}")
            logger.debug(f"Applying contrast factor: {contrast_factor}, brightness offset: {brightness_offset}")
            adjusted_image = cv2.addWeighted(
                input, 
                contrast_factor, 
                np.zeros(input.shape, input.dtype), 
                0, 
                brightness_offset
                #adds a cap to the brightness for the contrast
                
        )

        logger.warning(f"Interploating lower value: {lower} to range (-1 to 1) New Value: {value}")# ask him what this does
        logger.debug(f"Applying a contrast value: {1 + (value * preprocess_config.CONTRAST_DELTA)}, brightness value: {value * preprocess_config.BRIGHTNESS_DELTA}")
        adjusted_image = cv2.addWeighted(input, 1 + (value * preprocess_config.CONTRAST_DELTA), np.zeros(input.shape, input.dtype), 0, value * preprocess_config.BRIGHTNESS_DELTA)
    else:
        logger.debug(f"Applying a contrast value: {contrast}, brightness value: {brightness}")
        adjusted_image = cv2.addWeighted(input, contrast, np.zeros(input.shape, input.dtype), 0, brightness)
    
    logger.debug("Contrast Complete\n")
    return adjusted_image

def blurImage(input: MatLike, sigma=preprocess_config.GAUSSIAN_SIGMA) -> MatLike:
    logger.debug(f"Applying blur with strength {sigma}")
    gaussian = cv2.GaussianBlur(input, (preprocess_config.KERNEL_DIMS, preprocess_config.KERNEL_DIMS), sigma)
    return gaussian

def rescaleImage(input: MatLike) -> MatLike:
    ''' Rescale images down to a standard acceptable for input '''
    logger.debug(f"Minimzing image to a maximum size of {preprocess_config.IMG_WIDTH}")

    img_height, img_width, _  = input.shape
    # logger.warning(f"Image shape {input.shape}")
    IMG_RATIO = img_width / img_height
    min_width = min(img_width, preprocess_config.IMG_WIDTH)
    result = cv2.resize(input, (min_width, int(min_width / IMG_RATIO))) #, fx=IMG_RATIO, fy=IMG_RATIO)

    logger.debug(f"Original Shape: {input.shape}, Resized Shape: {result.shape}\n")  
    return result

@log_execution_time
def correctSkew(input: MatLike, delta=preprocess_config.ANGLE_DELTA, limit=preprocess_config.ANGLE_LIMIT) -> MatLike:
    ''' Correct skew of image'''
    logger.debug("Correcting skew of image")
    def determine_score(arr, angle):
        data = rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1, dtype=float)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
        return histogram, score

    # gray = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
    flipped = flipImage(input) # Image already gray in NEW method
    thresh = getThreshold(flipped)

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(thresh, angle)
        # logger.debug(f"Angle: {angle}, Score: {score}")
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = input.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    corrected = cv2.warpAffine(flipped, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    logger.debug(f"Best Angle: {best_angle}")
    logger.debug(f"Complete correctSkew()\n")
    return flipImage(corrected)

@log_execution_time
def highlightBoundary(input: MatLike,
                      kernel_shape=preprocess_config.BOUND_KERNEL,
                      dilated_iter=preprocess_config.DILATE_ITER,
                      eroded_iter=preprocess_config.ERODE_ITER,
                      min_fac=preprocess_config.MIN_COUNTOUR_FACTOR,
                      max_fac=preprocess_config.MAX_COUNTOUR_FACTOR,
                      valid_ratio=preprocess_config.VALID_RATIO) -> MatLike:
    ''' Removes any background apart from the medium where the text is located '''
    logger.debug("Highlighting the boundary of the note image")

    # gray = BGRToGRAY(input)
    thresh = getThreshold(input)

    # New method overwriting the previous one
    edges = cv2.Canny(thresh, 50, 150)
    cnts = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Apply morphological operations to clean up image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_shape)
    dilated = cv2.dilate(thresh, kernel, iterations=dilated_iter)
    eroded = cv2.erode(dilated, kernel, iterations=eroded_iter)

    # Find contours of the eroded image
    cnts = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    if len(cnts) == 0:
        logger.warning("No contours could be generated")
        return input
    
    # Draw contours to verify that the correct region is being selected
    # cv2.drawContours(reversed, cnts, -1, (0, 255, 0), 2)
    # return reversed

    min_area = min_fac * input.shape[0] * input.shape[1]
    max_area = max_fac * input.shape[0] * input.shape[1]
    logger.debug(f"min_area: {min_area}, max_area: {max_area}, valid_ratio: {valid_ratio}")
    valid_counters = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        aspect_ratio = w / float(h)
        
        if area > min_area and area < max_area and valid_ratio < aspect_ratio:
            logger.debug(f"Keeping countor with at {x}, {y}, with an area of {area}")
            valid_counters.append(c)
        else:
            logger.warning(f"Removing contour at {x}, {y}, with area: {area}, and ar: {aspect_ratio}")
        
        
    if valid_counters:
        x_min = min([cv2.boundingRect(c)[0] for c in valid_counters])
        y_min = min([cv2.boundingRect(c)[1] for c in valid_counters])
        x_max = max([cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in valid_counters])
        y_max = max([cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in valid_counters])

    
        # cv2.rectangle(input, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        cropped = input[y_min:y_max, x_min:x_max]
    else:
        cropped = input

    return cropped
    
@log_execution_time
def findColorRange(input: MatLike, k = 2) -> Set:
    ''' Identify the color range for text and background by using k-clustering '''
    logger.debug("Finding text color range")
    # flip = flipImage(input)
    pixels = input.reshape(-1, 1).astype(np.float32)

    #K-Means Clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 110, 0.2)
    _, labels, center = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Count each cluster's occurence
    counts = Counter(labels.flatten())
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)

    # # Get two dominant colors
    dominant_labels = [i for i, _ in sorted_counts]
    cluster_pixels = {label: [] for label in dominant_labels}

    # Assign pixels to their clusters
    for pixel, label in zip(pixels, labels.flatten()):
        cluster_pixels[label].append(pixel[0])

    # Assign pixels to their clusters
    color_ranges = {}
    for label in dominant_labels:
        cluster_array = np.array(cluster_pixels[label], dtype=np.uint8)
        min_gray = int(np.min(cluster_array))
        max_gray = int(np.max(cluster_array))
        color_ranges[label] = (min_gray, max_gray)

    #Identify background as the most frequent cluster
    background_label = dominant_labels[0]
    text_label = dominant_labels[1]

    bg_range = color_ranges[background_label]
    text_range = color_ranges[text_label]
    # logger.debug(f"Background Color range (BGR): {bg_range}")
    # logger.debug(f"Text Color range (BGR): {text_range}")

    # # Convert this RGB range to GRAY range
    # text_range = [max(0, int(0.114 * text_range[0][0] + 0.587 * text_range[0][1] + 0.299 * text_range[0][2])), 
    #               min(255, int(0.114 * text_range[1][0] + 0.587 * text_range[1][1] + 0.299 * text_range[1][2]))] # Add offset to handle boundary case values
    # bg_range = [max(0, int(0.114 * bg_range[0][0] + 0.587 * bg_range[0][1] + 0.299 * bg_range[0][2])),
    #             min(255, int(0.114 * bg_range[1][0] + 0.587 * bg_range[1][1] + 0.299 * bg_range[1][2]))]

    logger.debug(f"Text Color range (GRAY): {text_range}")
    logger.debug(f"Background Color range (GRAY): {bg_range}")
    return text_range, bg_range

@log_execution_time
def highlightText(input: MatLike, text_range: list,
                  MIN_RANGE=preprocess_config.MIN_RANGE,
                  MAX_RANGE=preprocess_config.MAX_RANGE,
                  LOWER_RANGE=preprocess_config.LOWER_RANGE,
                  UPPER_RANGE=preprocess_config.UPPER_RANGE,
                  LINE_KERN=preprocess_config.HORIZONTAL_KERNEL,
                  LINE_ITER=preprocess_config.HORIZONTAL_ITER,
                  KERN_RATIO=preprocess_config.KERNEL_RATIO,
                  DILATE_ITER=preprocess_config.HIGH_DILATE_ITER,
                  MAX_HEIGHT=preprocess_config.MAX_HEIGHT,
                  MAX_AR=preprocess_config.MAX_AR,
                  MAX_AREA=preprocess_config.MAX_AREA,
                  MIN_AREA=preprocess_config.MIN_AREA,
                  ) -> MatLike:
    ''' Highlights text-only regions, excluding everything else (outputting a binary image of text and non-text) '''
    logger.debug("Highlighting text")
    
    text_range = [max(0, text_range[0] + MIN_RANGE), min(255, text_range[1] + MAX_RANGE)]
    logger.debug(f"Text Range: {text_range}")
    text_region = np.array([[[text_range[0], text_range[0], text_range[0]], 
                              [text_range[1], text_range[1], text_range[1]]]]).astype(np.uint8)
    
    try:
        hsv_text_range = BGRToHSV(text_region)
        # shaded = BGRToShades(input)
        # flip = flipImage(shaded)
        reverted = GRAYToBGR(input)
        hsv = BGRToHSV(reverted)
    except cv2.error as e:
        logger.error(f"Error: {e}")
        return input

    # Ensure range only considers value, not hue/saturation
    hsv_text_range[0][0][0] = LOWER_RANGE
    hsv_text_range[0][0][1] = LOWER_RANGE
    hsv_text_range[0][1][0] = UPPER_RANGE
    hsv_text_range[0][1][1] = UPPER_RANGE

    logger.debug(f"HSV Range: {hsv_text_range[0][0]} - {hsv_text_range[0][1]}")

    original_mask= cv2.inRange(hsv, hsv_text_range[0][0], hsv_text_range[0][1])

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    eroded = cv2.erode(original_mask, kernel, iterations=1)

    # Apply noise reduction before dilation
    mask = cv2.medianBlur(original_mask, 3)
    
    # Detect horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, LINE_KERN)
    detected_lines = cv2.morphologyEx(mask, cv2.MORPH_OPEN, horizontal_kernel, iterations=LINE_ITER)

    # Subtract detected lines from mask
    mask_no_lines = cv2.subtract(mask, detected_lines)

    # Interpolate missing space
    inpainted = cv2.inpaint(mask_no_lines, detected_lines, inpaintRadius=2, flags=cv2.INPAINT_TELEA)

    # Dilate the image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, KERN_RATIO)
    dilate = cv2.dilate(inpainted, kernel, iterations=DILATE_ITER)

    # Detect contours after lines removed
    cnts, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    logger.debug(f"Number of contours: {len(cnts)}")

    # Remove contours that are too small or too elongated (likely lines)
    scr_h = dilate.shape[0]
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        ar = w / float(h)
        area = cv2.contourArea(c)
        
        ''' Contours are valid if they follow these conditions
        - There area is greater than MIN_AREA (area > MIN_AREA)
        - There height is less than MAX_CNT_HEIGHT (h < MAX_CNT_HEIGHT)
        - There aspect ratio is lower than MAX_AR (ar < AR)'''

        if h > scr_h * MAX_HEIGHT or ar > MAX_AR:
            # logger.warning(f"Remvoing contour at ({x}, {y}) with height: {h} and ar: {ar}")
            cv2.drawContours(dilate, [c], -1, (0, 0, 0), -1)
            continue

        if area < MAX_AREA and area > MIN_AREA:
            pass
            # logger.debug(f"Keeping contour at ({x}, {y}) area: {area}")
        else:
            # logger.warning(f"Removing contour over limits at ({x}, {y}) area: {area}")
            cv2.drawContours(dilate, [c], -1, (0, 0, 0), -1) # Possibly smaller contours (smudges)

    logger.debug("Resulting highlighting complete")
    return blurImage(cv2.bitwise_and(dilate, eroded), 0.2)

@log_execution_time
def preprocessImage(input: MatLike) -> MatLike:
    ''' Main process of preprocessing each step is separated into individual functions '''
    logger.debug("Starting preprocess process")
    try:
        # Apply filters to image
        scaled = rescaleImage(input)

        # Simplify from BGR to gray
        grayed = BGRToGRAY(scaled)

        skewed = correctSkew(grayed)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        weighted = clahe.apply(skewed)

        blurred = blurImage(weighted)

        note = highlightBoundary(blurred)

        # Histogram analysis to determine the range of text colors
        text_range, bg_range = findColorRange(note)

        # Exclude everything else except the actual text that make up the note
        result = highlightText(note, text_range)

        # result = result.astype("float32") / 255.0

        # new = result
        # # cv2.imshow("Before boundary cropping", blurred)
        # cv2.imshow("New process", new)
        # cv2.waitKey(0)
        # return scaled
    
        logger.debug("Complete preprocessing process\n")

    except Exception as e:
        logger.error(f"Error: {e}")
        return None
    
    # return scaled
    # return weighted
    # return skewed
    # return blurred
    # return note
    # return bg_adjusted
    return result