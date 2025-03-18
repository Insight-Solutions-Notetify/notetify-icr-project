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
import subprocess
import re

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)
os.chdir(project_root)

from src.config.preprocess_config import preprocess_config
from src.modules.logger import logger, log_execution_time

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

def contrastImage(input: MatLike, contrast=preprocess_config.CONTRAST, brightness=preprocess_config.BRIGHTNESS):
    ''' Apply a contrast and brightness adjustment to the image '''
    logger.debug(f"Applying a contrast value: {contrast}, brightness value: {brightness}")
    # # Testing new method of lab conversion
    # lab = cv2.cvtColor(input, cv2.COLOR_BGR2LAB)

    # l, a, b = cv2.split(lab)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # l = clahe.apply(l)

    # lab = cv2.merge((l, a, b))
    # enchanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


    # Old method
    # weighted = cv2.normalize(input, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    adjusted_image = cv2.addWeighted(input, contrast, np.zeros(input.shape, input.dtype), 0, brightness)
    logger.debug("Contrast Complete\n")
    return adjusted_image

def blurImage(input: MatLike, sigma=preprocess_config.GAUSSIAN_SIGMA) -> MatLike:
    logger.debug(f"Applying blur with strength {sigma}")
    gaussian = cv2.GaussianBlur(input, (preprocess_config.KERNEL_DIMS, preprocess_config.KERNEL_DIMS), sigma)
    return gaussian

def rescaleImage(input: MatLike) -> MatLike:
    ''' Rescale images down to a standard acceptable for input '''
    logger.debug(f"Rescaling image to a maximum size of {preprocess_config.IMG_WIDTH}")

    img_width, img_height, _  = input.shape
    IMG_RATIO = img_width / img_height
    result = cv2.resize(input,(preprocess_config.IMG_WIDTH, int(preprocess_config.IMG_WIDTH * IMG_RATIO))) #, fx=IMG_RATIO, fy=IMG_RATIO)

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

    gray = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
    flipped = flipImage(gray)
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
    corrected = cv2.warpAffine(input, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    logger.debug(f"Best Angle: {best_angle}")
    logger.debug(f"Complete correctSkew()\n")
    return corrected


@log_execution_time
def highlightBoundary(input: MatLike,
                      kernel=preprocess_config.BOUND_KERNEL,
                      dilated=preprocess_config.DILATE_ITER,
                      eroded=preprocess_config.ERODE_ITER,
                      min_fac=preprocess_config.MIN_COUNTOUR_FACTOR,
                      max_fac=preprocess_config.MAX_COUNTOUR_FACTOR,
                      valid_ratio=preprocess_config.VALID_RATIO) -> MatLike:
    ''' Removes any background apart from the medium where the text is located '''
    logger.debug("Highlighting the boundary of the note image")

    gray = BGRToGRAY(input)
    thresh = getThreshold(gray)

    # New method overwriting the previous one
    edges = cv2.Canny(thresh, 50, 150)
    cnts = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Apply morphological operations to clean up image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel)
    dilated = cv2.dilate(thresh, kernel, iterations=dilated)
    eroded = cv2.erode(dilated, kernel, iterations=eroded)
    
    # Find contours of the eroded image
    cnts = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    if len(cnts) == 0:
        return input
    
    # Draw contours to verify that the correct region is being selected
    # cv2.drawContours(reversed, cnts, -1, (0, 255, 0), 2)
    # return reversed

    valid_counters = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        aspect_ratio = w / float(h)
        
        min_area = min_fac * input.shape[0] * input.shape[1]
        max_area = max_fac * input.shape[0] * input.shape[1]
        if area > min_area and area < max_area and valid_ratio < aspect_ratio:
            valid_counters.append(c)
        
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
    flip = flipImage(input)
    image = BGRToRGB(flip)
    pixels = image.reshape(-1, 3)

    #K-Means Clustering
    pixels = np.float32(pixels)
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
        cluster_pixels[label].append(pixel)

    # Assign pixels to their clusters
    color_ranges = {}
    for label in dominant_labels:
        cluster_array = np.array(cluster_pixels[label], dtype=np.uint8)
        min_color = np.min(cluster_array, axis=0)
        max_color = np.max(cluster_array, axis=0)
        color_ranges[label] = (min_color.tolist(), max_color.tolist())

    #Identify background as the most frequent cluster
    background_label = dominant_labels[0]
    text_label = dominant_labels[1]

    bg_range = color_ranges[background_label]
    text_range = color_ranges[text_label]
    logger.debug(f"Background Color range (BGR): {bg_range}")
    logger.debug(f"Text Color range (BGR): {text_range}")

    # Convert this RGB range to GRAY range
    text_range = [max(0, int(0.114 * text_range[0][0] + 0.587 * text_range[0][1] + 0.299 * text_range[0][2]) + preprocess_config.MIN_RANGE), 
                  min(255, int(0.114 * text_range[1][0] + 0.587 * text_range[1][1] + 0.299 * text_range[1][2]) + preprocess_config.MAX_RANGE)] # Add offset to handle boundary case values
    bg_range = [max(0, int(0.114 * bg_range[0][0] + 0.587 * bg_range[0][1] + 0.299 * bg_range[0][2])),
                min(255, int(0.114 * bg_range[1][0] + 0.587 * bg_range[1][1] + 0.299 * bg_range[1][2]))]

    logger.debug(f"Text Color range (GRAY): {text_range}")
    logger.debug(f"Background Color range (GRAY): {bg_range}")
    return text_range, bg_range

@log_execution_time
def highlightText(input: MatLike, text_range: list) -> MatLike:
    ''' Highlights text-only regions, excluding everything else (outputting a binary image of text and non-text) '''
    logger.debug("Highlighting text")
    
    text_region = np.array([[[text_range[0], text_range[0], text_range[0]], 
                              [text_range[1], text_range[1], text_range[1]]]]).astype(np.uint8)
    
    try:
        hsv_text_range = BGRToHSV(text_region)
        shaded = BGRToShades(input)
        flip = flipImage(shaded)
        reverted = GRAYToBGR(flip)
        hsv = BGRToHSV(reverted)
    except cv2.error as e:
        logger.error(f"Error: {e}")
        return input

    # Ensure range only considers value, not hue/saturation
    hsv_text_range[0][0][0] = preprocess_config.LOWER_RANGE
    hsv_text_range[0][0][1] = preprocess_config.LOWER_RANGE
    hsv_text_range[0][1][0] = preprocess_config.UPPER_RANGE
    hsv_text_range[0][1][1] = preprocess_config.UPPER_RANGE

    logger.debug(f"HSV Range: {hsv_text_range[0][0]} - {hsv_text_range[0][1]}")

    mask = cv2.inRange(hsv, hsv_text_range[0][0], hsv_text_range[0][1])

    # Apply noise reduction before dilation
    mask = cv2.medianBlur(mask, 3)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, preprocess_config.KERNEL_RATIO)
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Dilate text regions while preserving size constraints
    dilate = cv2.dilate(closed_mask, kernel, iterations=preprocess_config.HIGH_DILATE_ITER)
    
    cnts, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Remove contours that are too small or too elongated (likely lines)
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        ar = w / float(h)
        area = cv2.contourArea(c)

        # Ignore contours that are likely horizontal or vertical lines
        if h < preprocess_config.MIN_HEIGHT or ar > preprocess_config.MAX_AR:
            logger.info(f"Ignoring line-like contour at ({x}, {y}) with AR: {ar}, H: {h}")
            continue

        if area > preprocess_config.MIN_AREA:
            
            cv2.drawContours(dilate, [c], -1, (0, 0, 0), -1)  # Purplish Blue (Valid Text)
        else:
            logger.warning(f"Keeping small contour with AR: {ar}, AREA: {area}")
            cv2.drawContours(dilate, [c], -1, (0, 0, 0), -1)  # Yellow (Possible Small Text)

    logger.debug("Resulting highlighting complete")
    return flipImage(blurImage(cv2.bitwise_and(dilate, flipImage(mask)), 0.2))# Change blur after text extraction to be 0.5

# @log_execution_time
# def highlightText(input: MatLike, text_range: list) -> MatLike:
#     ''' Highlights text-only regions, excluding everything else (outputting a binary image of text and non-text) '''
#     logger.debug("Highlighting text")
#     text_region = np.array([[[text_range[0], text_range[0], text_range[0]], [text_range[1], text_range[1], text_range[1]]]]).astype(np.uint8)
#     try:
#         hsv_text_range = BGRToHSV(text_region)
#         shaded = BGRToShades(input)
#         flip = flipImage(shaded)
#         reverted = GRAYToBGR(flip)
#         hsv = BGRToHSV(reverted)
#     except cv2.error as e:
#         logger.error(f"Error: {e}")
#         return input

#     # Range of hsv should only care about the value rather than the hue and saturation (TODO More elegant solution)
#     hsv_text_range[0][0][0] = preprocess_config.LOWER_RANGE
#     hsv_text_range[0][0][1] = preprocess_config.LOWER_RANGE
#     hsv_text_range[0][1][0] = preprocess_config.UPPER_RANGE
#     hsv_text_range[0][1][1] = preprocess_config.UPPER_RANGE

#     logger.debug(f"HSV Range: {hsv_text_range[0][0]} - {hsv_text_range[0][1]}")
#     mask = cv2.inRange(hsv, hsv_text_range[0][0], hsv_text_range[0][1])

#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, preprocess_config.KERNEL_RATIO)
#     # print(kernel)
#     dilate = cv2.dilate(mask, kernel, iterations=preprocess_config.HIGH_DILATE_ITER)
#     cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     # Remove contours that are too small to be text
#     cnts = cnts[0] if len(cnts) == 2 else cnts[1]
#     # print(len(cnts))
#     for c in cnts:
#         x, y, w, h = cv2.boundingRect(c)
#         ar = w / float(h)
#         area = cv2.contourArea(c)
#         if ar > preprocess_config.ASPECT_RATIO: # Greater than 10 (10:1)
#             cv2.drawContours(hsv, [c], -1, (200, 150, 150), -1) # Blue
#         elif area > preprocess_config.MIN_AREA: # Less than 10 pixels
#             cv2.drawContours(hsv, [c], -1, (250, 150, 150), -1) # Purplish Blue
#         else:
#             logger.warning(f"Keeping contour with AR: {ar}, AREA: {area}")
#             cv2.drawContours(hsv, [c], -1, (50, 150, 150), -1) # Yellow

    

#     print(dilate.shape)
#     return hsv
#     # Verify that the final result here is the binarized output
#     logger.debug("Resulting highlighting\n")
#     return flipImage(blurImage(cv2.bitwise_and(dilate, mask), 0.2))# Change blur after text extraction to be 0.5

@log_execution_time
def preprocessImage(input: MatLike) -> MatLike:
    ''' Main process of preprocessing each step is separated into individual functions '''
    logger.debug("Starting preprocess process")
    try:
        # Apply filters to image
        weighted = contrastImage(input)

        scaled = rescaleImage(weighted)

        skewed = correctSkew(scaled)

        blurred = blurImage(skewed)

        note = highlightBoundary(blurred)

        # Histogram analysis to determine the range of text colors
        text_range, bg_range= findColorRange(note)

        # Exclude everything else except the actual text that make up the note
        result = highlightText(note, text_range)
        logger.debug("Complete preprocessing process\n")

    except Exception as e:
        logger.error(f"Error: {e}")
        return None
    
    # return weighted
    # return scaled
    # return skewed
    # return blurred
    return note
    return result


if __name__ == "__main__":
    logger.info("Testing preprocessing module")

    # Run through all the user-inputted files to ensure proper handling of images (basis)
    # NCR generic sample retrieval
    image_path = "src/NCR_samples/"
    IMAGE_REGEX = r'[a-zA-Z0-9\-]*.jpg'
    logger.debug(f"IMAGE_REGEX: {IMAGE_REGEX}\n")
    files = subprocess.check_output(["ls", image_path]).decode("utf-8")
    file_names = re.findall(IMAGE_REGEX, files)
    joined_files = "\n".join(file_names)
    logger.debug(f"File imported:\n{joined_files}\n")

    images = []
    for name in file_names:
        if os.path.exists(image_path + name):
            images.append(cv2.imread(f"{image_path}{name}"))
        else:
            logger.warning(f"{name} not found in NCR_samples... skipping")
    
    for i in range(len(file_names)):
    # for i in range(0, 5):
        try:
            logger.debug(f"Processing {file_names[i]}")
            result = preprocessImage(images[i])
            if result is None:
                logger.warning(f"No result generated for {file_names[i]}")
                raise Exception("No result found")
            else:
                cv2.imshow(file_names[i], result)
                print("Press 'space' to continue to the next image")
                print("Press 'q' to quit")
                while True:
                    key = cv2.waitKey(1) & 0xFF
                    if key == 32:
                        break
                    if key == ord('q'):
                        print("Quitting...")
                        exit()
                    cv2.moveWindow(file_names[i], 0, 0)

                cv2.destroyAllWindows()
        except Exception as e:
            logger.error(f"Error: {e}")
            logger.debug("Return a None value to the main driver...")
            # return None # Would return when integrated with the main driver

    logger.info("Complete preprocess module")