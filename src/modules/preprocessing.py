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

### PREPROCESS MODULE ###

# Preprocess Module will process type .jpg images to prepare for segmentation
# Requirements:
# - Input is the MatLike surface since the main driver already retrieved image and convert it to cv2 ready format
# - non-text regions are excluded from the image
# - binary result of image either being white (1) or black(0)
# - Resizing/ De-blurring for varying image inputs and quality.
# - Range of desired shades of handwriting to be included
# - Mask on desired range

### NS4-28-converison-functions
# Several functions for conversion should be accurate and verified 
def BGRToShades(input: MatLike ) -> MatLike:
    ''' Convert BGR to specialized Shades of GRAY'''
    gray_image = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
    gray_image = gray_image.astype(np.float32)/255

    result = 255 * np.floor(gray_image * preprocess_config.SHADES + 0.5) / preprocess_config.SHADES
    result = result.clip(0 ,255).astype(np.uint8)
    
    return result

def GRAYToBGR(input: MatLike) -> MatLike:
    ''' Convert GRAY to BGR '''
    reverted = cv2.cvtColor(input, cv2.COLOR_GRAY2BGR)
    return reverted

def BGRToHSV(input: MatLike) -> MatLike:
    ''' Convert BGR to HSV '''
    hsv = cv2.cvtColor(input, cv2.COLOR_BGR2HSV)
    return hsv

def BGRToRGB(input: MatLike) -> MatLike:
    ''' Convert BGR to RGB '''
    rgb = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
    return rgb

def flipImage(input: MatLike) -> MatLike:
    ''' Inverse of inputted image '''
    return 255 - input

def contrastImage(input: MatLike, contrast=preprocess_config.CONTRAST, brightness=preprocess_config.BRIGHTNESS):
    ''' Apply a contrast and brightness adjustment to the image '''
    adjusted_image = cv2.addWeighted(input, contrast, np.zeros(input.shape, input.dtype), 0, brightness)
    return adjusted_image

def blurImage(input: MatLike, sigma=preprocess_config.GAUSSIAN_SIGMA) -> MatLike:
    gaussian = cv2.GaussianBlur(input, (preprocess_config.KERNEL_DIMS, preprocess_config.KERNEL_DIMS), sigma)
    return gaussian

def rescaleImage(input: MatLike) -> MatLike:
    ''' Rescale images down to a standard acceptable for input '''
    img_width, img_height, _  = input.shape

    # if img_width > preprocess_config.IMG_WIDTH:
    #     IMG_RATIO = preprocess_config.IMG_WIDTH / img_width
    # else:
    #     IMG_RATIO = img_width / preprocess_config.IMG_WIDTH

    IMG_RATIO = img_width / img_height

    # print(IMG_RATIO)

    result = cv2.resize(input,(preprocess_config.IMG_WIDTH, int(preprocess_config.IMG_WIDTH * IMG_RATIO))) #, fx=IMG_RATIO, fy=IMG_RATIO)

    # print("SHAPE:", result.shape)
    
    return result

def correctSkew(input: MatLike, delta=10, limit=40) -> MatLike:
    ''' Correct skew of image'''
    def determine_score(arr, angle):
        data = rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1, dtype=float)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
        return histogram, score

    gray = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        # print(angle)
        histogram, score = determine_score(thresh, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = input.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    corrected = cv2.warpAffine(input, M, (w, h), flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)

    print(f"Best Angle: {best_angle}")
    return corrected

def findColorRange(input: MatLike, k = 2) -> Set:
    ''' Identify the color range for text and background by using k-clustering '''
    image = BGRToRGB(input)
    
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

    # Convert this RGB range to GRAY range
    text_range = [max(0, int(0.114 * text_range[0][0] + 0.587 * text_range[0][1] + 0.299 * text_range[0][2]) + preprocess_config.MIN_RANGE), min(255, int(0.114 * text_range[1][0] + 0.587 * text_range[1][1] + 0.299 * text_range[1][2]) + preprocess_config.MAX_RANGE)] # Add offset to handle boundary case values
    # bg_range = [int(0.114 * bg_range[0][0] + 0.587 * bg_range[0][1] + 0.299 * bg_range[0][2]), int(0.114 * bg_range[1][0] + 0.587 * bg_range[1][1] + 0.299 * bg_range[1][2])]
    return text_range

def highlightBoundary(input: MatLike) -> MatLike:
    ''' Removes any background apart from the medium where the text is located '''
    flipped = flipImage(input)
    shaded = BGRToShades(flipped)
    reversed = cv2.cvtColor(shaded, cv2.COLOR_GRAY2BGR)
    gray = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    dilate = cv2.dilate(thresh, kernel, iterations=preprocess_config.DILATE_ITER)
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # New method overwriting the previous one
    edges = cv2.Canny(gray, 50, 150)
    cnts = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # return dilate

    # Draw contours to verify that the correct region is being selected
    # cv2.drawContours(reversed, cnts[0], -1, (0, 255, 0), 2)
    # return reversed
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    x_box, y_box, min_width, min_height = float('inf'), float('inf'), 0, 0

    largest_contour = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    # If largest contour is too small try other method
    print(f"Size: {w * h}")
    print(f"Threshold: {0.4 * shaded.shape[0] * shaded.shape[1]}")
    if w * h > 0.2 * (shaded.shape[0] * shaded.shape[1]):
        cropped = reversed[y:y + h, x:x + w]
        return cropped
    else:
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            # Ignore small contours and contours that match the image size
            # print(f"Width: {w}, Height: {h}")
            if w == shaded.shape[1] or h == shaded.shape[0]:
                continue
            width_ratio = w / float(preprocess_config.IMG_WIDTH)
            
            print(f"Width Ratio: {width_ratio}")
            if width_ratio > 0.001 and width_ratio < 0.92:
                if (w * h) < 0.1 * (shaded.shape[0] * shaded.shape[1]): # Ignore small contours
                    continue
                if (w * h) > 0.5 * (shaded.shape[0] * shaded.shape[1]): # Ignore large contours
                    continue
                # print(f"Box: {x}, {y}, {w}, {h}")
                x_box = min(x_box, x)
                y_box = min(y_box, y)
                min_width = max(min_width,x + w)
                min_height = max(min_height,y + h)

        # print(f"Cropped Box: {x_box}, {y_box}, {min_width}, {min_height}")
        if x_box == float('inf') or y_box == float('inf'):
            return reversed
        
        cropped = reversed[y_box:min_height - 100, x_box:min_width]
        return cropped
    

def highlightText(input: MatLike, text_range: list) -> MatLike:
    ''' Highlights text-only regions, excluding everything else (outputting a binary image of text and non-text) '''
    text_region = np.array([[[text_range[0], text_range[0], text_range[0]], [text_range[1], text_range[1], text_range[1]]]]).astype(np.uint8)
    try:
        hsv_text_range = BGRToHSV(text_region)
        hsv = BGRToHSV(input)
    except cv2.error as e:
        print(f"Error: {e}")
        return input

    # Range of hsv should only care about the value rather than the hue and saturation (TODO More elegant solution)
    hsv_text_range[0][0][0] = preprocess_config.LOWER_RANGE
    hsv_text_range[0][0][1] = preprocess_config.LOWER_RANGE
    hsv_text_range[0][1][0] = preprocess_config.UPPER_RANGE
    hsv_text_range[0][1][1] = preprocess_config.UPPER_RANGE

    # print(hsv_text_range)
    mask = cv2.inRange(hsv, hsv_text_range[0][0], hsv_text_range[0][1])
    # return mask
    # return flipImage(cv2.bitwise_and(input, hsv, mask=mask))

    # mask = cv2.inRange(hsv, preprocess_config.LOWER_MASK, preprocess_config.UPPER_MASK)

    # return cv2.bitwise_and(input, mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, preprocess_config.KERNEL_RATIO)
    # print(kernel)d
    dilate = cv2.dilate(mask, kernel, iterations=preprocess_config.DILATE_ITER)
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Remove contours that are too small to be text
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # print(len(cnts))
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        ar = w / float(h)
        # print(ar)
        if ar < preprocess_config.ASPECT_RATIO:
            # print(ar)
            cv2.drawContours(dilate, [c], -1, (0, 0, 0), -1)

    # return dilate
    # Verify that the final result here is the binarized output
    return flipImage(blurImage(cv2.bitwise_and(dilate, mask), 0.2))# Change blur after text extraction to be 0.5

def preprocessImage(input: MatLike) -> MatLike:
    ''' Main process of preprocessing each step is separated into individual functions '''

    # Apply filters to image
    weighted = contrastImage(input)
    skewed = correctSkew(weighted)
    scaled = rescaleImage(skewed)
    # return scaled
    # print(scaled.shape)
    blurred = blurImage(scaled)
    # return blurred

    # Exclude everything else except the region of the note
    note = highlightBoundary(blurred)
    # return note
    # Histogram analysis to determine the range of text colors
    text_range = findColorRange(note)

    # return note
    # print(f"Text Color range (GRAY): {text_range}")
    # print(f"Background Color Range (RGB): {bg_range}")

    # Exclude everything else except the actual text that make up the note
    result = highlightText(note, text_range)

    return result


if __name__ == "__main__":
    print("Testing preprocessing module")

    # Run through all the user-inputted files to ensure proper handling of images (basis)
    # NCR generic sample retrieval
    image_path = "src/NCR_samples/"
    IMAGE_REGEX = r'[a-zA-Z0-9\-]*.jpg'
    files = subprocess.check_output(f"ls {image_path}").decode("utf-8")
    file_names = re.findall(IMAGE_REGEX, files)
    # print(file_names)

    images = []
    for name in file_names:
        if os.path.exists(image_path + name):
            images.append(cv2.imread(f"{image_path}{name}"))
        else:
            print(f"{name} not found in NCR_samples... skipping")
    
    for i in range(len(file_names)):
        result = preprocessImage(images[i])
        cv2.imshow(file_names[i], result)
        cv2.moveWindow(file_names[i], 0, 0)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print("Complete preprocess module")