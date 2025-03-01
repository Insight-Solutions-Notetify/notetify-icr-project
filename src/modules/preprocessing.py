# Import the necessary libraries
import cv2
from cv2.typing import MatLike

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from typing import Set
from collections import Counter

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
    img_width, _, _  = input.shape

    if img_width > preprocess_config.MAX_WIDTH:
        IMG_RATIO = preprocess_config.MAX_WIDTH / img_width
    else:
        IMG_RATIO = img_width / preprocess_config.MAX_WIDTH

    result = cv2.resize(input,(0, 0), fx=IMG_RATIO, fy=IMG_RATIO)
    
    return result

def findColorRange(input: MatLike, k = 2) -> Set:
    
    image = BGRToRGB(input)
    
    pixels = image.reshape(-1, 3)

    #K-Means Clustering
    pixels = np.float32(pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, center = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Counte each cluster's occurence
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

    background_range = color_ranges[background_label]
    text_range = color_ranges[text_label]

    return background_range, text_range

def rangeOfText(input: MatLike) -> Set:
    ''' Analyze the range of text colors in the image '''
    hist = cv2.calcHist([input], [0], None, [256], [0, 256])
    hist = hist[preprocess_config.LOWER_RANGE:preprocess_config.UPPER_RANGE]

    text_range = set()
    foreground_range = set()

    for i in range(len(hist)):
        if hist[i] > preprocess_config.TEXT_THRESHOLD:
            text_range.add(i)
        else:
            foreground_range.add(i)

    return (text_range, foreground_range)


def highlightBoundary(input: MatLike, text_range: Set, foreground_range: Set) -> MatLike:
    ''' Removes any background apart from the medium where the the text is located'''
    x_box, y_box = 0, 0
    min_width, min_height = 0, 0

    # Need to implement range of text colors to dynamically handle a wide variety of light conditions and color ranges of text and background
    # shaded = BGRToShades(input)
    # analyzed, text_range, foreground_range = rangeOfText(shaded)
    # reverted = flipImage(GRAYToBGR(analyzed))
    

    flipped = flipImage(input)
    shaded = cv2.cvtColor(flipped, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(shaded, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    dilate = cv2.dilate(thresh, kernel, iterations=preprocess_config.DILATE_ITER)
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        width_ratio = w / float(preprocess_config.MAX_WIDTH)
        ar = w / float(h)
        if width_ratio > 0.7:
            cv2.drawContours(shaded, [c], -1, (0, 0, 0), -1)
            min_width = max(min_width, w)
            min_height = max(min_height, h)
            x_box, y_box = max(x_box, x), max(y_box, y)
    
    result = shaded[y_box:y_box + min_height, x_box:x_box + min_width]
    return flipImage(GRAYToBGR(shaded))

def highlightText(input: MatLike, text_range: Set, foreground_range: Set) -> MatLike:
    ''' Highlights text-only regions, excluding everything else (outputting a binary image of text and non-text) '''

    # Need to implement range of text colors to dynamically handle a wide variety of light conditions and color ranges of text and background
    # Need to implement range of text to
    # shaded = BGRToShades(input)
    # analyzed, text_range, foreground_range = rangeOfText(shaded)
    # reverted = flipImage(GRAYToBGR(analyzed))
    hsv = BGRToHSV(input)

    mask = cv2.inRange(hsv, preprocess_config.LOWER_MASK, preprocess_config.UPPER_MASK)

    # return cv2.bitwise_and(input, mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, preprocess_config.KERNEL_RATIO)
    dilate = cv2.dilate(mask, kernel, iterations=preprocess_config.DILATE_ITER)
    cnts = cv2.findContours(flipImage(dilate), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Remove contours that are too small to be text
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # print(len(cnts))
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        ar = w / float(h)
        # print(ar)
        if ar < preprocess_config.ASPECT_RATIO:
            cv2.drawContours(dilate, [c], -1, (0, 0, 0), -1)

    # return dilate
    return blurImage(cv2.bitwise_and(dilate, mask), 0.2) # Change blur after text extraction to be 0.5



def preprocessImage(input: MatLike) -> MatLike:
    ''' Main process of preprocessing each step is separated into individual functions '''

    # Apply filters to image
    weighted = contrastImage(input)
    scaled = rescaleImage(weighted)
    blurred = blurImage(scaled)

    # Histogram analysis to determine the range of text colors
    (text_range, bg_range) = findColorRange(blurred)

    print(f"Text Color range (RGB): {text_range}")
    print(f"Background Color Range (RGB): {bg_range}")

    # Exclude everything else except the region of the note
    note = highlightBoundary(blurred, text_range, bg_range)

    return note
    # Exclude everything else except the actual text that make up the note
    result = highlightText(note, text_range, foreground_range)

    return result


if __name__ == "__main__":
    print("Testing preprocessing module")
    
    sample_image = cv2.imread("src/images/black_sample.jpg")
    # hist = cv2.calcHist([sample_image], [0], None, [256], [0, 256])
    # plt.plot(hist)
    # plt.show()
    # cv2.imshow("Original", sample_image)

    result = preprocessImage(sample_image)
    cv2.imshow("Result", result)
    
    # inverse = flipImage(result)
    # cv2.imshow("Inverted Result", inverse)
    cv2.waitKey(0)

    print("Complete preprocess module")


# # # Check if the image is loaded
# # if image is None:
# #     raise FileNotFoundError("Image not loaded. Ensure 'black_sampel.jpg' exists in the same directory as the script.")

# image = cv2.imread('src/images/black_sample.jpg')
# # Convert BGR image to RGB
# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# # Convert the image to grayscale
# image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Define the scale factors
# scale_factor_1 = 3.0  # Increase size
# scale_factor_2 = 1/3.0  # Decrease size

# # Get the original image dimensions
# height, width = image_rgb.shape[:2]

# # Resize for zoomed image
# new_height = int(height * scale_factor_1)
# new_width = int(width * scale_factor_1)
# zoomed_image = cv2.resize(image_rgb, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

# # Resize for scaled image
# new_height1 = int(height * scale_factor_2)
# new_width1 = int(width * scale_factor_2)
# scaled_image = cv2.resize(image_rgb, (new_width1, new_height1), interpolation=cv2.INTER_AREA)

# # Convert the image to grayscale
# image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Define the scale factors
# scale_factor_1 = 3.0  # Increase size
# scale_factor_2 = 1/3.0  # Decrease size

# # Get the original image dimensions

# # Create subplots
# fig, axs = plt.subplots(1, 4, figsize=(15, 4))


# # SHOWING STEPS
# # Plot the original image
# axs[0].imshow(image_rgb)
# axs[0].set_title(f'Original Image\nShape: {image_rgb.shape}')

# # Plot the grayscale image
# axs[1].imshow(image_gray, cmap='gray')  # Specify cmap for grayscale
# axs[1].set_title(f'Grayscale Image\nShape: {image_gray.shape}')

# # Plot the zoomed-in image
# axs[2].imshow(zoomed_image)
# axs[2].set_title(f'Zoomed Image\nShape: {zoomed_image.shape}')

# # Plot the scaled-down image
# axs[3].imshow(scaled_image)
# axs[3].set_title(f'Scaled Image\nShape: {scaled_image.shape}')

# # Remove axis ticks
# for ax in axs:
#     ax.set_xticks([])
#     ax.set_yticks([])

# # Display the images
# plt.tight_layout()
# plt.show()