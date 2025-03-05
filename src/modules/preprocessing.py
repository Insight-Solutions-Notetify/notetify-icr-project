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

def rotateImage(input: MatLike) -> MatLike:
    ''' Rotate image to be upright '''
    gray = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(threshold > 0))
    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    
    (h, w) = input.shape[:2]
    center = (w // 2, h // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(input, rot_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated_image

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
    thresh = cv2.threshold(shaded, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    dilate = cv2.dilate(thresh, kernel, iterations=preprocess_config.DILATE_ITER)
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    x_box, y_box, min_width, min_height = float('inf'), float('inf'), 0, 0

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        # Ignore small contours and contours that match the image size
        # print(f"Width: {w}, Height: {h}")
        if w == shaded.shape[1] or h == shaded.shape[0]:
            continue
        width_ratio = w / float(preprocess_config.IMG_WIDTH)
        # print(width_ratio)
        if width_ratio > 0.001:
            x_box = min(x_box, x)
            y_box = min(y_box, y)
            min_width = max(min_width, x + w)
            min_height = max(min_height, y + h)

    # print(f"Cropped Box: {x_box}, {y_box}, {min_width}, {min_height}")
    if x_box == float('inf') or y_box == float('inf'):
        return input

    cropped = reversed[y_box:min_height, x_box:min_width]
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
    rotated = rotateImage(weighted)
    scaled = rescaleImage(rotated)
    # print(scaled.shape)
    blurred = blurImage(scaled)

    # Exclude everything else except the region of the note
    note = highlightBoundary(blurred)

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
    
    sample_1 = cv2.imread("src/images/black_sample.jpg")
    sample_2 = cv2.imread("src/images/small.jpg")
    sample_3 = cv2.imread("src/images/test_sample_2.jpg")
    sample_4 = cv2.imread("src/images/pink_slanted.jpg")
    sample_5 = cv2.imread("src/images/green_background.jpg")
    sample_6 = cv2.imread("src/images/distraction_colors.jpg")
    sample_7 = cv2.imread("src/images/problem_1.jpg")
    sample_8 = cv2.imread("src/images/blue_slanted.jpg")

    result = preprocessImage(sample_1)
    cv2.imshow("Result: Blue Text", result)
    result2 = preprocessImage(sample_2)
    cv2.imshow("Result: Small", result2)
    result3 = preprocessImage(sample_3)
    cv2.imshow("Result: Scribble", result3)
    cv2.waitKey(0)
    reuslt4 = preprocessImage(sample_4)
    cv2.imshow("Result: Pink Slanted", reuslt4)
    result5 = preprocessImage(sample_5)
    cv2.imshow("Result: Green Background", result5)
    result6 = preprocessImage(sample_6)
    cv2.waitKey(0)
    cv2.imshow("Result: Distraction Colors", result6)
    result7 = preprocessImage(sample_7)
    cv2.imshow("Result: Problem 1", result7)
    result8 = preprocessImage(sample_8)
    cv2.imshow("Result: Blue Slanted", result8)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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