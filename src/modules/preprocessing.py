# Import the necessary libraries
import cv2
from cv2.typing import MatLike

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

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

def preprocessImage(input: MatLike) -> MatLike:
    shaded = RGBToShades(input)
    # scaled = rescaleImage(shaded)
    blurred = blurImage(shaded)
    hsv = convertToHSV(blurred)
    highlighted = highlightText(hsv)
    result = flipImage(highlighted)

    return result
    
def RGBToShades(input: MatLike ) -> MatLike:
    gray_image = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
    gray_image = gray_image.astype(np.float32)/255

    result = 255 * np.floor(gray_image * preprocess_config.SHADES + 0.5) / preprocess_config.SHADES
    result = result.clip(0 ,255).astype(np.uint8)
    
    return result

def flipImage(input: MatLike) -> MatLike:
    return 255 - input

def blurImage(input: MatLike) -> MatLike:
    gaussian = cv2.GaussianBlur(input, (preprocess_config.KERNEL_DIMS, preprocess_config.KERNEL_DIMS), preprocess_config.GAUSSIAN_SIGMA)
    reverted = cv2.cvtColor(flipImage(gaussian), cv2.COLOR_GRAY2BGR)
    return reverted

def rescaleImage(input: MatLike) -> MatLike:
    pass

def convertToHSV(input: MatLike) -> MatLike:
    hsv = cv2.cvtColor(input, cv2.COLOR_BGR2HSV)
    return hsv

def highlightText(input: MatLike) -> MatLike:
    mask = cv2.inRange(input, preprocess_config.LOWER_MASK, preprocess_config.UPPER_MASK)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, preprocess_config.KERNEL_RATIO)
    dilate = cv2.dilate(mask, kernel, iterations=preprocess_config.DILATE_ITER)

    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        ar = w / float(h)
        if ar < 5:
            cv2.drawContours(dilate, [c], -1, (0, 0, 0), -1)
    
    return flipImage(cv2.bitwise_and(dilate, mask))

# # Check if the image is loaded
# if image is None:
#     raise FileNotFoundError("Image not loaded. Ensure 'black_sampel.jpg' exists in the same directory as the script.")

image = cv2.imread('src/images/black_sample.jpg')
# Convert BGR image to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert the image to grayscale
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Define the scale factors
scale_factor_1 = 3.0  # Increase size
scale_factor_2 = 1/3.0  # Decrease size

# Get the original image dimensions
height, width = image_rgb.shape[:2]

# Resize for zoomed image
new_height = int(height * scale_factor_1)
new_width = int(width * scale_factor_1)
zoomed_image = cv2.resize(image_rgb, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

# Resize for scaled image
new_height1 = int(height * scale_factor_2)
new_width1 = int(width * scale_factor_2)
scaled_image = cv2.resize(image_rgb, (new_width1, new_height1), interpolation=cv2.INTER_AREA)

# Convert the image to grayscale
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Define the scale factors
scale_factor_1 = 3.0  # Increase size
scale_factor_2 = 1/3.0  # Decrease size

# Get the original image dimensions

# Create subplots
fig, axs = plt.subplots(1, 4, figsize=(15, 4))


# SHOWING STEPS
# Plot the original image
axs[0].imshow(image_rgb)
axs[0].set_title(f'Original Image\nShape: {image_rgb.shape}')

# Plot the grayscale image
axs[1].imshow(image_gray, cmap='gray')  # Specify cmap for grayscale
axs[1].set_title(f'Grayscale Image\nShape: {image_gray.shape}')

# Plot the zoomed-in image
axs[2].imshow(zoomed_image)
axs[2].set_title(f'Zoomed Image\nShape: {zoomed_image.shape}')

# Plot the scaled-down image
axs[3].imshow(scaled_image)
axs[3].set_title(f'Scaled Image\nShape: {scaled_image.shape}')

# Remove axis ticks
for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])

# Display the images
plt.tight_layout()
plt.show()



if __name__ == "__main__":
    print("Testing preprocessing module")
    
    sample_image = cv2.imread("src/images/black_sample.jpg")

    result = preprocessImage(sample_image)
    
    cv2.imshow("Original", sample_image)
    cv2.imshow("Result", result)
    cv2.waitKey()

    print("Complete preprocess module")