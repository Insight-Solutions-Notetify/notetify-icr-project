# Import the necessary libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
# image = cv2.imread('black_sampel.jpg')

# # Check if the image is loaded
# if image is None:
#     raise FileNotFoundError("Image not loaded. Ensure 'black_sampel.jpg' exists in the same directory as the script.")

image = cv2.imread(r'C:\Users\glee5\OneDrive\Documents\GitHub\notetify-icr-project\src\modules\black_sampel.jpg')
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

# Create subplots
fig, axs = plt.subplots(1, 4, figsize=(15, 4))

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
