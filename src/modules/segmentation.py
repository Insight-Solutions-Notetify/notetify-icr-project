import cv2
from cv2.typing import MatLike

import numpy as np
import os
import sys

import subprocess
import re
import logging
import preprocessing

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)
os.chdir(project_root)

from src.config.segmentation_config import segmentation_config
from src.modules.logger import logger
import matplotlib.pyplot as plt

# TODO - Remove preprocess image function and config parameters (moved to segmentation_config)

# --- FIX 1: Use a clearer name for max_value in thresholding. ---
# Instead of THRESH_BINARY_INV = 255, let's use MAX_VALUE = 255 
# to avoid confusion with cv2.THRESH_BINARY_INV.
MAX_VALUE = 255

# Constants for preprocessing and segmentation
BLUR_KERNEL_SIZE = (3, 3)

def preprocess_image(image_path): 
    """
    Preprocess the input image by converting it to grayscale, binarizing, and reducing noise.
    """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # --- FIX 1 continued: Use MAX_VALUE for the second argument in adaptiveThreshold ---
        binary = cv2.adaptiveThreshold(
            gray,
            MAX_VALUE,  # was THRESH_BINARY_INV (confusing)
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV,
            11,
            8
        )
        # Apply a small Gaussian blur to reduce noise
        binary = cv2.GaussianBlur(binary, BLUR_KERNEL_SIZE, 0)
        return binary
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        return None

def deskew(image):
    """
    Deskew the image to correct for any rotation.
    """
    # Identify all 'on' pixels
    coords = np.column_stack(np.where(image > 0))
    
    # --- FIX 2: Handle the case where coords might be empty ---
    if coords.shape[0] == 0:
        logger.warning("No foreground pixels found. Skipping deskew.")
        return image  # Return as-is if there's nothing to rotate
    
    angle = cv2.minAreaRect(coords)[-1]
    
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    # Use borderMode=cv2.BORDER_REPLICATE to reduce black borders
    return cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


## END TODO
# def plot_projection(binary_image):
#     projection = np.sum(binary_image, axis=1)
#     plt.plot(projection)
#     plt.title("Horizontal Projection Profile")
#     plt.xlabel("Row Index")
#     plt.ylabel("Sum of Pixel Values")
#     plt.show()

# Call this function on your binary image


def segment_lines(image):
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image to binary
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Sum pixel values horizontally
    horizontal_projection = np.sum(thresh, axis=1)

    # Detect lines based on where the projection is > 0 (text exists)
    lines = []
    start = None
    for i, row_sum in enumerate(horizontal_projection):
        if row_sum > 0 and start is None:
            start = i
        elif row_sum == 0 and start is not None:
            end = i
            line_img = thresh[start:end, :]
            if (end - start) > 10:  # Filter small noise
                lines.append(line_img)
            start = None

    return lines
    

def segment_words(line_image, min_gap=segmentation_config.MIN_WORD_GAP):
    """
    Segment a line image into individual words based on vertical projection.
    """
    try:
        projection = np.sum(line_image, axis=0)
        
        # Simple threshold = 10% of max projection
        threshold = np.max(projection) * 0.1
        
        # Avoid zero-threshold edge case
        if threshold == 0:
            logger.warning("Word projection threshold is zero. Check your image or threshold logic.")
            return []
        
        word_indices = np.where(projection > threshold)[0]
        words = []
        if len(word_indices) == 0:
            logger.warning("No words detected in the line.")
            return words

        # Group columns into word segments
        start_idx = word_indices[0]
        for i in range(1, len(word_indices)):
            if word_indices[i] - word_indices[i - 1] > min_gap:
                words.append(line_image[:, start_idx:word_indices[i]])
                start_idx = word_indices[i]
        words.append(line_image[:, start_idx:])
        return words
    except Exception as e:
        logger.error(f"Error in word segmentation: {e}")
        return []

def segment_characters(word_image, min_char_size=segmentation_config.MIN_CHAR_SIZE):
    """
    Segment a word image into individual characters using contour detection.
    """
    try:
        # We assume word_image is already binarized
        contours, _ = cv2.findContours(word_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        characters = []

        # Sort contours from left-to-right by x-coordinate
        for contour in sorted(contours, key=lambda c: cv2.boundingRect(c)[0]):
            x, y, w, h = cv2.boundingRect(contour)
            # Filter out small blobs
            if w >= min_char_size[0] and h >= min_char_size[1]:
                char = word_image[y:y + h, x:x + w]
                # Optional: resize the character
                char_resized = cv2.resize(char, (32, 32), interpolation=cv2.INTER_AREA)
                characters.append(char_resized)
        return characters
    except Exception as e:
        logger.error(f"Error in character segmentation: {e}")
        return []

def save_images(images, folder, prefix):
    """
    Save a list of images to the specified folder with a given prefix.
    """
    os.makedirs(folder, exist_ok=True)
    for i, img in enumerate(images):
        cv2.imwrite(f"{folder}/{prefix}_{i + 1}.png", img)

## TODO Rely on preprocessing module

def segmentate_image(image: MatLike, output_dir: str) -> list[MatLike]:
    """
    Complete processing pipeline for an image: preprocess, deskew, segment lines, words, and characters.
    Returns the segmented line images as a list of image arrays.
    """
    lines = segment_lines(image)
    logger.debug(f"Detected {len(lines)} lines.")
    save_images(lines, os.path.join(output_dir, "lines"), "line")

    for i, line in enumerate(lines):
        words = segment_words(line)
        logger.debug(f"Line {i + 1}: Detected {len(words)} words.")
        save_images(words, os.path.join(output_dir, f"line_{i + 1}_words"), "word")

        for j, word in enumerate(words):
            characters = segment_characters(word)
            logger.debug(f"Word {j + 1}: Detected {len(characters)} characters.")
            save_images(
                characters,
                os.path.join(output_dir, f"line_{i + 1}_word_{j + 1}_characters"),
                "char"
            )

    logger.debug("Complete segmentation\n")
    return lines

if __name__ == "__main__":
    logger.info("Testing segmentation module")

    output_dir = "segmented_output"

    # Run through all the user-inputted files to ensure proper handling of images (basis)
    # NCR generic sample retrieval
    image_path = "src/seg_images/"
    IMAGE_REGEX = r'[a-zA-Z0-9\-]*.jpg'
    logger.debug(f"IMAGE_REGEX: {IMAGE_REGEX}\n")
    logger.info("Loading files using os.listdir()")
    if not os.path.exists(image_path):
        logger.error(f"Directory does not exist: {image_path}")
        sys.exit(1)
    files = os.listdir(image_path)
    file_names = [f for f in files if re.match(IMAGE_REGEX, f)]
    joined_files = "\n".join(file_names)
    logger.debug(f"Files imported:\n{joined_files}")

    images = []
    for name in files:
        if os.path.exists(image_path + name):
            images.append(cv2.imread(f"{image_path}{name}"))
        else:
            logger.warning(f"{name} not found in images... skipping")
    
    for i in range(len(file_names)):
        # preprocessed = preprocessing.preprocessImage(images[i])
        logger.debug(f"Segmenting image {file_names[i]}")
        #segmented = segmentate_image(file_names[i], output_dir)
        segmented = segmentate_image(images[i], output_dir)

        for idx, line_img in enumerate(segmented):
            cv2.imshow(f"Line {idx+1}", line_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        event = input("Press Enter to continue...\nPress q to exit...")
        if event == "q":
            break
    
    logger.info("Complete segmentation module")