import cv2
import numpy as np
import os
import sys
import logging

# Set up logging for better debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Constants for preprocessing and segmentation
MAX_VALUE = 255  # For thresholding
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
        
        # Adaptive thresholding to convert to binary image
        binary = cv2.adaptiveThreshold(
            gray,
            MAX_VALUE, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV,
            15,
            10
        )
        # Apply Gaussian blur to reduce noise
        binary = cv2.GaussianBlur(binary, BLUR_KERNEL_SIZE, 0)
        return binary
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        return None

def segment_lines(binary_image, min_gap=50):
    """
    Segment the binary image into individual lines based on horizontal projection.
    """
    try:
        projection = np.sum(binary_image, axis=1)
        threshold = np.max(projection) * 0.2
        
        # Avoid zero-threshold edge case
        if threshold == 0:
            logger.warning("Projection threshold is zero.")
            return []
        
        line_indices = np.where(projection > threshold)[0]
        lines = []
        if len(line_indices) == 0:
            logger.warning("No text lines detected.")
            return lines
        
        start_idx = line_indices[0]
        for i in range(1, len(line_indices)):
            if line_indices[i] - line_indices[i - 1] > min_gap:
                lines.append(binary_image[start_idx:line_indices[i], :])
                start_idx = line_indices[i]
        lines.append(binary_image[start_idx:, :])
        return lines
    except Exception as e:
        logger.error(f"Error in line segmentation: {e}")
        return []

def segment_words(line_image, min_gap=10):
    """
    Segment a line image into individual words based on vertical projection.
    """
    try:
        projection = np.sum(line_image, axis=0)
        threshold = np.max(projection) * 0.1
        
        # Avoid zero-threshold edge case
        if threshold == 0:
            logger.warning("Word projection threshold is zero.")
            return []
        
        word_indices = np.where(projection > threshold)[0]
        words = []
        if len(word_indices) == 0:
            logger.warning("No words detected in the line.")
            return words

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

def save_images(images, folder, prefix):
    """
    Save a list of images to the specified folder with a given prefix.
    """
    os.makedirs(folder, exist_ok=True)
    for i, img in enumerate(images):
        cv2.imwrite(f"{folder}/{prefix}_{i + 1}.png", img)

def segmentate_image(image_path: str, output_dir: str):
    """
    Complete processing pipeline for an image: preprocess, segment lines, segment words.
    """
    logger.info("Starting segmentation pipeline.")

    # Preprocess the image
    binary = preprocess_image(image_path)
    if binary is None:
        logger.error("Image preprocessing failed.")
        return

    # Segment lines
    lines = segment_lines(binary)
    logger.debug(f"Detected {len(lines)} lines.")
    save_images(lines, os.path.join(output_dir, "lines"), "line")

    # Segment words within each line
    for i, line in enumerate(lines):
        words = segment_words(line)
        logger.debug(f"Line {i + 1}: Detected {len(words)} words.")
        save_images(words, os.path.join(output_dir, f"line_{i + 1}_words"), "word")

    logger.info("Segmentation pipeline complete.")

if __name__ == "__main__":
    logger.info("Starting segmentation module")

    output_dir = "segmented_output"
    image_path = "src/NCR_samples/example_image.jpg"  # Path to the image

    segmentate_image(image_path, output_dir)

    logger.info("Complete segmentation module")
