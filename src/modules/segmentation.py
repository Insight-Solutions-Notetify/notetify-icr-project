import cv2
from cv2. typing import MatLike

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

# TODO - Remove preprocess image function and config parameters (moved to segmentation_config)

# --- FIX 1: Use a clearer name for max_value in thresholding. ---
# Instead of THRESH_BINARY_INV = 255, let's use MAX_VALUE = 255 
# to avoid confusion with cv2.THRESH_BINARY_INV.
MAX_VALUE = 255

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

def segment_characters(word_image: MatLike, char_size=segmentation_config.MIN_CHAR_SIZE):
    """
    Segment a word image into individual characters using contour detection.
    """
    ## TODO
    # 1. Ensure the input image is binary - DONE
    # 2. Find contours of characters (white characters on black background) - DONE
    # 3. If contours overlap in the x-direction, merge them - DONE
    # 4. Filter out small contours based on area and aspect ratio - DONE
    # 5. Optional: resize the character to a fixed size (e.g., 26x26) - DONE
    # 6. Return a list of character images as a list
    # 7. For testing: display the word image with vertical gaps
    try:
        flipped = cv2.bitwise_not(word_image)

        # Find contours of characters
        cnts, _ = cv2.findContours(flipped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Sort contours from left to right based on x-coord
        cnts = sorted(cnts, key=lambda ctr: cv2.boundingRect(ctr)[0])

        word_image = cv2.cvtColor(word_image, cv2.COLOR_GRAY2BGR)

        logger.debug(f"Detected {len(cnts)} contours.")
        # Merge overlapping contours in the x-direction
        logger.debug("Merging overlapping contours")
        for i in range(len(cnts) - 1):
            for j in range(i + 1, len(cnts)):
                if len(cnts[i]) == 0:
                    continue
                x1, y1, w1, h1 = cv2.boundingRect(cnts[i])
                x2, y2, w2, h2 = cv2.boundingRect(cnts[j])
                if x2 - (x1 + w1) < segmentation_config.MERGING_MIN_X:
                    if y2 - (y1 + h1) < segmentation_config.MERGING_MIN_Y:
                        if w1 + w2 > char_size[0]:
                            cnts[i] = np.concatenate((cnts[i], cnts[j]))
                            cnts[j] = np.array([])
        
        # Remove empty contours
        cnts = [c for c in cnts if len(c) > 0]
        logger.debug(f"Detected {len(cnts)} contours after merging.")

        cnts = sorted(cnts, key=lambda ctr: cv2.boundingRect(ctr)[0])
        characters_images = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h

            # Filter out small contours based on area and aspect ratio
            if area < word_image.shape[0] * segmentation_config.HEIGHT_INFLUENCE:
                continue

            char_resized = cv2.resize(word_image[0:word_image.shape[0],
                                             max(0, x - segmentation_config.WIDTH_CHAR_BUFFER):
                                             min(word_image.shape[1],
                                             x + w + segmentation_config.WIDTH_CHAR_BUFFER)], 
                                      segmentation_config.IMAGE_DIMS, 
                                      interpolation=cv2.INTER_AREA)
            
            characters_images.append(char_resized)
        
    except Exception as e:
        logger.error(f"Error in character segmentation: {e}")
        return []

    return characters_images

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

def test_character_segmentation(word: str, output_dir: str) -> None:
    """
    Test the character segmentation function on a sample image of words.
    """
    logger.debug("Testing character segmentation alone")
    image = cv2.imread(f"./src/NCR_samples/{word}", cv2.IMREAD_GRAYSCALE)
    if image is None:
        logger.error(f"Image not found at {word}")
        return
    binary = cv2.threshold(image, 0, MAX_VALUE, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    characters = segment_characters(binary)
    logger.debug(f"Detected {len(characters)} characters.")
    if len(characters) == 1:
        save_images(characters, os.path.join(output_dir, f"words_{word}"), "character")
    else:
        save_images(characters, os.path.join(output_dir, f"words_{word}"), "character" )
    logger.debug("Character segmentation test complete.")

def test_word_segmentation(word: str, output_dir: str) -> None:
    """
    Test the word segmentation function on a sample image of lines
    """
    logger.debug("Testing word segmentation alone")
    image = cv2.imread(f"./src/NCR_samples/{word}", cv2.IMREAD_GRAYSCALE)
    if image is None:
        logger.error(f"Image not found at {word}")
        return
    characters = segment_words(image)
    logger.debug(f"Detected {len(characters)} words.")
    if len(characters) == 1:
        save_images(characters, os.path.join(output_dir, f"line_{word}"), "word")
    else:
        save_images(characters, os.path.join(output_dir, f"line_{word}"), "word" )
    logger.debug("Word segmentation test complete.")

def test_line_segmentation(word: str, output_dir: str) -> None:
    """
    Test the word segmentation function on a sample image of lines
    """
    logger.debug("Testing line segmentation alone")
    image = cv2.imread(f"./src/NCR_samples/{word}", cv2.IMREAD_GRAYSCALE)
    if image is None:
        logger.error(f"Image not found at {word}")
        return
    characters = segment_lines(image)
    logger.debug(f"Detected {len(characters)} lines.")
    if len(characters) == 1:
        save_images(characters, os.path.join(output_dir, f"image_{word}"), "lines")
    else:
        save_images(characters, os.path.join(output_dir, f"image_{word}"), "lines" )
    logger.debug("Line segmentation test complete.")

if __name__ == "__main__":
    logger.info("Starting segmentation module")

    output_dir = "segmented_output"
    file_names = []
    user = input("Use 'ls' or np.endswith(t or f)?:")

    if user == "f":
        image_path = "src/NCR_samples/" # Add your directory here
        # Use specific file instead
        # os.path.join()
        file_names = []
        for subdir, _, files in os.walk(image_path):
            for file in files:
                if file.endswith(".jpg"):
                    file_names.append(file)
    
    elif user == "t":
        # Run through all the user-inputted files to ensure proper handling of images (basis)
        # NCR generic sample retrieval
        image_path = "src/NCR_samples/"
        IMAGE_REGEX = r'[a-zA-Z0-9\-]*.(?:jpg|png)'
        logger.debug(f"IMAGE_REGEX: {IMAGE_REGEX}\n")
        files = subprocess.check_output(["ls", image_path]).decode("utf-8")
        file_names = re.findall(IMAGE_REGEX, files)
        joined = "\n".join(file_names)
        logger.debug(f"File imported:\n{joined}\n")

    images = []
    for name in file_names:
        if os.path.exists(image_path + name):
            images.append(cv2.imread(f"{image_path}{name}"))
        else:
            logger.warning(f"{name} not found in NCR_samples... skipping")
    
    for i in range(len(file_names)):
        logger.debug("Begining segmentation")
        # preprocessed = preprocessing.preprocessImage(images[i])
        # segmented = segmentate_image(images[i], output_dir)

        # FOR DEVELOPMENT TESTING
        logger.debug(f"Segmenting image {file_names[i]}")
        # segmented = test_line_segmentation(file_names[i], output_dir)
        # segmented = test_word_segmentation(file_names[i], output_dir)
        segmented = test_character_segmentation(file_names[i], output_dir)
    
    logger.info("Complete segmentation module")
