import cv2
from cv2. typing import MatLike

import numpy as np
import os
import sys

import subprocess
import re

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)
os.chdir(project_root)

from src.config.segmentation_config import segmentation_config
from src.modules.logger import logger, log_execution_time

@log_execution_time
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

@log_execution_time
def segment_words(line_image: MatLike, min_gap=10, threshold_factor=1.5):
    """
    Segment a line image into individual words based on vertical projection.
    """
    try:
        flipped = cv2.bitwise_not(line_image)

        # Vertical Projection
        projection = np.sum(flipped, axis=0)

        # Find where text is present
        text_columns = np.where(projection > 0)[0]
        if len(text_columns) == 0:
            logger.warning("No text found in line.")
            return []
        
        # Group continous non-zero regions (start-end of character/word blobs)
        boundaries = []
        start = text_columns[0]

        for i in range(1, len(text_columns)):
            if text_columns[i] != text_columns[i - 1] + 1:
                end = text_columns[i - 1]
                boundaries.append((start, end))
                start = text_columns[i]
        boundaries.append((start, text_columns[-1]))

        # Compute gaps between blobs
        gaps = [boundaries[i + 1][0] - boundaries[i][1] for i in range(len(boundaries) - 1)]
        if not gaps:
            return [line_image[:, b[0]:b[1]+1] for b in boundaries]
        
        # Estimate dynamic threshold for word separation
        median_gap = np.median(gaps)
        word_gap_thresh = median_gap * threshold_factor

        # Segment words
        words = []
        current_word_start = boundaries[0][0]

        for i in range(len(gaps)):
            if gaps[i] > word_gap_thresh:
                current_word_end = boundaries[i][1]
                words.append(line_image[:, max(0, current_word_start - segmentation_config.WIDTH_CHAR_BUFFER):min(line_image.shape[1], current_word_end + 1 + segmentation_config.WIDTH_CHAR_BUFFER)])
                current_word_start = boundaries[i + 1][0]

        # Append the last word
        words.append(line_image[:, current_word_start:boundaries[-1][1] + 1])

        return words
    except Exception as e:
        logger.error(f"Error in word segmentation: {e}")
        return []

@log_execution_time
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


def segmentate_image(image: MatLike, output_dir: str):
    """
    Complete processing pipeline for an image: preprocess, segment lines, segment words.
    """
    logger.info("Starting segmentation pipeline.")
    # Segment lines
    binary = cv2.threshold(image, 0, segmentation_config.MAX_VALUE, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    lines = segment_lines(binary)
    logger.debug(f"Detected {len(lines)} lines.")
    save_images(lines, os.path.join(output_dir, "lines"), "line")

    # Segment words within each line
    for i, line in enumerate(lines):
        words = segment_words(line)
        logger.debug(f"Line {i + 1}: Detected {len(words)} words.")
        save_images(words, os.path.join(output_dir, f"line_{i + 1}_words"), "word")

    logger.info("Segmentation pipeline complete.")

@log_execution_time
def segmentImage(image: MatLike, model=None) -> tuple:
    """
    Segmentation pipeline to break down iamges into lines, words, and characters. (Metadata + character images)
    """
    segmented_images = []
    segmented_metadata = []

    # Line segmentation
    # line_images = segment_lines(image)
    line_images = image
    line_idx = 0

    # Word Segmentation
    for line_idx, line_img in enumerate(range(1)):#in enumerate(line_images):
        word_images = segment_words(line_images)

        # Character Segmentation
        for word_idx, word_img in enumerate(word_images):
            char_images = segment_characters(word_img)

            # Run character recognition here since post_processing is done already
            predicted_characters = []
            ## THIS METHOD IS JUST HERE TO KEEP TRACK OF IT FOR THE FUTURE (NOT USED)
            if model is not None: 
                predictions = model.predict(char_images)

                if len(predictions) < len(word_img):
                    logger.error("Failed to obtain predictions for all characters")
                    return []
                for predict_char in predictions:
                    if predict_char[1] > 0.5: # Confidence saved in model
                        predicted_characters.append(predict_char[0])
                    else:
                        predicted_characters.append(' ')

                # Text Hierarchy Preserving
                for char_idx, char in enumerate(predicted_characters):
                    segmented_metadata.append({
                        'line_idx': line_idx,
                        'word_idx': word_idx,
                        'char_idx': char_idx,
                        'char': char
                    })
            else:
                # Text Hierarchy Preserving
                for char_idx, char_img in enumerate(char_images):
                    # logger.debug(f"Added another char to word: {word_idx}")
                    segmented_images.append(char_img)
                    segmented_metadata.append({
                        'line_idx': line_idx,
                        'word_idx': word_idx,
                        'char_idx': char_idx,
                        'image_idx': len(char_images) - 1 # index into char_images
                    })

    logger.debug(f"Segmented Data: {segmented_metadata}")
    return char_images, segmented_metadata

def save_images(images, folder, prefix):
    """
    Save a list of images to the specified folder with a given prefix.
    """
    os.makedirs(folder, exist_ok=True)
    for i, img in enumerate(images):
        cv2.imwrite(f"{folder}/{prefix}_{i + 1}.png", img)

def test_character_segmentation(word: str, output_dir: str) -> None:
    """
    Test the character segmentation function on a sample image of words.
    """
    logger.debug("Testing character segmentation alone")
    image = cv2.imread(f"./src/NCR_samples/{word}", cv2.IMREAD_GRAYSCALE)
    if image is None:
        logger.error(f"Image not found at {word}")
        return
    
    binary = cv2.threshold(image, 0, segmentation_config.MAX_VALUE, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    characters = segment_characters(binary)
    logger.debug(f"Detected {len(characters)} characters.")

    if len(characters) == 1:
        save_images(characters, os.path.join(output_dir, f"words_{word}"), "character")
    else:
        save_images(characters, os.path.join(output_dir, f"words_{word}"), "character" )
    logger.debug("Character segmentation test complete.")

def test_word_segmentation(line: str, output_dir: str) -> None:
    """
    Test the word segmentation function on a sample image of lines
    """
    logger.debug("Testing word segmentation alone")
    image = cv2.imread(f"./src/NCR_samples/{line}", cv2.IMREAD_GRAYSCALE)
    if image is None:
        logger.error(f"Image not found at {line}")
        return
    
    binary = cv2.threshold(image, 0, segmentation_config.MAX_VALUE, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    words = segment_words(binary)
    logger.debug(f"Detected {len(words)} words.")

    if len(words) == 1:
        save_images(words, os.path.join(output_dir, f"line_{line}"), "word")
    else:
        save_images(words, os.path.join(output_dir, f"line_{line}"), "word" )

    characters = [segment_characters(word) for word in words]

    for i, chars in enumerate(characters):
        save_images(chars, os.path.join(output_dir, f"line_{line}_word_{i}"), "character")

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
    
    binary = cv2.threshold(image, 0, segmentation_config.MAX_VALUE, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    characters = segment_lines(binary)
    logger.debug(f"Detected {len(characters)} lines.")

    if len(characters) == 1:
        save_images(characters, os.path.join(output_dir, f"image_{word}"), "lines")
    else:
        save_images(characters, os.path.join(output_dir, f"image_{word}"), "lines" )
    logger.debug("Line segmentation test complete.")

if __name__ == "__main__":
    logger.info("Starting segmentation module")

    output_dir = "segmented_output"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

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
            images.append(cv2.imread(f"{image_path}{name}", cv2.IMREAD_GRAYSCALE))
        else:
            logger.warning(f"{name} not found in NCR_samples... skipping")
        
    for i in range(len(file_names)):
        logger.debug(f"Beginning segmentation on {file_names[i]}")
        binary = cv2.threshold(images[i], 0, segmentation_config.MAX_VALUE, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        segmented_images, segmented_metadata = segmentImage(binary) # Should return a list of dict elements that hold the metadata and image of characters

        import json
        file_path = f"{output_dir}/{i}image.json"
        with open(file_path, 'w') as json_file:
            json.dump(segmented_metadata, json_file)
        # FOR DEVELOPMENT TESTING
        # logger.debug(f"Test segmenting image {file_names[i]}")
        # segmented = test_line_segmentation(file_names[i], output_dir)
        # segmented = test_word_segmentation(file_names[i], output_dir)
        # segmented = test_character_segmentation(file_names[i], output_dir)
    
    logger.info("Complete segmentation module")
