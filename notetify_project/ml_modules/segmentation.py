import cv2
from cv2.typing import MatLike

import numpy as np
import os
import sys
import subprocess
import re

# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
# sys.path.insert(0, project_root)
# os.chdir(project_root)

from .preprocessing import preprocessImage
from .config.segmentation_config import segmentation_config
from .logger import logger, log_execution_time

def resize_and_center_char(img, output_size=(28, 28)):
    h, w = img.shape
    target_h, target_w = output_size

    # Compute scaling factor to fit character into the output size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create white square canvas
    canvas = np.zeros(output_size, dtype=np.uint8) * 255

    # Compute center placement
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2

    # Paste resized character into the center
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    return canvas

def add_square_padding(img: np.ndarray) -> np.ndarray:
    height, width = img.shape
    if height == width:
        return img

    # Determine padding size
    if height > width:
        padding = (height - width) // 2
        return np.pad(img, ((0, 0), (padding, height - width - padding)), mode='constant')
    else:
        padding = (width - height) // 2
        return np.pad(img, ((padding, width - height - padding), (0, 0)), mode='constant')

def trim_and_resize(line_img, target_height=32):
    # Trim top/bottom whitespace
    row_sums = np.sum(line_img, axis=1)
    non_empty = np.where(row_sums > 0)[0]
    if non_empty.size == 0:
        return None  # skip empty lines
    top, bottom = non_empty[0], non_empty[-1] + 1
    trimmed = line_img[top:bottom, :]
    
    # Resize height
    h, w = trimmed.shape
    scale = target_height / h
    resized = cv2.resize(trimmed, (int(w * scale), target_height), interpolation=cv2.INTER_NEAREST)
    return resized

def flipImage(input: MatLike) -> MatLike:
    ''' Inverse of inputted image '''
    try:
        flipped = cv2.bitwise_not(input)
    except cv2.error as e:
        logger.error(f"Error: {e}")
        return input
    
    return flipped

def add_padding(img: MatLike, padding, axis=0):
    """ 
    Add symmetric padding to an image along the specified axis.
    """
    pad_shape = list (img.shape)
    pad_shape[axis] = padding
    pad_block = np.zeros(pad_shape, dtype = img.dtype)
    return np.concatenate([pad_block, img, pad_block], axis=axis)

@log_execution_time
def segment_lines(image: MatLike, 
                  line_gap_factor=segmentation_config.LINE_GAP_FACTOR, 
                  text_thresh=segmentation_config.TEXT_LINE_THRESHOLD, 
                  line_padding=segmentation_config.HEIGHT_CHAR_BUFFER) -> list:
    """
    Segment lines fro ma binary image using horizontal projection and dynamic gap thresholding
    """
    try:

        # Horizontal projection (sum along columns → shape = height,)
        projection = np.sum(image, axis=1)

        # Find rows with any ink
        ink_threshold = np.max(projection) * text_thresh
        text_rows = np.where(projection > ink_threshold)[0]
        if len(text_rows) == 0:
            logger.warning("No text lines detected.")
            return []

        # Detect continuous blocks of rows (start and end of line regions)
        boundaries = []
        start = text_rows[0]

        for i in range(1, len(text_rows)):
            if text_rows[i] != text_rows[i - 1] + 1:
                end = text_rows[i - 1]
                boundaries.append((start, end))
                start = text_rows[i]
        boundaries.append((start, text_rows[-1]))

        # Compute gaps between lines
        gaps = [boundaries[i + 1][0] - boundaries[i][1] for i in range(len(boundaries) - 1)]
        if not gaps:
            return [add_padding(image[start:end+1, :], line_padding, axis=0) for (start, end) in boundaries]

        #median_gap = np.median(gaps)
        gap_stat = np.percentile(gaps, 40)
        line_gap_thresh = gap_stat * line_gap_factor
        # logger.warning(f"Threshold line gap: {line_gap_thresh}")

        # Merge line blocks if the gap is smaller than the threshold
        merged_boundaries = []
        current_start = boundaries[0][0]

        for i in range(len(gaps)):
            if gaps[i] > line_gap_thresh:
                current_end = boundaries[i][1]
                merged_boundaries.append((current_start, current_end))
                current_start = boundaries[i + 1][0]

        merged_boundaries.append((current_start, boundaries[-1][1]))

        # Crop the line images and apply vertical padding
        line_images = []
        h, w = image.shape

        for (start, end) in merged_boundaries:
            padded_start = max(0, start - line_padding)
            padded_end = min(h, end + line_padding)
            line_crop = image[padded_start:padded_end, :]
            line_images.append(line_crop)

        return line_images

    except Exception as e:
        logger.error(f"Error in line segmentation: {e}")
        return []
    
@log_execution_time
def segment_words(line_image: MatLike, 
                  threshold_factor=segmentation_config.WORD_GAP_FACTOR, 
                  WIDTH_BUFFER=segmentation_config.WIDTH_CHAR_BUFFER) -> list:
    """
    Segment a line image into individual words based on vertical projection.
    """
    try:

        if segmentation_config.MIN_WORD_IMG_HEIGHT > line_image.shape[0]:
            logger.warning(f"Word to small to obtain image")
            return []

        # Vertical Projection
        projection = np.sum(line_image, axis=0)

        # Find where text is present
        text_columns = np.where(projection > 0)[0]
        if len(text_columns) == 0:
            logger.warning("No text found in line.")
            return []
        
        # Group continuos non-zero regions (start-end of character/word blobs)
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
        gap_stat = np.percentile(gaps, 50)
        word_gap_thresh = gap_stat * threshold_factor
        # logger.warning(f"Word Gap Threshold: {word_gap_thresh}")

        # Segment words
        words = []
        current_word_start = boundaries[0][0]

        for i in range(len(gaps)):
            if gaps[i] > word_gap_thresh:
                current_word_end = boundaries[i][1]
                words.append(add_padding(line_image[:, current_word_start:current_word_end + 1], WIDTH_BUFFER, axis=1))
                current_word_start = boundaries[i + 1][0]

        # Append the last word
        words.append(add_padding(line_image[:, current_word_start:boundaries[-1][1] + 1], WIDTH_BUFFER, axis=1))

        return words
    except Exception as e:
        logger.error(f"Error in word segmentation: {e}")
        return []

@log_execution_time
def segment_characters(word_image: MatLike, 
                       WIDTH_BUFFER=segmentation_config.WIDTH_CHAR_BUFFER, 
                       HEIGHT_INF=segmentation_config.HEIGHT_INFLUENCE,
                       proximity_thresh=segmentation_config.PROXIMITY_THRESHOLD) -> list:
    """
    Segment a word image into individual characters using contour detection.
    """
    try:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(word_image, kernel, iterations=1)
        
        # Find contours of characters
        cnts = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        if len(cnts) == 0:
            return []
        
        logger.debug(f"Detected {len(cnts)} contours.")
        boxes = [cv2.boundingRect(c) for c in cnts]
        merged = []
        used = [False] * len(cnts)

        for i in range(len(cnts)):
            if used[i]:
                continue
            x, y, w, h = boxes[i]
            current = [i]
            used[i] = True

            for j in range(i + 1, len(cnts)):
                if used[j]:
                    continue
                x2, y2, w2, h2 = boxes[j]
                if abs((x + w / 2) -  (x2 + w2 / 2)) < proximity_thresh and abs((y + h / 2) - (y2 + h2 / 2)) < proximity_thresh:
                    current.append(j)
                    used[j] = True

            merged_pts = np.vstack([cnts[k] for k in current])
            merged.append(cv2.convexHull(merged_pts))
 
        logger.debug(f"Detected {len(merged)} contours after merging.")

        cnts = sorted(merged, key=lambda ctr: cv2.boundingRect(ctr)[0])
        characters_images = []
        for i, c in enumerate(cnts):
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            
            # Filter out small contours based on area and aspect ratio
            if area < word_image.shape[0] * HEIGHT_INF:
                continue

            # char_resized = add_padding(dilated[0:word_image.shape[0], x:x + w], WIDTH_BUFFER, axis=1)
            # h, w = char_resized.shape
            # diff = abs(h - w)
            # if h > w:
            #     char_resized = add_padding(char_resized, diff // 2, axis=1)
            # else:
            #     char_resized = add_padding(char_resized, diff // 2, axis=0)

            # char_resized = cv2.resize(char_resized, segmentation_config.IMAGE_DIMS, interpolation=cv2.INTER_AREA)
            # characters_images.append(char_resized.astype(np.float32) / 255.0)

            
            # char_resized = cv2.resize(add_padding(word_image[0:word_image.shape[0], x:x + w], WIDTH_BUFFER, axis=1), segmentation_config.IMAGE_DIMS, interpolation=cv2.INTER_AREA)
            # char_crop = word_image[y:y+h, x:x+w]
            # squared = add_square_padding(char_crop)
            # resized = cv2.resize(squared, (28, 28), interpolation=cv2.INTER_AREA)
            # characters_images.append(resized)
            char_crop = word_image[y:y+h, x:x+w]
            square_char = resize_and_center_char(char_crop, output_size=(28, 28))
            # cv2.imshow("Character image", square_char)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            characters_images.append(square_char.astype(np.float32) / 255.0)
            # characters_images.append(square_char)
            
        
    except Exception as e:
        logger.error(f"Error in character segmentation: {e}")
        return []

    # for char in characters_images:
    #     cv2.imshow("char", char)
    #     cv2.waitKey(0)
    return characters_images

@log_execution_time
def segmentImage(image: MatLike) -> tuple:
    """
    Segmentation pipeline to break down images into lines, words, and characters. (Metadata + character images)
    """
    segmented_images = []
    segmented_metadata = []

    # Line segmentation
    line_images = segment_lines(image)

    # Word Segmentation
    for line_idx, line_img in enumerate(line_images):
        # cv2.imshow("Line", line_img)
        # cv2.waitKey(0)
        word_images = segment_words(line_img)

        # Character Segmentation
        for word_idx, word_img in enumerate(word_images):
            # cv2.imshow("Word", word_img)
            # cv2.waitKey(0)
            char_images = segment_characters(word_img)

            # Text Hierarchy Preserving
            for char_idx, char_img in enumerate(char_images):
                # logger.debug(f"Added another char to word: {word_idx}")
                # cv2.imshow(f"Character {char_idx + 1}", char_img)
                # cv2.waitKey(0)
                segmented_images.append(char_img)
                segmented_metadata.append({
                    'line_idx': line_idx,
                    'word_idx': word_idx,
                    'char_idx': char_idx,
                    'image_idx': len(segmented_images) - 1 # index into char_images
                })

    # logger.debug(f"Segmented Data: {segmented_metadata}")

    # Reshape images to fit tensorflow input
    segmented_images = np.reshape(segmented_images, (len(segmented_images),28, 28, 1))
    return segmented_images, segmented_metadata

# TESTING ONLY
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

def test_line_segmentation(image: str, output_dir: str) -> None:
    """
    Test the word segmentation function on a sample image of lines
    """
    logger.debug("Testing line segmentation alone")
    img = cv2.imread(f"./src/NCR_samples/{image}", cv2.IMREAD_GRAYSCALE)
    if img is None:
        logger.error(f"Image not found at {image}")
        return
    
    binary = cv2.threshold(img, 0, segmentation_config.MAX_VALUE, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    lines = segment_lines(binary)
    processed_lines = []
    for line in segment_lines(binary):
        processed = trim_and_resize(line)
        if processed is not None:
            processed_lines.append(processed)
    logger.debug(f"Detected {len(processed_lines)} lines.")

    if len(processed_lines) == 1:
        save_images(processed_lines, os.path.join(output_dir, f"image_{image}"), "line")
    else:
        save_images(processed_lines, os.path.join(output_dir, f"image_{image}"), "line" )

    words = [segment_words(line) for line in lines]
    if len(words) != 0:
        for word_idx, word in enumerate(words):
            for img in word:
                pass
            # save_images(word, os.path.join(output_dir, f"line_{image}_word_{word_idx}"), "word")

    characters = []
    for line_idx, word_line in enumerate(words):
        for word_idx, word in enumerate(word_line):
            # logger.warning(f"Words: {words}")
            characters.append(segment_characters(word))

            for char_idx, chars in enumerate(characters):
                for char in chars:
                    pass
                    # cv2.imshow("Char", char)
                    # cv2.waitKey(0)
                # save_images(chars, os.path.join(output_dir, f"line_{line_idx}_word_{word_idx}_char{char_idx}"), "char")

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
        print(name)
        if os.path.exists(image_path + name):
            images.append(cv2.imread(f"{image_path}{name}", cv2.IMREAD_GRAYSCALE))
        else:
            logger.warning(f"{name} not found in NCR_samples... skipping")
        
    for i in range(len(file_names)):
        preprocessed = preprocessImage(cv2.imread(f"{image_path}{file_names[i]}"))
        logger.debug(f"Beginning segmentation on {file_names[i]}")
        # cv2.imshow("Preprocessed", preprocessed)
        # cv2.waitKey(0)
        # binary = cv2.threshold(images[i], 0, segmentation_config.MAX_VALUE, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        segmented_images, segmented_metadata = segmentImage(preprocessed) # Should return a list of dict elements that hold the metadata and image of characters

        for j, segment in enumerate(segmented_images):
            char_img = segment["image"] if isinstance(segment, dict) else segment  # support for dict or just array
            cv2.imshow(f"Character {j+1}", char_img)
            cv2.waitKey(0)  # Wait for key press
            cv2.destroyAllWindows()

        import json
        file_path = f"{output_dir}/{i}image.json"
        with open(file_path, 'w') as json_file:
            json.dump(segmented_metadata, json_file)

        # FOR DEVELOPMENT TESTING
        # logger.debug(f"Test segmenting image {file_names[i]}")
        # segmented = test_line_segmentation(file_names[i], output_dir)
        # segmented = test_word_segmentation(file_names[i], output_dir)
        segmented = test_character_segmentation(file_names[i], output_dir)
    
    logger.info("Complete segmentation module")