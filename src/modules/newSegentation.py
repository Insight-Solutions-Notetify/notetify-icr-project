import cv2
import numpy as np
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FIX 1: Use a clearer name for max_value in thresholding. ---
# Instead of THRESH_BINARY_INV = 255, let's use MAX_VALUE = 255 
# to avoid confusion with cv2.THRESH_BINARY_INV.
MAX_VALUE = 255

# Constants for preprocessing and segmentation
BLUR_KERNEL_SIZE = (3, 3)
MIN_CHAR_SIZE = (5, 15)
MIN_WORD_GAP = 5
MIN_LINE_GAP = 5

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
            15,
            10
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

def segment_lines(binary_image, min_gap=MIN_LINE_GAP):
    """
    Segment the binary image into individual lines based on horizontal projection.
    """
    try:
        projection = np.sum(binary_image, axis=1)
        
        # Simple threshold = 20% of max projection
        threshold = np.max(projection) * 0.2
        
        # Avoid zero-threshold edge case
        if threshold == 0:
            logger.warning("Projection threshold is zero. Check your image or threshold logic.")
            return []
        
        line_indices = np.where(projection > threshold)[0]
        lines = []
        if len(line_indices) == 0:
            logger.warning("No text lines detected.")
            return lines

        # Group rows into line segments
        start_idx = line_indices[0]
        for i in range(1, len(line_indices)):
            if line_indices[i] - line_indices[i - 1] > min_gap:
                lines.append(binary_image[start_idx:line_indices[i], :])
                start_idx = line_indices[i]
        # Append the last line segment
        lines.append(binary_image[start_idx:, :])
        return lines
    except Exception as e:
        logger.error(f"Error in line segmentation: {e}")
        return []

def segment_words(line_image, min_gap=MIN_WORD_GAP):
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

def segment_characters(word_image, min_char_size=MIN_CHAR_SIZE):
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

def process_image(image_path, output_dir):
    """
    Complete processing pipeline for an image: preprocess, deskew, segment lines, words, and characters.
    """
    # 1. Preprocess
    binary = preprocess_image(image_path)
    if binary is None:
        logger.error("Image preprocessing failed.")
        return

    # 2. Deskew
    binary = deskew(binary)

    # 3. Line segmentation
    lines = segment_lines(binary)
    logger.info(f"Detected {len(lines)} lines.")
    save_images(lines, os.path.join(output_dir, "lines"), "line")

    # 4. Word segmentation (per line)
    for i, line in enumerate(lines):
        words = segment_words(line)
        logger.info(f"Line {i + 1}: Detected {len(words)} words.")
        save_images(words, os.path.join(output_dir, f"line_{i + 1}_words"), "word")

        # 5. Character segmentation (per word)
        for j, word in enumerate(words):
            characters = segment_characters(word)
            logger.info(f"Word {j + 1}: Detected {len(characters)} characters.")
            save_images(
                characters,
                os.path.join(output_dir, f"line_{i + 1}_word_{j + 1}_characters"),
                "char"
            )

if __name__ == "__main__":
    # File path for the input image
    image_path = "b4980b85-53cd-42bd-90f4-a1e3fd5afa78.png"
    output_dir = "segmented_output"

    process_image(image_path, output_dir)
