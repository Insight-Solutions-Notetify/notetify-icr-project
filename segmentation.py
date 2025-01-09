import cv2
import numpy as np


def preprocess_image(image_path):
    """
    Preprocess the input image by converting to grayscale, applying thresholding, and noise reduction.
    """
    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding for better handling of uneven lighting
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Remove noise using Gaussian blur
    blurred = cv2.GaussianBlur(binary, (5, 5), 0)
    return blurred


def segment_lines(binary_image, min_gap=5):
    """
    Segment the binary image into lines using horizontal projection.
    """
    projection = np.sum(binary_image, axis=1)  # Sum pixel values along rows
    threshold = np.max(projection) * 0.2  # Dynamic threshold for text regions
    line_indices = np.where(projection > threshold)[0]

    lines = []
    start_idx = line_indices[0]

    for i in range(1, len(line_indices)):
        if line_indices[i] - line_indices[i - 1] > min_gap:  # Detect gap between lines
            lines.append(binary_image[start_idx:line_indices[i], :])
            start_idx = line_indices[i]
    lines.append(binary_image[start_idx:, :])  # Add the last line
    return lines


def segment_words(line_image, min_gap=5):
    """
    Segment a line image into words using vertical projection.
    """
    projection = np.sum(line_image, axis=0)  # Sum pixel values along columns
    threshold = np.max(projection) * 0.1  # Dynamic threshold for gaps
    word_indices = np.where(projection > threshold)[0]

    words = []
    start_idx = word_indices[0]

    for i in range(1, len(word_indices)):
        if word_indices[i] - word_indices[i - 1] > min_gap:  # Detect gap between words
            words.append(line_image[:, start_idx:word_indices[i]])
            start_idx = word_indices[i]
    words.append(line_image[:, start_idx:])  # Add the last word
    return words


def segment_characters(word_image):
    """
    Segment a word image into characters using connected components analysis.
    """
    # Find contours for character segmentation
    contours, _ = cv2.findContours(word_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    characters = []

    for contour in sorted(contours, key=lambda c: cv2.boundingRect(c)[0]):  # Sort by x-coordinate
        x, y, w, h = cv2.boundingRect(contour)
        if w > 2 and h > 10:  # Filter out noise based on size
            char = word_image[y:y + h, x:x + w]
            char_resized = cv2.resize(char, (32, 32), interpolation=cv2.INTER_AREA)  # Normalize size
            characters.append(char_resized)
    return characters


def debug_show_images(images, title="Image"):
    """
    Display multiple images for debugging purposes.
    """
    for i, img in enumerate(images):
        cv2.imshow(f"{title} {i + 1}", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Example Usage
if __name__ == "__main__":
    # Step 1: Preprocess the image
    binary = preprocess_image("b4980b85-53cd-42bd-90f4-a1e3fd5afa78.png")

    # Step 2: Segment lines
    lines = segment_lines(binary)
    print(f"Detected {len(lines)} lines.")

    # Optional: Debug lines
    debug_show_images(lines, "Line")

    # Step 3: Segment words and characters
    for i, line in enumerate(lines):
        words = segment_words(line)
        print(f"Line {i + 1}: Detected {len(words)} words.")

        # Optional: Debug words
        debug_show_images(words, f"Words in Line {i + 1}")

        for j, word in enumerate(words):
            characters = segment_characters(word)
            print(f"Word {j + 1}: Detected {len(characters)} characters.")

            # Optional: Debug characters
            debug_show_images(characters, f"Characters in Word {j + 1}")
