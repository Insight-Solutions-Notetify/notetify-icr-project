import cv2
import numpy as np


def preprocess_image(image_path):

    try:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Adaptive thresholding for binary image
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 10
        )

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(binary, (3, 3), 0)
        return blurred
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None


def segment_lines(binary_image, min_gap=5):

    try:
        projection = np.sum(binary_image, axis=1)  # Sum pixel values along rows
        threshold = np.max(projection) * 0.2  # Adjust sensitivity dynamically
        line_indices = np.where(projection > threshold)[0]

        lines = []
        if len(line_indices) == 0:
            print("No text lines detected.")
            return lines

        start_idx = line_indices[0]
        for i in range(1, len(line_indices)):
            if line_indices[i] - line_indices[i - 1] > min_gap:
                lines.append(binary_image[start_idx:line_indices[i], :])
                start_idx = line_indices[i]
        lines.append(binary_image[start_idx:, :])  # Add the last line
        return lines
    except Exception as e:
        print(f"Error in line segmentation: {e}")
        return []


def segment_words(line_image, min_gap=5):

    try:
        projection = np.sum(line_image, axis=0)  # Sum pixel values along columns
        threshold = np.max(projection) * 0.1  # Adjust sensitivity dynamically
        word_indices = np.where(projection > threshold)[0]

        words = []
        if len(word_indices) == 0:
            print("No words detected in the line.")
            return words

        start_idx = word_indices[0]
        for i in range(1, len(word_indices)):
            if word_indices[i] - word_indices[i - 1] > min_gap:
                words.append(line_image[:, start_idx:word_indices[i]])
                start_idx = word_indices[i]
        words.append(line_image[:, start_idx:])  # Add the last word
        return words
    except Exception as e:
        print(f"Error in word segmentation: {e}")
        return []


def segment_characters(word_image, min_char_size=(5, 15)):

    try:
        contours, _ = cv2.findContours(word_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        characters = []

        for contour in sorted(contours, key=lambda c: cv2.boundingRect(c)[0]):  # Sort by x-coordinate
            x, y, w, h = cv2.boundingRect(contour)
            if w >= min_char_size[0] and h >= min_char_size[1]:  # Filter noise
                char = word_image[y:y + h, x:x + w]
                char_resized = cv2.resize(char, (32, 32), interpolation=cv2.INTER_AREA)  # Normalize size
                characters.append(char_resized)
        return characters
    except Exception as e:
        print(f"Error in character segmentation: {e}")
        return []


def debug_display_images(images, title="Image"):

    try:
        for i, img in enumerate(images):
            cv2.imshow(f"{title} {i + 1}", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error in displaying images: {e}")


if __name__ == "__main__":
    # File path for the input image
    image_path = "b4980b85-53cd-42bd-90f4-a1e3fd5afa78.png"

    # Step 1: Preprocess the image
    binary = preprocess_image(image_path)
    if binary is None:
        print("Image preprocessing failed.")
        exit()

    # Step 2: Segment lines
    lines = segment_lines(binary)
    print(f"Detected {len(lines)} lines.")
    debug_display_images(lines, "Line")

    # Step 3: Segment words and characters
    for i, line in enumerate(lines):
        words = segment_words(line)
        print(f"Line {i + 1}: Detected {len(words)} words.")
        debug_display_images(words, f"Words in Line {i + 1}")

        for j, word in enumerate(words):
            characters = segment_characters(word)
            print(f"Word {j + 1}: Detected {len(characters)} characters.")
            debug_display_images(characters, f"Characters in Word {j + 1}")
