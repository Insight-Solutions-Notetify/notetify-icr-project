import cv2
import pytesseract
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from typing import List
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)
os.chdir(project_root)
from src.config.preprocess_config import preprocess_config

def process_images(image_paths: List[str]):
    for image_path in image_paths:
        print(f"Processing: {image_path}")
        img = cv2.imread(image_path)
        if img is None:
            print(f"Image not found: {image_path}")
            continue
        
        processed_img = preprocess_image(img)
        ocr_result, recognized_text = apply_tesseract(processed_img)
        
        display_results(img, processed_img, ocr_result, recognized_text)

def preprocess_image(input_img):
    """ Iterates through preprocessing functions """
    weighted = contrast_image(input_img)
    scaled = rescale_image(weighted)
    blurred = blur_image(scaled)
    note = highlight_boundary(blurred)
    text_only = highlight_text(note)
    return text_only

def apply_tesseract(input_img):
    """ Perform OCR using Tesseract """
    gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    recognized_text = pytesseract.image_to_string(gray_img)
    boxes = pytesseract.image_to_boxes(gray_img)
    h, w = input_img.shape[:2]
    for b in boxes.splitlines():
        b = b.split()
        x1, y1, x2, y2 = map(int, b[1:5])
        cv2.rectangle(input_img, (x1, h - y1), (x2, h - y2), (0, 255, 0), 2)
    return input_img, recognized_text

def display_results(original, processed, ocr_image, text):
    """ Display results using Matplotlib """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image")
    
    axes[1].imshow(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Processed Image")
    
    axes[2].imshow(cv2.cvtColor(ocr_image, cv2.COLOR_BGR2RGB))
    axes[2].set_title("OCR Result")
    
    for ax in axes:
        ax.axis("off")
    
    plt.show()
    print("Recognized Text:\n", text)

def contrast_image(input_img):
    return cv2.addWeighted(input_img, preprocess_config.CONTRAST, np.zeros(input_img.shape, input_img.dtype), 0, preprocess_config.BRIGHTNESS)

def rescale_image(input_img):
    img_width = input_img.shape[1]
    ratio = preprocess_config.MAX_WIDTH / img_width if img_width > preprocess_config.MAX_WIDTH else 1
    return cv2.resize(input_img, (0, 0), fx=ratio, fy=ratio)

def blur_image(input_img):
    return cv2.GaussianBlur(input_img, (preprocess_config.KERNEL_DIMS, preprocess_config.KERNEL_DIMS), preprocess_config.GAUSSIAN_SIGMA)

def highlight_boundary(input_img):
    flipped = 255 - input_img
    gray = cv2.cvtColor(flipped, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(gray, contours, -1, (100, 100, 100), 10)
    return 255 - cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def highlight_text(input_img):
    gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

if __name__ == "__main__":
    image_folder = "src/images/"
    image_files = [os.path.join(image_folder, file) for file in os.listdir(image_folder) if file.endswith(".jpg")]
    process_images(image_files)
