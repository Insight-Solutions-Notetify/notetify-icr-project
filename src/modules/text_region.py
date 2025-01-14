import cv2
import numpy as np

# Load image, convert to HSV format, define lower/upper ranges, and perform
# color segmentation to create a binary mask
image = cv2.imread('black_sampel.jpg')

shades = 5

# convert to gray scale as float in range 0 to 1
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_image = gray_image.astype(np.float32)/255

# Quantize and convert back to 0 to 255 as 8-bits
result = 255 * np.floor(gray_image * shades + 0.5) / shades
result = result.clip(0, 255).astype(np.uint8)

# Gaussian Blur
kernel_dimension = 3
sigma = 0

# Gaussian blur to remove any noise from the picture
gaussian = cv2.GaussianBlur(result, (kernel_dimension, kernel_dimension), sigma)
gaussian = 255 - gaussian

# Save result as shaded.jpg so we can convert to hsv
cv2.imwrite('shaded.jpg', gaussian)
shaded = cv2.imread('shaded.jpg')
hsv = cv2.cvtColor(shaded, cv2.COLOR_BGR2HSV)
lower = np.array([0, 0, 100])
upper = np.array([0, 0, 255])
mask = cv2.inRange(hsv, lower, upper)

# Create horizontal kernel and dilate to connect text characters
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,3))
dilate = cv2.dilate(mask, kernel, iterations=4)

# Find contours and filter using aspect ratio
# Remove non-text contours by filling in the contour
cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    ar = w / float(h)
    if ar < 5:
        cv2.drawContours(dilate, [c], -1, (0,0,0), -1)

# Bitwise dilated image with mask, invert, then OCR
final = 255 - cv2.bitwise_and(dilate, mask)

cv2.imshow('shaded', result)
cv2.waitKey(0)
cv2.imshow("gaussian blur", gaussian)
cv2.waitKey(0)
cv2.imshow('mask', mask)
cv2.imshow('dilate', dilate)
cv2.imshow('result', final)
cv2.waitKey(0)