import cv2
import numpy as np

# Blur simulation
def simulate_blur(img):
    return cv2.GaussianBlur(img, (7,7), 0)

# Low light simulation
def simulate_low_light(img):
    return (img * 0.4).astype("uint8")

# Color shift (green tint)
def simulate_color_shift(img):
    img = img.copy()
    img[:,:,1] = np.clip(img[:,:,1] * 1.5, 0, 255)
    return img.astype("uint8")