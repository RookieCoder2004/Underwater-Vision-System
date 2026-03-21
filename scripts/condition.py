import cv2
import numpy as np

def get_brightness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

def get_blur(img):
    return cv2.Laplacian(img, cv2.CV_64F).var()

def classify_condition(img):
    brightness = get_brightness(img)
    blur = get_blur(img)

    if brightness < 50:
        return "low_light"
    elif blur < 100:
        return "blurry"
    else:
        return "clear"      
        