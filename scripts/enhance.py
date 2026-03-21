import cv2

def enhance_image(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)

    enhanced = cv2.merge((l, a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def sharpen(img):
    import numpy as np
    kernel = np.array([[0,-1,0],
                       [-1,5,-1],
                       [0,-1,0]])
    return cv2.filter2D(img, -1, kernel)


def enhance_by_condition(img, condition):
    if condition == "low_light":
        return enhance_image(img)
    elif condition == "blurry":
        return sharpen(img)
    else:
        return img