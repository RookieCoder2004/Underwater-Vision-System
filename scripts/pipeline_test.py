import cv2
from simulate import simulate_blur, simulate_low_light
from condition import classify_condition
from enhance import enhance_by_condition

# Load image
img = cv2.imread("images/test.jpg")

if img is None:
    print("Error: Image not found")
    exit()

# ---- ORIGINAL ----
condition_orig = classify_condition(img)
enhanced_orig = enhance_by_condition(img, condition_orig)

# ---- BLUR SIMULATION ----
blur_img = simulate_blur(img)
condition_blur = classify_condition(blur_img)
enhanced_blur = enhance_by_condition(blur_img, condition_blur)

# ---- LOW LIGHT SIMULATION ----
dark_img = simulate_low_light(img)
condition_dark = classify_condition(dark_img)
enhanced_dark = enhance_by_condition(dark_img, condition_dark)

# Show results
cv2.imshow("Original", img)
cv2.imshow("Enhanced Original", enhanced_orig)

cv2.imshow("Blur", blur_img)
cv2.imshow("Enhanced Blur", enhanced_blur)

cv2.imshow("Low Light", dark_img)
cv2.imshow("Enhanced Low Light", enhanced_dark)

print("Original Condition:", condition_orig)
print("Blur Condition:", condition_blur)
print("Low Light Condition:", condition_dark)

cv2.waitKey(0)
cv2.destroyAllWindows()