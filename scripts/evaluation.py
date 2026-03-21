import cv2
from simulate import simulate_blur, simulate_low_light
from condition import classify_condition

img = cv2.imread("images/test.jpg")

# Conditions
clear_cond = classify_condition(img)
blur_cond = classify_condition(simulate_blur(img))
dark_cond = classify_condition(simulate_low_light(img))

print("\n--- Evaluation ---")
print(f"Clear Image Condition: {clear_cond}")
print(f"Blur Image Condition: {blur_cond}")
print(f"Low Light Condition: {dark_cond}")