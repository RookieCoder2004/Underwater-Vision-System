import cv2
from condition import classify_condition
from enhance import enhance_by_condition

img = cv2.imread("images/test.jpg")

if img is None:
    print("Error: Image not found")
    exit()

condition = classify_condition(img)
print("Condition:", condition)

enhanced = enhance_by_condition(img, condition)

cv2.imshow("Original", img)
cv2.imshow("Enhanced", enhanced)

cv2.waitKey(0)
cv2.destroyAllWindows()