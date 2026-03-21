import cv2
from simulate import simulate_blur, simulate_low_light, simulate_color_shift

img = cv2.imread("images/test.jpg")

if img is None:
    print("Error: Image not found")
    exit()

blur = simulate_blur(img)
dark = simulate_low_light(img)
green = simulate_color_shift(img)

cv2.imshow("Original", img)
cv2.imshow("Blur", blur)
cv2.imshow("Low Light", dark)
cv2.imshow("Green Tint", green)

cv2.waitKey(0)
cv2.destroyAllWindows()