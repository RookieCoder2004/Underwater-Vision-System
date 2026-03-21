from ultralytics import YOLO
import cv2
from condition import classify_condition
from enhance import enhance_by_condition

# Load YOLO model (temporary default model)
model = YOLO("yolov8n.pt")

# Load image
img = cv2.imread("images/test.jpg")

if img is None:
    print("Error: Image not found")
    exit()

# STEP 1: Detect condition
condition = classify_condition(img)
print("Condition:", condition)

# STEP 2: Enhance based on condition
enhanced = enhance_by_condition(img, condition)

# STEP 3: Run YOLO detection
results = model(enhanced)

# Draw bounding boxes
output = results[0].plot()

# Show results
cv2.imshow("Original", img)
cv2.imshow("Enhanced", enhanced)
cv2.imshow("Detection", output)

cv2.waitKey(0)
cv2.destroyAllWindows()