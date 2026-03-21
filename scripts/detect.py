from ultralytics import YOLO
import cv2
import os
from condition import classify_condition
from enhance import enhance_by_condition

model = YOLO("models/best.pt")

image_folder = "images"

for filename in os.listdir(image_folder):
    if filename.endswith((".jpg", ".png", ".jpeg")):

        path = os.path.join(image_folder, filename)
        img = cv2.imread(path)

        if img is None:
            continue

        print(f"\nProcessing: {filename}")

        # STEP 1: Condition detection
        condition = classify_condition(img)
        print("Condition:", condition)

        # STEP 2: Enhancement
        enhanced = enhance_by_condition(img, condition)

        # STEP 3: Detection
        results = model(enhanced)
        output = results[0].plot()

        # ADD CONDITION TEXT ON IMAGE
        cv2.putText(
            output,
            f"Condition: {condition}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # SAVE OUTPUT (with condition name)
        cv2.imwrite(f"outputs/{condition}_{filename}", output)

        # SHOW ALL STAGES
        cv2.imshow("Original Image", img)
        cv2.imshow("Enhanced Image", enhanced)
        cv2.imshow("Detection Output", output)

        cv2.waitKey(0)

cv2.destroyAllWindows()