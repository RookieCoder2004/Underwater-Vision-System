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

        # 🟢 STEP 4: POLLUTION SCORING

        boxes = results[0].boxes

        # Count objects
        count = len(boxes)

        # Total bounding box area
        total_area = 0
        for box in boxes.xyxy:
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)
            total_area += area

        # Normalize by image size
        img_area = img.shape[0] * img.shape[1]
        coverage = total_area / img_area if img_area > 0 else 0

        # Average confidence
        avg_conf = float(boxes.conf.mean()) if len(boxes) > 0 else 0

        # Final pollution score
        score = count * 2 + coverage * 100 + avg_conf * 10

        # Pollution level
        if score > 50:
            level = "HIGH"
        elif score > 25:
            level = "MEDIUM"
        else:
            level = "LOW"

        print(f"Pollution Score: {score:.2f}")
        print(f"Pollution Level: {level}")

        # 🟢 ADD TEXT ON IMAGE

        cv2.putText(
            output,
            f"Condition: {condition}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            output,
            f"Pollution: {level}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        # SAVE OUTPUT
        os.makedirs("outputs", exist_ok=True)
        cv2.imwrite(f"outputs/{condition}_{filename}", output)

        # SHOW ALL STAGES
        cv2.imshow("Original Image", img)
        cv2.imshow("Enhanced Image", enhanced)
        cv2.imshow("Detection Output", output)

        cv2.waitKey(0)

cv2.destroyAllWindows()