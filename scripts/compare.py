import os
import sys
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# FIX PATH (IMPORTANT)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# IMPORTS
from scripts.condition import classify_condition
from scripts.enhance import enhance_by_condition

# LOAD MODEL
model = YOLO("models/best.pt")

# IMAGE FOLDER
image_folder = "images"

original_counts = []
enhanced_counts = []
image_names = []

for filename in os.listdir(image_folder):
    if filename.endswith((".jpg", ".png", ".jpeg")):

        path = os.path.join(image_folder, filename)
        img = cv2.imread(path)

        if img is None:
            continue

        print(f"Processing: {filename}")

        # 🔹 Detection on ORIGINAL image
        results_orig = model(img)
        count_orig = len(results_orig[0].boxes)

        # 🔹 Condition + Enhancement
        condition = classify_condition(img)
        enhanced = enhance_by_condition(img, condition)

        # 🔹 Detection on ENHANCED image
        results_enh = model(enhanced)
        count_enh = len(results_enh[0].boxes)

        # Store results
        original_counts.append(count_orig)
        enhanced_counts.append(count_enh)
        image_names.append(filename)

# 📊 GRAPH (SIDE-BY-SIDE)
x = range(len(image_names))

plt.figure()

plt.bar([i - 0.2 for i in x], original_counts, width=0.4, label="Original")
plt.bar([i + 0.2 for i in x], enhanced_counts, width=0.4, label="Enhanced")

plt.xticks(x, image_names, rotation=45)
plt.xlabel("Images")
plt.ylabel("Number of Detections")
plt.title("Original vs Enhanced Detection Comparison")

plt.legend()

plt.tight_layout()
plt.show()