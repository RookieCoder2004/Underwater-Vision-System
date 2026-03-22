import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.condition import classify_condition
from scripts.enhance import enhance_by_condition

model = YOLO("models/best.pt")
image_folder = "images"

orig_conf = []
enh_conf = []
names = []

for filename in os.listdir(image_folder):
    if filename.endswith((".jpg", ".png", ".jpeg")):

        path = os.path.join(image_folder, filename)
        img = cv2.imread(path)

        if img is None:
            continue

        # ORIGINAL
        res_o = model(img)
        conf_o = np.mean(res_o[0].boxes.conf.cpu().numpy()) if len(res_o[0].boxes) > 0 else 0

        # ENHANCED
        condition = classify_condition(img)
        enhanced = enhance_by_condition(img, condition)

        res_e = model(enhanced)
        conf_e = np.mean(res_e[0].boxes.conf.cpu().numpy()) if len(res_e[0].boxes) > 0 else 0

        orig_conf.append(conf_o)
        enh_conf.append(conf_e)
        names.append(filename)

plt.style.use('ggplot')

# GRAPH 1 — Confidence Comparison
plt.figure()
plt.plot(orig_conf, marker='o', label="Original")
plt.plot(enh_conf, marker='o', label="Enhanced")
plt.xticks(range(len(names)), names, rotation=45)
plt.ylabel("Confidence")
plt.title("Confidence Comparison")
plt.legend()

# GRAPH 2 — High Confidence Detections
orig_high = [sum(model(cv2.imread(os.path.join(image_folder, n)))[0].boxes.conf.cpu().numpy() > 0.5) for n in names]
enh_high = []

for n in names:
    img = cv2.imread(os.path.join(image_folder, n))
    condition = classify_condition(img)
    enhanced = enhance_by_condition(img, condition)
    res = model(enhanced)
    enh_high.append(sum(res[0].boxes.conf.cpu().numpy() > 0.5))

plt.figure()
plt.bar([i - 0.2 for i in range(len(names))], orig_high, width=0.4, label="Original")
plt.bar([i + 0.2 for i in range(len(names))], enh_high, width=0.4, label="Enhanced")
plt.xticks(range(len(names)), names, rotation=45)
plt.ylabel("High Confidence Detections")
plt.title("Filtered Detection Comparison")
plt.legend()

# GRAPH 3 — Improvement %
improvement = [(e - o) * 100 for o, e in zip(orig_conf, enh_conf)]

plt.figure()
plt.bar(names, improvement)
plt.xticks(rotation=45)
plt.ylabel("Improvement (%)")
plt.title("Confidence Improvement (%)")

# 🔥 SHOW ALL AT ONCE
plt.show()