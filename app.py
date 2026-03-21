import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import sys

# FIX PATH
sys.path.append(".")

from scripts.condition import classify_condition
from scripts.enhance import enhance_by_condition

st.title("🌊 Underwater Vision System")

# Load model
model = YOLO("models/best.pt")

uploaded_file = st.file_uploader("Upload an underwater image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.subheader("📷 Original Image")
    st.image(img, channels="BGR")

    # STEP 1: Condition detection
    condition = classify_condition(img)
    st.success(f"Condition detected: {condition}")

    # STEP 2: Enhancement
    enhanced = enhance_by_condition(img, condition)
    st.subheader("✨ Enhanced Image")
    st.image(enhanced, channels="BGR")

    # STEP 3: Detection
    results = model(enhanced)
    output = results[0].plot()

    # Add condition label on image
    cv2.putText(
        output,
        f"Condition: {condition}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    st.subheader("🎯 Detection Output")
    st.image(output, channels="BGR")