🌊 Adaptive Underwater Vision System
📌 Overview

This project is an AI-based system for detecting underwater debris using YOLOv8. It enhances images based on environmental conditions and performs object detection with pollution scoring to assess environmental impact.

🚀 Features
🔍 Object Detection using YOLOv8
🌫️ Condition-based Image Enhancement
📊 Pollution Scoring (based on bounding boxes)
📈 Evaluation using confidence & detection metrics
🖥️ Streamlit UI for visualization
🧠 System Pipeline
Image → Condition Detection → Enhancement → YOLO Detection → Pollution Scoring → Output
📂 Project Structure
underwater-vision/
├── images/
├── outputs/
├── models/
├── scripts/
│   ├── detect.py
│   ├── enhance.py
│   ├── condition.py
│   ├── compare.py
├── app.py
├── README.md
▶️ How to Run
🔹 Run Detection
python scripts/detect.py
🔹 Run Evaluation
python scripts/compare.py
🔹 Run UI
streamlit run app.py
📊 Pollution Scoring

Pollution level is calculated using:

📦 Number of detected objects
📐 Bounding box area (coverage)
🎯 Detection confidence
📈 Results
✅ Improved detection in low visibility conditions
✅ Reduced false positives after enhancement
✅ Pollution level classification (LOW / MEDIUM / HIGH)
🚧 Future Work
🔄 Multi-class detection
🎥 Real-time video processing
🧩 Instance segmentation
👨‍💻 Author

Anshul Tickoo
