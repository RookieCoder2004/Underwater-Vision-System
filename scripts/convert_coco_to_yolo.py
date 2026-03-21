import json
import os
from PIL import Image

# paths
json_path = "dataset.json"
images_dir = "all_images"   # we'll create this
labels_dir = "dataset/labels/train"

os.makedirs(labels_dir, exist_ok=True)

# load json
with open(json_path) as f:
    data = json.load(f)

images = {img["id"]: img for img in data["images"]}

# create label files
for ann in data["annotations"]:
    img_id = ann["image_id"]
    img_info = images[img_id]

    img_path = os.path.join(images_dir, img_info["file_name"])

    if not os.path.exists(img_path):
        continue

    # image size
    img = Image.open(img_path)
    w, h = img.size

    # bbox (COCO format)
    x, y, bw, bh = ann["bbox"]

    # convert to YOLO
    x_center = (x + bw/2) / w
    y_center = (y + bh/2) / h
    bw /= w
    bh /= h

    label_file = os.path.join(labels_dir, img_info["file_name"].replace(".jpg", ".txt"))

    with open(label_file, "a") as f:
        f.write(f"0 {x_center} {y_center} {bw} {bh}\n")

print("Conversion complete!")