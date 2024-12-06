
import cv2
import time
import numpy as np
from flask import Flask, render_template
import os
import requests
from pathlib import Path
from zipfile import ZipFile
from sklearn.model_selection import train_test_split
import re
from tqdm import tqdm
import shutil

from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm

download_path = Path("/content/common-crop-diseases48-classes.zip")
extracted_path = Path("/content/common-crop-diseases48-classes")
dataset_path = extracted_path / "Common_Crops"
output_path = Path("/content/yolo_classification_dataset")
train_dir = output_path / "train"
val_dir = output_path / "val"
test_dir = output_path / "test"

download_url = "https://www.kaggle.com/api/v1/datasets/download/divyantiwari/common-crop-diseases48-classes"

def download_with_progress(url, dest_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        total_size = int(response.headers.get('content-length', 0))
        with open(dest_path, "wb") as f, tqdm(
            desc=f"Downloading {dest_path.name}",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
                bar.update(len(chunk))
    else:
        raise Exception(f"Failed to download dataset. HTTP Status Code: {response.status_code}")

if not dataset_path.exists():
    download_with_progress(download_url, download_path)
    print(f"Dataset downloaded to: {download_path}")


    with ZipFile(download_path, 'r') as zip_ref:
        zip_ref.extractall(extracted_path)
    print(f"Dataset extracted to: {extracted_path}")
else:
    print(f"Dataset already exists at: {dataset_path}")

train_dir.mkdir(parents=True, exist_ok=True)
val_dir.mkdir(parents=True, exist_ok=True)
test_dir.mkdir(parents=True, exist_ok=True)

image_pattern = re.compile(r"\.(jpg|jpeg|png)$", re.IGNORECASE)

print(f"Scanning dataset at: {dataset_path}")
classes = [cls.name for cls in dataset_path.iterdir() if cls.is_dir()]
print(f"Found {len(classes)} class folders in the dataset: {classes}")

for cls in tqdm(classes, desc="Processing classes"):
    cls_path = dataset_path / cls

    images = [img for img in cls_path.iterdir() if image_pattern.search(img.name.lower())]
    print(f"Class '{cls}' contains {len(images)} valid images.")

    if len(images) == 0:
        print(f"Skipping empty class folder: {cls}")
        continue

    train_images, val_test_images = train_test_split(images, test_size=0.15, random_state=42)
    val_images, test_images = train_test_split(val_test_images, test_size=0.5, random_state=42)

    (train_dir / cls).mkdir(parents=True, exist_ok=True)
    (val_dir / cls).mkdir(parents=True, exist_ok=True)
    (test_dir / cls).mkdir(parents=True, exist_ok=True)

    for img in train_images:
        shutil.copy(img, train_dir / cls / img.name)
    for img in val_images:
        shutil.copy(img, val_dir / cls / img.name)
    for img in test_images:
        shutil.copy(img, test_dir / cls / img.name)

    print(f"Processed class '{cls}': {len(train_images)} train, {len(val_images)} val, {len(test_images)} test.")

print(f"Dataset organized successfully in YOLO classification format at: {output_path}")
print(f"Train directory: {len(list(train_dir.rglob('*.*')))} files.")
print(f"Validation directory: {len(list(val_dir.rglob('*.*')))} files.")
print(f"Test directory: {len(list(test_dir.rglob('*.*')))} files.")

dataset_path = "/content/yolo_classification_dataset"

model = YOLO("yolov8n-cls.pt")

results = model.train(
    data=dataset_path,
    epochs=50,
    imgsz=224,
    batch=16,
    half=True,
    name="yolov8n_classification"
)
print(f"Training complete. Best model saved at: {results['best']}")
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm

model_path = "/content/runs/classify/yolov8n_classification/weights/best.pt"
model = YOLO(model_path)

test_dir = Path("/content/yolo_classification_dataset/test")
classes = sorted([cls.name for cls in test_dir.iterdir() if cls.is_dir()])

print(f"Classes: {classes}")
print(f"Testing on images from: {test_dir}")

total_images = 0
correct_predictions = 0
for cls in tqdm(classes, desc="Testing classes"):
    cls_path = test_dir / cls
    images = list(cls_path.glob("*.*"))

    for img_path in images:
        total_images += 1
        results = model(img_path)
        predicted_class = classes[results[0].probs.top1]
        if predicted_class == cls:
            correct_predictions += 1

accuracy = (correct_predictions / total_images) * 100 if total_images > 0 else 0
print(f"Accuracy on the test set: {accuracy:.2f}%")
print(f"Total images tested: {total_images}")
print(f"Correct predictions: {correct_predictions}")



trees=""
x=""
app = Flask(__name__)
@app.route('/')
def home():
    global x
    message =x
    return render_template("inter.html",message=message)

if (__name__ == "__main__"):
    app.run(debug=True)
stability_counter = 0
stable_frames_required = 5
templates = {
    "good health condition": cv2.imread(r"C:\Users\rayen\OneDrive\Bureau\tsyp\assp\templates\lbs.jpg", 0),
    "good health condition 2": cv2.imread(r"C:\Users\rayen\OneDrive\Bureau\tsyp\assp\templates\lbss.jpg", 0),
    "too much water": cv2.imread(r"C:\Users\rayen\OneDrive\Bureau\tsyp\assp\templates\ms9iya brcha.jpg", 0),
    "lack of water": cv2.imread(r"C:\Users\rayen\OneDrive\Bureau\tsyp\assp\templates\n9ssa me.jpg", 0),
    "signs of sickness": cv2.imread(r"C:\Users\rayen\OneDrive\Bureau\tsyp\assp\templates\9a3da tmrdh.jpg", 0),
    "sick": cv2.imread(r"C:\Users\rayen\OneDrive\Bureau\tsyp\assp\templates\mridha.jpg", 0),
    }

cap = cv2.VideoCapture(0)

ret, frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
_, pic0 = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
for tree, template in templates.items():
    if template is None:
        continue
    result = cv2.matchTemplate( trees,template, cv2.TM_CCOEFF_NORMED)
    _, max_val= cv2.minMaxLoc(result)
    if max_val > 0.8:
        x=tree
        home()
