import os
import kagglehub
from ultralytics import YOLO
import shutil


path = kagglehub.dataset_download("pkdarabi/bone-fracture-detection-computer-vision-project")
folder = "BoneFractureYolo8"
model = YOLO("yolo11n.yaml")

# Train the model
results = model.train(data=os.path.join(path,  folder,  "data.yaml"), epochs=100, batch=4, imgsz=640) # Train the model
