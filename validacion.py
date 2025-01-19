import os
from ultralytics import YOLO

IMAGES_DIR = "/home/sergio/.cache/kagglehub/datasets/pkdarabi/bone-fracture-detection-computer-vision-project/versions/2"
FOLDER = "BoneFractureYolo8"



model_path = "/home/sergio/Repositorios/Bones-Fracture/runs/detect/train/weights/best.pt"

model = YOLO(model_path)


# Train the model
results = model.val(data=os.path.join(IMAGES_DIR,  FOLDER,  "data.yaml"), epochs=100, batch=4, imgsz=640) # Train the model
