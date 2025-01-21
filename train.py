import os
# import kagglehub
from ultralytics import YOLO
# import shutil


# path = kagglehub.dataset_download("dataset")
folder = "/home/sergio/Repositorios/Bones-Fracture/dataset"
model = YOLO("yolo11m.yaml")

# Train the model
results = model.train(data=os.path.join(folder,  "data.yaml"), epochs=100) # Train the model
