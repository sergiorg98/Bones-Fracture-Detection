import cv2
import os
from ultralytics import YOLO
import matplotlib.pyplot as plt

IMAGES_DIR = "/home/sergio/.cache/kagglehub/datasets/pkdarabi/bone-fracture-detection-computer-vision-project/versions/2"
FOLDER = "BoneFractureYolo8/test/images"
IMAGE_NAME = "distal-humerus-fracture-1_jpg.rf.831cb137cfcbde1079f86abd5f5f2867.jpg"

image_path = os.path.join(IMAGES_DIR, FOLDER, IMAGE_NAME)

print(f"Ruta a la imagen: {image_path}")

frame = cv2.imread(image_path)

H, W, _ = frame.shape

# cargo modelo oentrenado
model_path = "/home/sergio/Repositorios/Bones-Fracture/runs/detect/train3/weights/best.pt"

print(os.path.exists(model_path))  # Debería imprimir True
print(os.path.exists(image_path)) # Debería imprimir True

model = YOLO(model_path)

threshold = 0.5# Umbral para filtrar detecciones (por debajo de este score no dtecta)

#procesar la imagen
results =  model(frame)[0]

#plot boxes
for result in results.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = result
    print(f"{class_id}")

    if score > threshold:
       print(f"scores: {score}")
       cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)

# Convert BGR image to RGB
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Display the image using matplotlib
plt.imshow(frame_rgb)
plt.title("Detections")
plt.axis("off")
plt.show()