import cv2
import os
import random
from ultralytics import YOLO
import matplotlib.pyplot as plt

IMAGES_DIR = "/home/sergio/Repositorios/Bones-Fracture/dataset"
FOLDER_IMAGES = "test/images"
FOLDER_LABELS = "BoneFractureYolo8/valid/labels"
MODEL_PATH = "/home/sergio/Repositorios/Bones-Fracture/runs/detect/train9/weights/best.pt"

class_names = ['elbow positive', 'fingers positive', 'forearm fracture', 'humerus fracture', 'humerus', 'shoulder fracture', 'wrist positive']

model = YOLO(MODEL_PATH)

test_images_path  = os.path.join(IMAGES_DIR, FOLDER_IMAGES)

test_images_list = os.listdir(test_images_path)

random_test_images = random.sample(test_images_list, 16)
# random_test_labels = [image.replace('.jpg', '.txt') for image in random_test_images]

# Set up the plot
fig, axs = plt.subplots(4, 4, figsize=(20, 10))  # Adjust the grid size based on the number of images

# Loop over the random images and display them with predictions
for i, image_file in enumerate(random_test_images):
    # Load the image
    image_path = os.path.join(test_images_path, image_file)
    image = cv2.imread(image_path)

    # Run the model's predict method on the image
    results = model.predict(source=image_path, show=False)  # Set show=True if running locally to see pop-up windows

    # Draw the bounding boxes on the image
    # `results` is an object that contains predictions, use its method to draw bounding boxes
    for result in results:
        # Iterate over each prediction
        for box in result.boxes:
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0])  # Bounding box coordinates
            class_id = int(box.cls[0])  # Extract class ID
            confidence = box.conf[0]  # Extract confidence score

            # Get the class name
            class_name = class_names[class_id]

            # Draw the bounding box
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Green color, thickness=2

            # Draw the class name and confidence on the image
            label = f"{class_name} {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            label_ymin = max(y_min, label_size[1] + 10)
            cv2.rectangle(image, (x_min, label_ymin - label_size[1] - 10), (x_min + label_size[0], label_ymin + 5), (0, 255, 0), -1)
            cv2.putText(image, label, (x_min, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            # Add the image name to the plot
            # axs[row, col].set_title(image_file, fontsize=10)

    # Plot the image with predictions
    row = i // 4
    col = i % 4
    axs[row, col].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[row, col].axis('off')

# Show the plot
plt.show()