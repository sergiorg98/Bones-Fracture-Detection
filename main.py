import sys
# import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
    QLabel, QLineEdit, QPushButton, QFileDialog, QSlider
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from ultralytics import YOLO

from skimage.morphology import binary_erosion, rectangle
from skimage.measure import label, regionprops_table
from skimage.color import label2rgb

import numpy as np

YOLO_MODEL_PATH = "runs/detect/train9/weights/best.pt"

import cv2
class ImageEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Editor with OpenCV")
        self.setGeometry(100, 100, 800, 600)

        self.image = None
        self.original_image = None
        self.initUI()

    def initUI(self):
        # Main widget and layout
        main_widget = QWidget()
        main_widget.setStyleSheet("background-color: pink;")
        main_layout = QHBoxLayout() 
        main_widget.setLayout(main_layout)

        # Left section for displaying the image
        self.image_label = QLabel("No Image Loaded")
        self.image_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.image_label, 3)

        # Right sidebar for controls
        sidebar = QVBoxLayout()
        sidebar.setAlignment(Qt.AlignTop)

        # Upload button
        upload_button = QPushButton("Upload Image")
        upload_button.clicked.connect(self.load_image)
        sidebar.addWidget(upload_button)
        
        # Canny edge detection button
        binarization_button = QPushButton("Binarización")
        binarization_button.clicked.connect(self.binarice)
        sidebar.addWidget(binarization_button)

        # Threshold slider
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(100)
        self.threshold_slider.setValue(90)
        self.threshold_slider.setSingleStep(1)
        self.threshold_slider.setTickInterval(1)
        sidebar.addWidget(self.threshold_slider)

        self.threshold_label = QLabel("Threshold: 80")
        self.threshold_label.setFixedHeight(self.threshold_label.sizeHint().height())
        self.threshold_slider.valueChanged.connect(lambda value: self.threshold_label.setText(f"Threshold: {value}"))
        sidebar.addWidget(self.threshold_label)

        #Boton detectar con Yolo
        yolo_button = QPushButton("Detectar con Yolo")
        yolo_button.clicked.connect(self.yolo_detection)
        sidebar.addWidget(yolo_button)

        # Canny edge detection button
        distingir_regiones_button = QPushButton("Distinguir Regiones")
        distingir_regiones_button.clicked.connect(self.distinguir_regiones)
        sidebar.addWidget(distingir_regiones_button)

        # Threshold slider Yolo
        self.threshold_slider_yolo = QSlider(Qt.Horizontal)
        self.threshold_slider_yolo.setMinimum(0)
        self.threshold_slider_yolo.setMaximum(100)
        self.threshold_slider_yolo.setValue(15)
        self.threshold_slider_yolo.setSingleStep(1)
        self.threshold_slider_yolo.setTickInterval(1)
        sidebar.addWidget(self.threshold_slider_yolo)

        self.threshold_label_yolo = QLabel("Threshold: 15")
        self.threshold_label_yolo.setFixedHeight(self.threshold_label.sizeHint().height())
        self.threshold_slider_yolo.valueChanged.connect(lambda value: self.threshold_label_yolo.setText(f"Threshold: {value}"))
        sidebar.addWidget(self.threshold_label_yolo)

        # Canny edge detection button
        canny_button = QPushButton("Apply Canny Edge Detection")
        canny_button.clicked.connect(self.apply_canny)
        sidebar.addWidget(canny_button)

        # Add sidebar to the layout
        main_layout.addLayout(sidebar, 1)

        self.setCentralWidget(main_widget)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Image Files (*.png *.jpg *.bmp)")
        if file_path:
            # self.path_input.setText(file_path)
            self.image = cv2.imread(file_path)
            self.original_image = self.image.copy()

            self.display_image()

    def display_image(self, image=None):
        if self.image is not None:
            # Convert image to QPixmap
            rgb_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.image_label.setPixmap(pixmap)
        else:
            self.image_label.setText("No Image Loaded")

    def apply_canny(self):
        if self.original_image is not None:
            # Convert to grayscale and apply Canny
            gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray_image, 100, 200)

            # Convert edges back to 3-channel image for display
            edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            self.image = edges_colored
            self.display_image()
        else:
            self.image_label.setText("No Image Loaded")

    def binarice(self):
        if self.original_image is not None:
            # Convert to grayscale and apply Canny
            gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)

            # Apply Otsu's binarization
            threshold_value = (float(self.threshold_label.text().split(": ")[1])/100)*255
            _, binarized = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)

            self.image = binarized
            self.display_image()
        else:
            self.image_label.setText("No Image Loaded")

        
    def yolo_detection(self):
        if self.original_image is not None:
            self.image = self.original_image.copy()
            model = YOLO(YOLO_MODEL_PATH)
            results =  model(self.original_image)[0]

            for result in results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = result
                print(f"{class_id}")
                threshold_value = float(self.threshold_label_yolo.text().split(": ")[1])/100
                if score > threshold_value:
                    print(f"scores: {score}")
                    detected = cv2.rectangle(self.image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                    self.image = detected
            self.display_image()

        else:
            self.image_label.setText("No Image Loaded")

    def distinguir_regiones(self):
        if self.image is not None:
            # Erodes y dilates
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 1))
            BW_eroded = cv2.erode(self.image, kernel, iterations=1)
            BW_filled = cv2.morphologyEx(BW_eroded, cv2.MORPH_CLOSE, kernel)

            # Cálculo de contornos
            labeled_image = label(BW_filled)
            colored_labels = label2rgb(labeled_image, bg_label=0)
            props = regionprops_table(labeled_image, properties=['area', 'bbox', 'perimeter', 'eccentricity'])
            areas = props['area']
            sorted_indices = np.argsort(areas)[::-1]
            main_regions = sorted_indices[:2]

            bbox1 = props['bbox-0'][main_regions[0]], props['bbox-1'][main_regions[0]], props['bbox-2'][main_regions[0]], props['bbox-3'][main_regions[0]]
            bbox2 = props['bbox-0'][main_regions[1]], props['bbox-1'][main_regions[1]], props['bbox-2'][main_regions[1]], props['bbox-3'][main_regions[1]]

            self.image = self.original_image.copy()
            cv2.rectangle(self.image, (bbox1[1], bbox1[0]), (bbox1[3], bbox1[2]), (0, 0, 255), 1)
            cv2.rectangle(self.image, (bbox2[1], bbox2[0]), (bbox2[3], bbox2[2]), (255, 0, 0), 1)

            # Dibujar rectángulo que coincide con bbox1 y bbox2
            rect_x_min = max(bbox1[1], bbox2[1])
            rect_x_max = min(bbox1[3], bbox2[3])
            sep_min = max(bbox1[0], bbox2[0])
            sep_max = min(bbox1[2], bbox2[2])
            rect_width = rect_x_max - rect_x_min
            rect_height = sep_max - sep_min

            cv2.rectangle(self.image, (rect_x_min, sep_min), (rect_x_min + rect_width, sep_min + rect_height), (0, 255, 0), 2)
            
            self.display_image()


        else:
            self.image_label.setText("No Image Loaded")




if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageEditor()
    window.show()
    sys.exit(app.exec_())
