import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
    QLabel, QLineEdit, QPushButton, QFileDialog
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

class ImageEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Editor with OpenCV")
        self.setGeometry(100, 100, 800, 600)

        self.image = None
        self.initUI()

    def initUI(self):
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)

        # Left section for displaying the image
        self.image_label = QLabel("No Image Loaded")
        self.image_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.image_label, 3)

        # Right sidebar for controls
        sidebar = QVBoxLayout()

        # Input field for image path
        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText("Enter image path or browse...")
        sidebar.addWidget(self.path_input)

        # Upload button
        upload_button = QPushButton("Upload Image")
        upload_button.clicked.connect(self.load_image)
        sidebar.addWidget(upload_button)

        # Canny edge detection button
        canny_button = QPushButton("Apply Canny Edge Detection")
        canny_button.clicked.connect(self.apply_canny)
        sidebar.addWidget(canny_button)

        # Canny edge detection button
        canny_button = QPushButton("Apply Segmentation")
        canny_button.clicked.connect(self.image_segmentation)
        sidebar.addWidget(canny_button)

        # Add sidebar to the layout
        main_layout.addLayout(sidebar, 1)

        self.setCentralWidget(main_widget)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Image Files (*.png *.jpg *.bmp)")
        if file_path:
            self.path_input.setText(file_path)
            self.image = cv2.imread(file_path)
            self.display_image()

    def display_image(self):
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
        if self.image is not None:
            # Convert to grayscale and apply Canny
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray_image, 100, 200)

            # Convert edges back to 3-channel image for display
            edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            self.image = edges_colored
            self.display_image()
        else:
            self.image_label.setText("No Image Loaded")

    def image_segmentation(self):
        if self.image is not None:
            # Convert to grayscale
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

            # Apply GaussianBlur to reduce noise and improve segmentation
            blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

            # Apply Otsu's thresholding for segmentation
            _, segmented_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Convert segmented image back to 3-channel image for display
            segmented_colored = cv2.cvtColor(segmented_image, cv2.COLOR_GRAY2BGR)
            self.image = segmented_colored
            self.display_image()
        else:
            self.image_label.setText("No Image Loaded")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageEditor()
    window.show()
    sys.exit(app.exec_())
