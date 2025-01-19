import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PyQt5.QtGui import QPixmap

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        # Set up the main layout
        main_layout = QHBoxLayout()

        # Left section with an image
        self.image_label = QLabel(self)
        pixmap = QPixmap('image.png')  # Replace with the path to your image
        self.image_label.setPixmap(pixmap)
        main_layout.addWidget(self.image_label)

        # Right section with a sidebar and buttons
        sidebar_layout = QVBoxLayout()
        
        button1 = QPushButton('Button 1', self)
        button2 = QPushButton('Button 2', self)
        sidebar_layout.addWidget(button1)
        sidebar_layout.addWidget(button2)
        main_layout.addLayout(sidebar_layout)

        # Set the main layout
        self.setLayout(main_layout)
        self.setWindowTitle('Image and Sidebar Interface')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())