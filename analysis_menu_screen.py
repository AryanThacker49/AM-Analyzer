import sys
import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QHBoxLayout, QFrame, QPushButton, 
    QGridLayout, QLineEdit, QSizePolicy, QApplication, QSlider
)
from PySide6.QtGui import QPixmap, Qt
from PySide6 import QtCore, QtGui

class AnalysisMenuScreen(QWidget):

    goPrev = QtCore.Signal()
    goNext = QtCore.Signal()
    
    # --- Signals for each tool ---
    goToPorosity = QtCore.Signal()
    goToThickness = QtCore.Signal()
    goToCustom = QtCore.Signal()
    goToNotch = QtCore.Signal() 
    
    def __init__(self, controller=None):
        super().__init__()
        self.controller = controller
        self.image_data = [] 
        self.current_index = 0
        self._build_ui()

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        title = QLabel("Analysis Menu")
        title.setStyleSheet("font-size: 24px; font-weight: bold;")
        main_layout.addWidget(title)

        # --- Main content area ---
        button_layout = QHBoxLayout()
        button_layout.setSpacing(20)
        button_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        # --- Porosity Button ---
        self.porosity_btn = QPushButton("Porosity Analysis")
        self.porosity_btn.clicked.connect(self.goToPorosity.emit)
        button_layout.addWidget(self.porosity_btn)
        
        # --- Thickness Button ---
        self.thickness_btn = QPushButton("Thickness v/s Height")
        self.thickness_btn.clicked.connect(self.goToThickness.emit)
        button_layout.addWidget(self.thickness_btn)
        
        # --- Notch Button (Swapped) ---
        self.notch_btn = QPushButton("Notch Detection")
        self.notch_btn.clicked.connect(self.goToNotch.emit)
        button_layout.addWidget(self.notch_btn)
        
        # --- Custom Button (Swapped) ---
        self.custom_btn = QPushButton("Custom Script")
        self.custom_btn.clicked.connect(self.goToCustom.emit)
        button_layout.addWidget(self.custom_btn)
        
        # --- Style the tool buttons ---
        for btn in (self.porosity_btn, self.thickness_btn, self.custom_btn, self.notch_btn):
            btn.setMinimumSize(200, 80) 
            btn.setStyleSheet("""
                QPushButton {
                    font-size: 16px;
                    font-weight: bold;
                    background-color: #0B3C5D; 
                    color: white;
                    border: 1px solid #1E5F8A; 
                    border-radius: 8px;
                    padding: 10px;
                }
                QPushButton:hover {
                    background-color: #1E5F8A;
                }
                QPushButton:pressed {
                    background-color: #0A324E;
                }
            """)

        main_layout.addLayout(button_layout)
        main_layout.addStretch() 

        # --- Bottom navigation ---
        bottom = QHBoxLayout()
        self.prev_btn = QPushButton("Prev: Substrate")
        self.prev_btn.setFixedHeight(50)
        self.prev_btn.clicked.connect(self.goPrev.emit)
        bottom.addWidget(self.prev_btn)
        bottom.addStretch() 
        main_layout.addLayout(bottom)

    def load_image_data(self, image_data_list):
        self.image_data = image_data_list
        print(f"AnalysisMenuScreen loaded {len(self.image_data)} images.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = AnalysisMenuScreen()
    win.resize(1200, 800)
    win.show()
    sys.exit(app.exec())