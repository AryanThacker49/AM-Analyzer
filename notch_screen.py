import sys
import os
import cv2
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QHBoxLayout, QFrame, QPushButton, 
    QGridLayout, QLineEdit, QSizePolicy, QApplication, QSlider, QRadioButton, QButtonGroup
)
from PySide6.QtGui import QPixmap, Qt, QImage, QPainter, QPen, QColor
from PySide6 import QtCore, QtGui

# Matplotlib for embedding graphs
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# =====================================================================
# HELPER CLASS: AspectRatioLabel
# =====================================================================
class AspectRatioLabel(QLabel):
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setMinimumSize(1, 1)
        self.setAlignment(Qt.AlignCenter)
        self._original_pixmap = None
        self._highlight_y = None # For the hover line

    def setPixmap(self, pixmap):
        self._original_pixmap = pixmap
        self._update_pixmap()

    def set_highlight_line(self, y_coord):
        """Sets a Y-coordinate to draw a horizontal line across."""
        self._highlight_y = y_coord
        self.update() # Trigger paintEvent

    def resizeEvent(self, event):
        self._update_pixmap()
        super().resizeEvent(event)

    def _update_pixmap(self):
        if self._original_pixmap and not self._original_pixmap.isNull():
            scaled = self._original_pixmap.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            super().setPixmap(scaled)

    def paintEvent(self, event):
        super().paintEvent(event) # Draw the image
        
        if self._highlight_y is not None and self._original_pixmap:
            # Map the image Y coordinate to the widget Y coordinate
            w_widget = self.width()
            h_widget = self.height()
            
            # Recalculate scaling (same logic as _update_pixmap)
            if self.pixmap():
                scaled_w = self.pixmap().width()
                scaled_h = self.pixmap().height()
                
                # Offset of the image within the label (it's centered)
                offset_y = (h_widget - scaled_h) // 2
                
                # Scale factor
                scale_y = scaled_h / self._original_pixmap.height()
                
                # Calculate widget Y
                widget_y = int(self._highlight_y * scale_y) + offset_y
                
                # Draw Line
                painter = QPainter(self)
                painter.setPen(QPen(QColor(0, 255, 255), 2)) # Cyan Line
                painter.drawLine(0, widget_y, w_widget, widget_y)
                painter.end()

# =====================================================================
# MAIN SCREEN CLASS
# =====================================================================
class NotchScreen(QWidget):

    goPrev = QtCore.Signal()
    
    def __init__(self, controller=None):
        super().__init__()
        self.controller = controller
        self.image_data = [] 
        self.current_index = 0
        
        # Store analysis data for interactivity
        self.substrate_y = 0 
        
        self._build_ui()

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        title = QLabel("Notch Detection (Derivative Analysis)")
        title.setStyleSheet("font-size: 24px; font-weight: bold;")
        main_layout.addWidget(title)

        # --- Main Content: Image (Left) vs Graphs (Right) ---
        content_layout = QHBoxLayout()
        
        # Left: Image
        image_frame = QFrame()
        image_layout = QVBoxLayout(image_frame)
        self.main_image_label = AspectRatioLabel("Image Preview")
        self.main_image_label.setStyleSheet("border: 2px solid #444; background: #111;")
        self.main_image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        image_layout.addWidget(self.main_image_label)
        
        nav_btns = QHBoxLayout()
        self.prev_img_btn = QPushButton("Prev Image")
        self.next_img_btn = QPushButton("Next Image")
        self.prev_img_btn.clicked.connect(self.prev_image)
        self.next_img_btn.clicked.connect(self.next_image)
        nav_btns.addWidget(self.prev_img_btn)
        nav_btns.addWidget(self.next_img_btn)
        image_layout.addLayout(nav_btns)
        
        content_layout.addWidget(image_frame, stretch=1)

        # Right: Graphs & Controls
        right_frame = QFrame()
        right_layout = QVBoxLayout(right_frame)
        
        # Matplotlib Figure (Taller to fit 4 graphs)
        self.figure = Figure(figsize=(5, 12), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        
        # Connect Hover Event
        self.canvas.mpl_connect('motion_notify_event', self._on_graph_hover)
        
        right_layout.addWidget(self.canvas)
        
        # Controls
        ctrl_layout = QGridLayout()
        
        # --- Edge Selection Toggle ---
        ctrl_layout.addWidget(QLabel("Edge to Analyze:"), 0, 0)
        edge_container = QWidget()
        edge_layout = QHBoxLayout(edge_container)
        edge_layout.setContentsMargins(0,0,0,0)
        
        self.radio_left = QRadioButton("Left")
        self.radio_right = QRadioButton("Right")
        self.radio_right.setChecked(True) # Default to Right
        
        self.edge_group = QButtonGroup(self)
        self.edge_group.addButton(self.radio_left)
        self.edge_group.addButton(self.radio_right)
        # Re-run analysis on toggle
        self.edge_group.buttonClicked.connect(self._run_notch_detection)

        edge_layout.addWidget(self.radio_left)
        edge_layout.addWidget(self.radio_right)
        ctrl_layout.addWidget(edge_container, 0, 1)
        
        ctrl_layout.addWidget(QLabel("Prominence (Depth):"), 1, 0)
        self.prominence_input = QLineEdit("2.0") 
        self.prominence_input.setToolTip("Minimum depth of the notch to be detected.")
        ctrl_layout.addWidget(self.prominence_input, 1, 1)
        
        ctrl_layout.addWidget(QLabel("Min Distance (px):"), 2, 0)
        self.distance_input = QLineEdit("10") 
        self.distance_input.setToolTip("Minimum pixel distance between two notches.")
        ctrl_layout.addWidget(self.distance_input, 2, 1)
        
        # --- Hide Bottom Slider ---
        ctrl_layout.addWidget(QLabel("Hide Bottom (px):"), 3, 0)
        slider_container = QWidget()
        slider_hbox = QHBoxLayout(slider_container)
        slider_hbox.setContentsMargins(0,0,0,0)
        self.hide_slider = QSlider(Qt.Horizontal)
        self.hide_slider.setRange(0, 300) 
        self.hide_slider.setValue(0)
        self.hide_val_label = QLabel("0")
        self.hide_val_label.setFixedWidth(30)
        self.hide_val_label.setAlignment(Qt.AlignCenter)
        self.hide_slider.valueChanged.connect(lambda v: self.hide_val_label.setText(str(v)))
        self.hide_slider.sliderReleased.connect(self._run_notch_detection) 
        slider_hbox.addWidget(self.hide_slider)
        slider_hbox.addWidget(self.hide_val_label)
        ctrl_layout.addWidget(slider_container, 3, 1)

        # --- Hide Top Slider ---
        ctrl_layout.addWidget(QLabel("Hide Top (px):"), 4, 0)
        top_slider_container = QWidget()
        top_slider_hbox = QHBoxLayout(top_slider_container)
        top_slider_hbox.setContentsMargins(0,0,0,0)
        self.hide_top_slider = QSlider(Qt.Horizontal)
        self.hide_top_slider.setRange(0, 300) 
        self.hide_top_slider.setValue(0)
        self.hide_top_val_label = QLabel("0")
        self.hide_top_val_label.setFixedWidth(30)
        self.hide_top_val_label.setAlignment(Qt.AlignCenter)
        self.hide_top_slider.valueChanged.connect(lambda v: self.hide_top_val_label.setText(str(v)))
        self.hide_top_slider.sliderReleased.connect(self._run_notch_detection) 
        top_slider_hbox.addWidget(self.hide_top_slider)
        top_slider_hbox.addWidget(self.hide_top_val_label)
        ctrl_layout.addWidget(top_slider_container, 4, 1)

        self.calc_btn = QPushButton("Run Analysis")
        self.calc_btn.setStyleSheet("background-color: #0B3C5D; font-weight: bold; padding: 10px;")
        self.calc_btn.clicked.connect(self._run_notch_detection)
        ctrl_layout.addWidget(self.calc_btn, 5, 0, 1, 2)
        
        right_layout.addLayout(ctrl_layout)

        content_layout.addWidget(right_frame, stretch=1)
        main_layout.addLayout(content_layout, stretch=1)

        # --- Bottom Navigation ---
        bottom = QHBoxLayout()
        self.prev_btn = QPushButton("Back to Menu")
        self.prev_btn.setFixedHeight(50)
        self.prev_btn.clicked.connect(self.goPrev.emit)
        bottom.addWidget(self.prev_btn)
        bottom.addStretch()
        main_layout.addLayout(bottom)

    def load_image_data(self, image_data_list):
        self.image_data = image_data_list
        self.current_index = 0
        if self.image_data:
            self.display_image()
            
    def display_image(self):
        if not self.image_data: return
        data = self.image_data[self.current_index]
        path = data.get('path')
        if path and os.path.exists(path):
            self.main_image_label.setPixmap(QPixmap(path))
            # Clear graphs
            self.figure.clear()
            self.canvas.draw()

    # =================================================================
    # INTERACTIVITY
    # =================================================================
    def _on_graph_hover(self, event):
        """Called when mouse moves over the matplotlib canvas."""
        if event.inaxes:
            # event.xdata corresponds to the X-axis of the graph
            # In our plots, X-axis is "Height from Substrate (px)"
            height_from_substrate = event.xdata
            
            if height_from_substrate is not None and self.substrate_y > 0:
                # Calculate image Y
                # Image Y = Substrate Y - Height
                target_y = self.substrate_y - height_from_substrate
                self.main_image_label.set_highlight_line(target_y)
        else:
            # Mouse left the graph, clear line
            self.main_image_label.set_highlight_line(None)

    # =================================================================
    # ANALYSIS LOGIC
    # =================================================================
    def calculate_moving_average(self, data, window_size=5):
        """Calculates moving average. Window size 5 = 2 before, current, 2 after."""
        kernel = np.ones(window_size) / window_size
        return np.convolve(data, kernel, mode='same')

    def _run_notch_detection(self):
        if not self.image_data: return
        data = self.image_data[self.current_index]
        path = data.get('path')
        
        # --- Get Tuning Params ---
        try:
            prominence_val = float(self.prominence_input.text())
            distance_val = int(self.distance_input.text())
            hide_bottom_val = self.hide_slider.value()
            hide_top_val = self.hide_top_slider.value()
            analyze_right = self.radio_right.isChecked()
        except ValueError:
            print("Invalid parameters")
            return

        # 1. Load and Preprocess
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Blackout Scalebar
        h, w = gray.shape
        roi_x = int((data.get("roi_x_pct", 0) / 100.0) * w)
        roi_y = int((data.get("roi_y_pct", 0) / 100.0) * h)
        roi_w = int((data.get("roi_w_pct", 30) / 100.0) * w)
        roi_h = int((data.get("roi_h_pct", 30) / 100.0) * h)
        cv2.rectangle(gray, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 0, 0), -1)

        # Contrast & Otsu
        clip_limit = data.get("contrast_clip_limit", 1.0)
        if clip_limit > 0.1:
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Keep largest blob only
        num, labels, stats, _ = cv2.connectedComponentsWithStats(otsu, connectivity=8)
        if num > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            cleaned = np.zeros_like(otsu)
            cleaned[labels == largest_label] = 255
        else:
            cleaned = otsu.copy()

        # 2. Determine Midline
        left_trace = data.get('substrate_trace_left')
        right_trace = data.get('substrate_trace_right')
        left_idx = data.get('substrate_left_endpoint_index', -1)
        right_idx = data.get('substrate_right_endpoint_index', -1)
        
        if not left_trace or not right_trace: 
            print("No substrate data found.")
            return

        if left_idx == -1: left_idx = len(left_trace) - 1
        if right_idx == -1: right_idx = len(right_trace) - 1
        
        p_left = left_trace[left_idx]
        p_right = right_trace[right_idx]
        
        mid_x = int((p_left[0] + p_right[0]) / 2)
        start_y = min(p_left[1], p_right[1])
        
        self.substrate_y = start_y # Store for hover event
        
        # 3. Scan Upwards
        y_coords = []
        raw_dists = []
        
        # Start 4 pixels below interface
        scan_start_y = min(h - 1, start_y + 4)
        
        for y in range(scan_start_y, 0, -1):
            row = cleaned[y, :]
            white_pixels = np.where(row == 255)[0]
            
            if len(white_pixels) > 0:
                # Determine which edge we care about
                l_edge = white_pixels[0]
                r_edge = white_pixels[-1]
                
                dist = 0
                if analyze_right:
                    dist = r_edge - mid_x # Positive distance from midline
                else:
                    dist = mid_x - l_edge # Positive distance from midline
                
                if dist > 0:
                    y_coords.append(start_y - y) # Height
                    raw_dists.append(dist)

        # 4. Filter Data 
        y_arr = np.array(y_coords)
        raw_arr = np.array(raw_dists)
        
        if len(y_arr) < 2: return

        max_height = np.max(y_arr)
        mask = (y_arr > hide_bottom_val) & (y_arr < (max_height - hide_top_val))
        
        y_arr = y_arr[mask]
        raw_arr = raw_arr[mask]
        
        if len(y_arr) < 5: 
            return
            
        # --- 5. Calculate Derivatives ---
        # First: Calculate Gradient of RAW data
        d1_raw = np.gradient(raw_arr)
        
        # Second: Smooth the Gradient
        d1_smoothed = self.calculate_moving_average(d1_raw, window_size=5)
        
        # Third: Calculate subsequent derivatives from the smoothed gradient
        d2 = np.gradient(d1_smoothed)
        d3 = np.gradient(d2)
        
        # 6. Find Peaks (Notches) using RAW data (Valleys)
        peaks, _ = find_peaks(-raw_arr, prominence=prominence_val, distance=distance_val)
        
        # 7. Visualize
        cv2.line(image, (mid_x, start_y), (mid_x, 0), (255, 255, 0), 1) # Cyan midline
        
        for idx in peaks:
            # We need to map the peak index (which refers to raw_arr)
            # back to the filtered array.
            # Wait, 'peaks' are indices into 'raw_arr'.
            # And 'raw_arr' is ALREADY the filtered array.
            # So this is correct.
            
            h_val = y_arr[idx]
            dist_val = raw_arr[idx]
            
            img_y = start_y - h_val
            
            if analyze_right:
                img_x = mid_x + dist_val
            else:
                img_x = mid_x - dist_val
                
            cv2.circle(image, (int(img_x), int(img_y)), 6, (0, 0, 255), -1)
            cv2.line(image, (0, int(img_y)), (w, int(img_y)), (0, 0, 255), 1)

        self._update_image_display(image)
        
        # 8. Plot Graphs
        self._plot_debug_graphs(y_arr, raw_arr, d1_smoothed, d2, d3, peaks, analyze_right)

    def _update_image_display(self, cv_img_bgr):
        h, w, ch = cv_img_bgr.shape
        bytes_per_line = ch * w
        qt_img = QImage(cv_img_bgr.data, w, h, bytes_per_line, QImage.Format_BGR888).rgbSwapped()
        self.main_image_label.setPixmap(QPixmap.fromImage(qt_img))

    def _plot_debug_graphs(self, y, raw, d1, d2, d3, peaks, is_right):
        self.figure.clear()
        
        edge_name = "Right" if is_right else "Left"
        
        # 1. Raw
        ax1 = self.figure.add_subplot(411)
        ax1.plot(y, raw, 'b-', label='Raw Dist')
        ax1.plot(y[peaks], raw[peaks], 'ro', label='Notch')
        ax1.set_ylabel("Dist")
        ax1.set_title(f"{edge_name} Edge Distance (Raw)")
        ax1.legend()
        ax1.grid(True)
        
        # 2. First Deriv (Smoothed)
        ax2 = self.figure.add_subplot(412, sharex=ax1)
        ax2.plot(y, d1, 'g-', label='Smoothed Slope')
        ax2.set_ylabel("d1 (Smooth)")
        ax2.legend()
        ax2.grid(True)

        # 3. Second Deriv
        ax3 = self.figure.add_subplot(413, sharex=ax1)
        ax3.plot(y, d2, 'm-')
        ax3.set_ylabel("d2 (Curve)")
        ax3.grid(True)
        
        # 4. Third Deriv
        ax4 = self.figure.add_subplot(414, sharex=ax1)
        ax4.plot(y, d3, 'r-')
        ax4.set_ylabel("d3 (Jerk)")
        ax4.set_xlabel("Height from Substrate (px)")
        ax4.grid(True)
        
        self.figure.tight_layout()
        self.canvas.draw()

    def next_image(self):
        if not self.image_data: return
        if self.current_index < len(self.image_data) - 1:
            self.current_index += 1
            self.display_image()

    def prev_image(self):
        if not self.image_data: return
        if self.current_index > 0:
            self.current_index -= 1
            self.display_image()