import sys
import os
import cv2
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QHBoxLayout, QFrame, QPushButton, 
    QGridLayout, QLineEdit, QSizePolicy, QApplication, QSlider, QRadioButton, QButtonGroup,
    QCheckBox
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
    # Signals for interaction
    clicked = QtCore.Signal(int, int) # x, y (original image coords)
    hovered = QtCore.Signal(int, int) # x, y

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setMinimumSize(1, 1)
        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(True) # Enable hover tracking
        
        self._original_pixmap = None
        self._highlight_y = None # For the graph hover line
        self._hover_notch = None # (x, y, radius) to draw highlight circle

    def setPixmap(self, pixmap):
        self._original_pixmap = pixmap
        self._update_pixmap()

    def set_highlight_line(self, y_coord):
        """Sets a Y-coordinate to draw a horizontal line across."""
        self._highlight_y = y_coord
        self.update() 

    def set_hover_notch(self, notch_tuple):
        """Highlights a specific notch (x, y). Pass None to clear."""
        self._hover_notch = notch_tuple
        self.update()

    def resizeEvent(self, event):
        self._update_pixmap()
        super().resizeEvent(event)

    def _update_pixmap(self):
        if self._original_pixmap and not self._original_pixmap.isNull():
            scaled = self._original_pixmap.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            super().setPixmap(scaled)

    # --- Coordinate Mapping ---
    def _get_image_coords(self, widget_pos):
        if not self._original_pixmap or self.pixmap().isNull(): return None
        
        w_widget = self.width()
        h_widget = self.height()
        pix_w = self.pixmap().width()
        pix_h = self.pixmap().height()
        
        off_x = (w_widget - pix_w) / 2
        off_y = (h_widget - pix_h) / 2
        
        x = widget_pos.x() - off_x
        y = widget_pos.y() - off_y
        
        scale = self._original_pixmap.width() / pix_w
        orig_x = int(x * scale)
        orig_y = int(y * scale)
        
        if 0 <= orig_x < self._original_pixmap.width() and 0 <= orig_y < self._original_pixmap.height():
            return (orig_x, orig_y)
        return None

    # --- Events ---
    def mousePressEvent(self, event):
        coords = self._get_image_coords(event.position())
        if coords:
            self.clicked.emit(coords[0], coords[1])
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        coords = self._get_image_coords(event.position())
        if coords:
            self.hovered.emit(coords[0], coords[1])
        else:
            self.hovered.emit(-1, -1) # Signal leaving image area
        super().mouseMoveEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event) # Draw the image
        
        painter = QPainter(self)
        w_widget = self.width()
        h_widget = self.height()
        
        # Draw Highlights
        if self._original_pixmap and self.pixmap():
            # Calc scaling again for overlay drawing
            scaled_h = self.pixmap().height()
            offset_y = (h_widget - scaled_h) // 2
            scale_y = scaled_h / self._original_pixmap.height()
            
            scaled_w = self.pixmap().width()
            offset_x = (w_widget - scaled_w) // 2
            scale_x = scaled_w / self._original_pixmap.width()

            # 1. Graph Hover Line (Cyan)
            if self._highlight_y is not None:
                widget_y = int(self._highlight_y * scale_y) + offset_y
                painter.setPen(QPen(QColor(0, 255, 255), 2)) 
                painter.drawLine(0, widget_y, w_widget, widget_y)

            # 2. Notch Hover Highlight (Yellow Ring)
            if self._hover_notch:
                nx, ny = self._hover_notch
                wx = int(nx * scale_x) + offset_x
                wy = int(ny * scale_y) + offset_y
                
                painter.setPen(QPen(QColor(255, 255, 0), 3)) # Yellow
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(QtCore.QPoint(wx, wy), 8, 8)

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
        
        # State
        self.substrate_y = 0 
        self.mid_x = 0
        self.edge_points_left = []
        self.edge_points_right = []
        
        # Notch Storage
        self.auto_notches_left = []  # List of (x, y)
        self.auto_notches_right = []
        # Store metadata for auto notches to reconstruct visualization: (x, y, angle, neighborB, neighborC)
        self.auto_notches_meta_left = [] 
        self.auto_notches_meta_right = []

        self.manual_added_left = [] # List of Y-coords (we snap to edge)
        self.manual_added_right = []
        
        self.deleted_notches = set() # Set of (x, y) tuples to exclude
        
        self.debug_mode = False

        self._build_ui()

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        # Header
        header = QHBoxLayout()
        title = QLabel("Notch Detection")
        title.setStyleSheet("font-size: 24px; font-weight: bold;")
        header.addWidget(title)
        
        self.debug_btn = QPushButton("Debug Mode: OFF")
        self.debug_btn.setCheckable(True)
        self.debug_btn.setStyleSheet("background-color: #555; color: #aaa; font-weight: bold;")
        self.debug_btn.clicked.connect(self._toggle_debug_mode)
        header.addWidget(self.debug_btn)
        
        # --- NEW: Light Markers Button ---
        self.light_markers_btn = QPushButton("Light Markers")
        self.light_markers_btn.setCheckable(True)
        self.light_markers_btn.setStyleSheet("background-color: #555; color: white; font-weight: bold;")
        self.light_markers_btn.toggled.connect(self._draw_overlay)
        header.addWidget(self.light_markers_btn)
        
        header.addStretch()
        
        main_layout.addLayout(header)

        # --- Main Content: Image (Left) vs Graphs (Right) ---
        content_layout = QHBoxLayout()
        
        # Left: Image
        image_frame = QFrame()
        image_layout = QVBoxLayout(image_frame)
        self.main_image_label = AspectRatioLabel("Image Preview")
        self.main_image_label.setStyleSheet("border: 2px solid #0B3C5D; background: #111;")
        self.main_image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        
        # Connect Interactive Signals
        self.main_image_label.clicked.connect(self._on_image_click)
        self.main_image_label.hovered.connect(self._on_image_hover_check)
        
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

        # Right: Graphs & Controls (Wrapped in a frame to hide/show)
        self.right_frame = QFrame()
        right_layout = QVBoxLayout(self.right_frame)
        
        # Matplotlib Figure
        self.figure = Figure(figsize=(5, 8), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.mpl_connect('motion_notify_event', self._on_graph_hover)
        right_layout.addWidget(self.canvas)
        
        # Controls
        ctrl_layout = QGridLayout()
        
        # Edge Selection (Only for Graph Viewing now)
        ctrl_layout.addWidget(QLabel("Graph Edge:"), 0, 0)
        edge_container = QWidget()
        edge_layout = QHBoxLayout(edge_container)
        edge_layout.setContentsMargins(0,0,0,0)
        self.radio_left = QRadioButton("Left")
        self.radio_right = QRadioButton("Right")
        self.radio_right.setChecked(True) 
        self.edge_group = QButtonGroup(self)
        self.edge_group.addButton(self.radio_left)
        self.edge_group.addButton(self.radio_right)
        self.edge_group.buttonClicked.connect(self._run_notch_detection)
        edge_layout.addWidget(self.radio_left)
        edge_layout.addWidget(self.radio_right)
        ctrl_layout.addWidget(edge_container, 0, 1)

        # Method
        ctrl_layout.addWidget(QLabel("Method:"), 1, 0)
        method_container = QWidget()
        method_layout = QHBoxLayout(method_container)
        method_layout.setContentsMargins(0,0,0,0)
        self.radio_angle = QRadioButton("Angle")
        self.radio_profile = QRadioButton("Profile")
        self.radio_profile.setChecked(True) # Default: Profile
        self.method_group = QButtonGroup(self)
        self.method_group.addButton(self.radio_angle)
        self.method_group.addButton(self.radio_profile)
        self.method_group.buttonClicked.connect(self._run_notch_detection)
        method_layout.addWidget(self.radio_angle)
        method_layout.addWidget(self.radio_profile)
        ctrl_layout.addWidget(method_container, 1, 1)
        
        # Prominence
        ctrl_layout.addWidget(QLabel("Prominence (Angle °):"), 2, 0)
        self.prominence_angle_input = QLineEdit("10.0") 
        ctrl_layout.addWidget(self.prominence_angle_input, 2, 1)

        ctrl_layout.addWidget(QLabel("Prominence (Dist px):"), 3, 0)
        self.prominence_dist_input = QLineEdit("1.0") # Default: 1.0
        ctrl_layout.addWidget(self.prominence_dist_input, 3, 1)
        
        # Angle Step - Changed default to 25
        ctrl_layout.addWidget(QLabel("Angle Step (px):"), 4, 0)
        self.step_input = QLineEdit("25") 
        ctrl_layout.addWidget(self.step_input, 4, 1)
        
        # Hide Bottom Slider
        ctrl_layout.addWidget(QLabel("Hide Bottom (px):"), 5, 0)
        self.hide_slider = QSlider(Qt.Horizontal)
        self.hide_slider.setRange(0, 300) 
        self.hide_slider.setValue(0)
        self.hide_slider.sliderReleased.connect(self._run_notch_detection) 
        ctrl_layout.addWidget(self.hide_slider, 5, 1)

        # Hide Top Slider
        ctrl_layout.addWidget(QLabel("Hide Top (px):"), 6, 0)
        self.hide_top_slider = QSlider(Qt.Horizontal)
        self.hide_top_slider.setRange(0, 300) 
        self.hide_top_slider.setValue(0)
        self.hide_top_slider.sliderReleased.connect(self._run_notch_detection) 
        ctrl_layout.addWidget(self.hide_top_slider, 6, 1)

        # Gaussian Blur Slider
        ctrl_layout.addWidget(QLabel("Gaussian Blur:"), 7, 0)
        self.blur_slider = QSlider(Qt.Horizontal)
        self.blur_slider.setRange(0, 20) 
        self.blur_slider.setValue(19) # Default 19
        self.blur_slider.sliderReleased.connect(self._run_notch_detection)
        ctrl_layout.addWidget(self.blur_slider, 7, 1)

        self.calc_btn = QPushButton("Re-Run Analysis")
        self.calc_btn.setStyleSheet("background-color: #0B3C5D; font-weight: bold; padding: 10px;")
        self.calc_btn.clicked.connect(self._run_notch_detection)
        ctrl_layout.addWidget(self.calc_btn, 8, 0, 1, 2)
        
        right_layout.addLayout(ctrl_layout)

        content_layout.addWidget(self.right_frame, stretch=1)
        main_layout.addLayout(content_layout, stretch=1)

        # --- Bottom Navigation ---
        bottom = QHBoxLayout()
        self.prev_btn = QPushButton("Back to Menu")
        self.prev_btn.setFixedHeight(50)
        self.prev_btn.clicked.connect(self.goPrev.emit)
        bottom.addWidget(self.prev_btn)
        bottom.addStretch()
        main_layout.addLayout(bottom)
        
        # Hide debug panel initially
        self.right_frame.setVisible(False)

    def _toggle_debug_mode(self):
        is_debug = self.debug_btn.isChecked()
        self.right_frame.setVisible(is_debug)
        if is_debug:
            self.debug_btn.setText("Debug Mode: ON")
            self.debug_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        else:
            self.debug_btn.setText("Debug Mode: OFF")
            self.debug_btn.setStyleSheet("background-color: #555; color: #aaa; font-weight: bold;")

    def load_image_data(self, image_data_list):
        self.image_data = image_data_list
        self.current_index = 0
        
        # Reset manual edits on load? Or keep persistence? 
        # For now, reset to avoid ghost points from previous image
        self.manual_added_left = []
        self.manual_added_right = []
        self.deleted_notches = set()
        self.auto_notches_left = []
        self.auto_notches_right = []
        self.auto_notches_meta_left = []
        self.auto_notches_meta_right = []
        
        if self.image_data:
            self.display_image()
            # Run detection immediately to populate image
            self._run_notch_detection()
            
    def display_image(self):
        if not self.image_data: return
        data = self.image_data[self.current_index]
        path = data.get('path')
        if path and os.path.exists(path):
            # Display Clean image initially
            self.main_image_label.setPixmap(QPixmap(path))

    # =================================================================
    # INTERACTIVITY (Manual Add/Delete)
    # =================================================================
    def _on_image_hover_check(self, x, y):
        """Highlights existing notches for deletion."""
        if x == -1: 
            self.main_image_label.set_hover_notch(None)
            return

        # Check proximity to known notches
        all_notches = self._get_current_notches_coords()
        
        hovered = None
        min_dist = 10 # Interaction radius
        
        for pt in all_notches:
            dist = np.hypot(pt[0] - x, pt[1] - y)
            if dist < min_dist:
                hovered = pt
                break
        
        self.main_image_label.set_hover_notch(hovered)

    def _on_image_click(self, x, y):
        """Add new notch or delete existing."""
        
        # 1. Check Delete
        all_notches = self._get_current_notches_coords()
        for pt in all_notches:
            dist = np.hypot(pt[0] - x, pt[1] - y)
            if dist < 10:
                # Delete this point
                self.deleted_notches.add(pt)
                
                # If it was a manual point, remove from manual list too
                if pt[1] in self.manual_added_left: self.manual_added_left.remove(pt[1])
                if pt[1] in self.manual_added_right: self.manual_added_right.remove(pt[1])
                
                self._draw_overlay() # Redraw
                return

        # 2. Add New
        # Determine side based on midline
        if x < self.mid_x:
            # Left Side
            self.manual_added_left.append(y)
        else:
            # Right Side
            self.manual_added_right.append(y)
            
        self._draw_overlay()

    def _get_current_notches_coords(self):
        """Returns list of all currently visible (x,y) notches."""
        visible = []
        
        # 1. Left
        # Filter Deleted Auto
        for pt in self.auto_notches_left:
            if pt not in self.deleted_notches:
                visible.append(pt)
        
        # Add Manual Left
        for y in self.manual_added_left:
            # Find X on edge trace for this Y
            if not self.edge_points_left: continue
            
            # Simple lookup: find point with y == manual_y
            closest_pt = min(self.edge_points_left, key=lambda p: abs(p[1] - y))
            if abs(closest_pt[1] - y) < 5: # Tolerance
                pt = closest_pt
                if pt not in self.deleted_notches:
                     visible.append(pt)

        # 2. Right
        for pt in self.auto_notches_right:
             if pt not in self.deleted_notches:
                visible.append(pt)
                
        for y in self.manual_added_right:
            if not self.edge_points_right: continue
            closest_pt = min(self.edge_points_right, key=lambda p: abs(p[1] - y))
            if abs(closest_pt[1] - y) < 5:
                pt = closest_pt
                if pt not in self.deleted_notches:
                     visible.append(pt)
                     
        return visible
        
    def _draw_overlay(self):
        """Redraws the image with current notches, showing angles and lines."""
        if not self.image_data: return
        path = self.image_data[self.current_index]['path']
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        
        h, w, _ = image.shape
        
        # Check toggle state for "Light Markers"
        is_light_mode = self.light_markers_btn.isChecked()

        # Draw Midline (Only if NOT light mode)
        if not is_light_mode and self.substrate_y > 0:
             cv2.line(image, (self.mid_x, self.substrate_y), (self.mid_x, 0), (255, 255, 0), 1)

        # Draw full traces (Only if NOT light mode)
        if not is_light_mode:
            if self.edge_points_left:
                 pts = np.array(self.edge_points_left, dtype=np.int32)
                 cv2.polylines(image, [pts], False, (0, 255, 0), 1)
            if self.edge_points_right:
                 pts = np.array(self.edge_points_right, dtype=np.int32)
                 cv2.polylines(image, [pts], False, (0, 255, 0), 1)

        # Helper to draw a single notch
        def draw_notch_marker(pt, angle=None, pB=None, pC=None):
             px, py = int(pt[0]), int(pt[1])
             
             # Draw Segments (Cyan) - Only if NOT light mode
             if not is_light_mode:
                 if pB is not None:
                     cv2.line(image, pt, pB, (255, 255, 0), 1)
                 if pC is not None:
                     cv2.line(image, pt, pC, (255, 255, 0), 1)
             
             # Circle
             # Light mode: Hollow (thickness=2)
             # Normal mode: Filled (thickness=-1)
             thickness = 2 if is_light_mode else -1
             
             cv2.circle(image, (px, py), 3, (0, 0, 255), thickness)
             
             # Angle Text (Always show)
             if angle is not None:
                 label = f"{angle:.1f}"
                 cv2.putText(image, label, (px + 10, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # 1. Draw Auto Notches (Left)
        for i, pt in enumerate(self.auto_notches_left):
             if pt not in self.deleted_notches:
                 # Retrieve metadata
                 if i < len(self.auto_notches_meta_left):
                     meta = self.auto_notches_meta_left[i]
                     # meta: (x, y, angle, pB, pC)
                     draw_notch_marker((meta[0], meta[1]), meta[2], meta[3], meta[4])
                 else:
                     draw_notch_marker(pt) # Fallback

        # 2. Draw Auto Notches (Right)
        for i, pt in enumerate(self.auto_notches_right):
             if pt not in self.deleted_notches:
                 if i < len(self.auto_notches_meta_right):
                     meta = self.auto_notches_meta_right[i]
                     draw_notch_marker((meta[0], meta[1]), meta[2], meta[3], meta[4])
                 else:
                     draw_notch_marker(pt)

        # 3. Draw Manual Notches
        # For manual notches, we calculate angle on the fly if possible
        try:
            step = int(self.step_input.text())
        except:
            step = 25

        for y in self.manual_added_left:
            if not self.edge_points_left: continue
            # Find closest point
            closest_idx = -1
            min_diff = float('inf')
            for i, p in enumerate(self.edge_points_left):
                diff = abs(p[1] - y)
                if diff < min_diff:
                    min_diff = diff
                    closest_idx = i
            
            if closest_idx != -1 and min_diff < 5:
                pt = self.edge_points_left[closest_idx]
                if pt not in self.deleted_notches:
                     # Calculate neighbors for angle visualization
                     idx_B = min(len(self.edge_points_left)-1, closest_idx + step)
                     idx_C = max(0, closest_idx - step)
                     pA = pt
                     pB = self.edge_points_left[idx_B]
                     pC = self.edge_points_left[idx_C]
                     angle = self._calculate_angle(pA, pB, pC)
                     draw_notch_marker(pt, angle, pB, pC)

        for y in self.manual_added_right:
            if not self.edge_points_right: continue
            closest_idx = -1
            min_diff = float('inf')
            for i, p in enumerate(self.edge_points_right):
                diff = abs(p[1] - y)
                if diff < min_diff:
                    min_diff = diff
                    closest_idx = i
            
            if closest_idx != -1 and min_diff < 5:
                pt = self.edge_points_right[closest_idx]
                if pt not in self.deleted_notches:
                     idx_B = min(len(self.edge_points_right)-1, closest_idx + step)
                     idx_C = max(0, closest_idx - step)
                     pA = pt
                     pB = self.edge_points_right[idx_B]
                     pC = self.edge_points_right[idx_C]
                     angle = self._calculate_angle(pA, pB, pC)
                     draw_notch_marker(pt, angle, pB, pC)
            
        # Update Display
        h, w, ch = image.shape
        bytes_per_line = ch * w
        qt_img = QImage(image.data, w, h, bytes_per_line, QImage.Format_BGR888).rgbSwapped()
        self.main_image_label.setPixmap(QPixmap.fromImage(qt_img))


    # =================================================================
    # ANALYSIS LOGIC
    # =================================================================
    def _calculate_angle(self, pA, pB, pC):
        v1 = (pB[0] - pA[0], pB[1] - pA[1])
        v2 = (pC[0] - pA[0], pC[1] - pA[1])
        dot = v1[0]*v2[0] + v1[1]*v2[1]
        mag1 = np.sqrt(v1[0]**2 + v1[1]**2)
        mag2 = np.sqrt(v2[0]**2 + v2[1]**2)
        if mag1 == 0 or mag2 == 0: return 180.0
        cos_theta = dot / (mag1 * mag2)
        cos_theta = max(-1.0, min(1.0, cos_theta))
        return np.degrees(np.arccos(cos_theta))

    def _run_notch_detection(self):
        if not self.image_data: return
        data = self.image_data[self.current_index]
        path = data.get('path')
        
        try:
            prominence_angle_val = float(self.prominence_angle_input.text())
            prominence_dist_val = float(self.prominence_dist_input.text())
            step_val = int(self.step_input.text())
            hide_bottom_val = self.hide_slider.value()
            hide_top_val = self.hide_top_slider.value()
            blur_val = self.blur_slider.value()
            
            show_right_graph = self.radio_right.isChecked()
            use_angle_method = self.radio_angle.isChecked()
        except ValueError:
            return

        # 1. Load & Preprocess
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        h, w = gray.shape
        roi_x = int((data.get("roi_x_pct", 0) / 100.0) * w)
        roi_y = int((data.get("roi_y_pct", 0) / 100.0) * h)
        roi_w = int((data.get("roi_w_pct", 30) / 100.0) * w)
        roi_h = int((data.get("roi_h_pct", 30) / 100.0) * h)
        cv2.rectangle(gray, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 0, 0), -1)

        # Gaussian Blur
        if blur_val > 0:
            k = blur_val * 2 + 1
            gray = cv2.GaussianBlur(gray, (k, k), 0)

        # Contrast & Otsu
        clip_limit = data.get("contrast_clip_limit", 1.0)
        if clip_limit > 0.1:
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        num, labels, stats, _ = cv2.connectedComponentsWithStats(otsu, connectivity=8)
        if num > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            cleaned = np.zeros_like(otsu)
            cleaned[labels == largest_label] = 255
        else:
            cleaned = otsu.copy()

        # 2. Midline
        left_trace = data.get('substrate_trace_left')
        right_trace = data.get('substrate_trace_right')
        left_idx = data.get('substrate_left_endpoint_index', -1)
        right_idx = data.get('substrate_right_endpoint_index', -1)
        
        if not left_trace or not right_trace: return

        if left_idx == -1: left_idx = len(left_trace) - 1
        if right_idx == -1: right_idx = len(right_trace) - 1
        
        p_left = left_trace[left_idx]
        p_right = right_trace[right_idx]
        
        self.mid_x = int((p_left[0] + p_right[0]) / 2)
        self.substrate_y = min(p_left[1], p_right[1])
        
        # 3. Collect Edge Points (BOTH SIDES)
        self.edge_points_left = []
        self.edge_points_right = []
        
        heights = [] 
        scan_start_y = min(h - 1, self.substrate_y + 4)
        
        for y in range(scan_start_y, 0, -1):
            row = cleaned[y, :]
            white_pixels = np.where(row == 255)[0]
            
            if len(white_pixels) > 0:
                l = white_pixels[0]
                r = white_pixels[-1]
                
                dist_l = self.mid_x - l
                dist_r = r - self.mid_x
                
                # Check validity
                if dist_l > 0 and dist_r > 0:
                    height = self.substrate_y - y
                    # Filter (Apply hide bottom here)
                    if height > hide_bottom_val: 
                         self.edge_points_left.append((l, y))
                         self.edge_points_right.append((r, y))
                         heights.append(height)
        
        if len(heights) < 25: return 

        # Filter Top
        y_arr = np.array(heights)
        max_h = np.max(y_arr)
        limit = max_h - hide_top_val
        
        valid_mask = y_arr < limit
        y_arr = y_arr[valid_mask]
        
        # Apply mask to lists
        arr_pts_L = np.array(self.edge_points_left)[valid_mask]
        arr_pts_R = np.array(self.edge_points_right)[valid_mask]
        
        self.edge_points_left = [tuple(p) for p in arr_pts_L]
        self.edge_points_right = [tuple(p) for p in arr_pts_R]
        
        # 4. Analyze Both Sides
        self.auto_notches_left, self.auto_notches_meta_left = self._analyze_edge(y_arr, self.edge_points_left, self.mid_x, False, step_val, prominence_angle_val, prominence_dist_val, use_angle_method)
        self.auto_notches_right, self.auto_notches_meta_right = self._analyze_edge(y_arr, self.edge_points_right, self.mid_x, True, step_val, prominence_angle_val, prominence_dist_val, use_angle_method)
        
        # 5. Draw
        self._draw_overlay()
        
        # 6. Plot Graphs
        if show_right_graph:
            self._generate_plots(y_arr, self.edge_points_right, True, step_val)
        else:
            self._generate_plots(y_arr, self.edge_points_left, False, step_val)

    def _analyze_edge(self, y_arr, edge_points, mid_x, is_right, step, prom_angle, prom_dist, use_angle):
        """Returns (list of notch points, list of metadata tuples)"""
        if len(edge_points) < 5: return [], []
        
        angles = []
        dists = []
        valid_indices = []
        meta_list = [] # Temporary storage
        
        for i in range(len(edge_points)):
            if i - step < 0 or i + step >= len(edge_points):
                continue
            
            pA = edge_points[i]
            pB = edge_points[i + step]
            pC = edge_points[i - step]
            
            angle = self._calculate_angle(pA, pB, pC)
            
            if is_right:
                dist = pA[0] - mid_x
            else:
                dist = mid_x - pA[0]
                
            angles.append(angle)
            dists.append(dist)
            valid_indices.append(i)
            # Store geometric data for potential notch
            meta_list.append((pA[0], pA[1], angle, pB, pC))
            
        if not valid_indices: return [], []
        
        angles = np.array(angles)
        dists = np.array(dists)
        
        if use_angle:
            peaks, _ = find_peaks(-angles, prominence=prom_angle, distance=10)
        else:
            peaks, _ = find_peaks(-dists, prominence=prom_dist, distance=10)
            
        notches = []
        notches_meta = []
        for idx in peaks:
            real_idx = valid_indices[idx]
            notches.append(edge_points[real_idx])
            # idx maps to meta_list
            notches_meta.append(meta_list[idx])
            
        return notches, notches_meta

    def _generate_plots(self, y_arr, edge_points, is_right, step):
        # Re-calc metrics just for plotting
        if len(edge_points) <= 2*step: return
        
        y_valid = y_arr[step:-step]
        dists = []
        angles = []
        
        valid_indices_for_plot = []
        
        for i in range(len(edge_points)):
             if i - step < 0 or i + step >= len(edge_points):
                continue
             
             pA = edge_points[i]
             pB = edge_points[i+step]
             pC = edge_points[i-step]
            
             angles.append(self._calculate_angle(pA, pB, pC))
             if is_right:
                dists.append(pA[0] - self.mid_x)
             else:
                dists.append(self.mid_x - pA[0])
                
             valid_indices_for_plot.append(i)
        
        y_plot = y_arr[valid_indices_for_plot]
        
        self.figure.clear()
        edge_name = "Right" if is_right else "Left"
        
        ax1 = self.figure.add_subplot(211)
        ax1.plot(y_plot, dists, 'b-')
        ax1.set_title(f"{edge_name} Distance")
        ax1.grid(True)
        
        ax2 = self.figure.add_subplot(212, sharex=ax1)
        ax2.plot(y_plot, angles, 'm-')
        ax2.set_title("Angle")
        ax2.grid(True)
        
        self.figure.tight_layout()
        self.canvas.draw()

    # =================================================================
    # INTERACTIVITY
    # =================================================================
    def _on_graph_hover(self, event):
        if event.inaxes:
            height = event.xdata
            if height is not None and self.substrate_y > 0:
                self.main_image_label.set_highlight_line(self.substrate_y - height)
        else:
            self.main_image_label.set_highlight_line(None)

    def next_image(self):
        if not self.image_data: return
        if self.current_index < len(self.image_data) - 1:
            self.current_index += 1
            self.display_image()
            self.auto_notches_left = []
            self.auto_notches_right = []
            self.auto_notches_meta_left = []
            self.auto_notches_meta_right = []
            self.manual_added_left = []
            self.manual_added_right = []
            self.deleted_notches = set()
            self._run_notch_detection() 

    def prev_image(self):
        if not self.image_data: return
        if self.current_index > 0:
            self.current_index -= 1
            self.display_image()
            self.auto_notches_left = []
            self.auto_notches_right = []
            self.auto_notches_meta_left = []
            self.auto_notches_meta_right = []
            self.manual_added_left = []
            self.manual_added_right = []
            self.deleted_notches = set()
            self._run_notch_detection()
