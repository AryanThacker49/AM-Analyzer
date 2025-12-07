import sys
import os
import cv2
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QHBoxLayout, QFrame, QPushButton, 
    QGridLayout, QLineEdit, QSizePolicy, QApplication, QSlider, QGroupBox,
    QComboBox, QCheckBox, QProgressDialog, QRadioButton
)
from PySide6.QtGui import QPixmap, Qt, QImage, QPainter, QPen, QColor, QPainterPath
from PySide6 import QtCore, QtGui

# =====================================================================
# HELPER CLASS: ZoomLabel (Replaces AspectRatioLabel)
# =====================================================================
class ZoomLabel(QLabel):
    endpointMoved = QtCore.Signal(str, int) # side, index
    pointAdded = QtCore.Signal(tuple) # (x, y)

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setMinimumSize(1, 1)
        self.setAlignment(Qt.AlignCenter)
        self._original_pixmap = None
        self._scaled_pixmap = QPixmap()
        
        self.setMouseTracking(True)
        
        self.magnify_data = None 
        
        self.show_always_on_left_loupe = False
        self.show_always_on_right_loupe = False
        
        self.left_trace = []
        self.right_trace = []
        self.left_idx = -1
        self.right_idx = -1
        self.orig_left_len = 0
        self.orig_right_len = 0
        
        self.manual_points = []
        self.full_manual_trace = [] 
        
        self.click_to_jump_enabled = False
        self.manual_point_mode_enabled = False
        
        self.click_mode = "auto"

    def setPixmap(self, pixmap):
        self._original_pixmap = pixmap
        self._update_pixmap()

    def resizeEvent(self, event):
        self._update_pixmap()
        super().resizeEvent(event)

    def _update_pixmap(self):
        if self._original_pixmap and not self._original_pixmap.isNull():
            self._scaled_pixmap = self._original_pixmap.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            super().setPixmap(self._scaled_pixmap)
            
    def show_magnifier(self, original_pixmap_point):
        """Shows the "active" magnifier."""
        self.magnify_data = {"orig_pt": original_pixmap_point}
        self.update() 

    def hide_magnifier(self):
        """Hides the "active" magnifier."""
        if self.magnify_data:
            self.magnify_data = None
            self.update() 
            
    def set_always_on_loupes(self, show_left, show_right):
        """NEW: Toggles the "passive" always-on magnifiers."""
        self.show_always_on_left_loupe = show_left
        self.show_always_on_right_loupe = show_right
        self.update()

    def set_auto_overlays(self, left_trace, right_trace, left_idx, right_idx, orig_left_len=0, orig_right_len=0):
        self.left_trace = left_trace if left_trace else []
        self.right_trace = right_trace if right_trace else []
        self.left_idx = left_idx
        self.right_idx = right_idx
        self.orig_left_len = orig_left_len if orig_left_len > 0 else len(self.left_trace)
        self.orig_right_len = orig_right_len if orig_right_len > 0 else len(self.right_trace)
        
        self.manual_points = []
        self.full_manual_trace = []
        
        self.update()
        
    def set_manual_overlays(self, points):
        self.manual_points = points if points else []
        self._generate_full_manual_trace()
        
        self.left_trace = []
        self.right_trace = []
        self.left_idx = -1
        self.right_idx = -1
        
        self.update()

    def _generate_full_manual_trace(self):
        if not self.manual_points:
            self.full_manual_trace = []
            return

        points = sorted(self.manual_points, key=lambda p: p[0])
        
        if not points or self._original_pixmap is None:
            self.full_manual_trace = []
            return
            
        w = self._original_pixmap.width()
        
        leftmost_pt = points[0]
        rightmost_pt = points[-1]
        
        edge_left_point = (0, leftmost_pt[1])
        edge_right_point = (w - 1, rightmost_pt[1])
        
        self.full_manual_trace = []
        if leftmost_pt[0] > 0:
            self.full_manual_trace.append(edge_left_point)
            
        self.full_manual_trace.extend(points)
        
        if rightmost_pt[0] < w - 1:
            self.full_manual_trace.append(edge_right_point)


    def set_click_to_jump_enabled(self, enabled):
        self.click_to_jump_enabled = enabled
        self._update_cursor()
        if not enabled: 
            self.hide_magnifier()
        
    def set_manual_point_mode_enabled(self, enabled):
        self.manual_point_mode_enabled = enabled
        self._update_cursor()
        
    def set_click_mode(self, mode):
        self.click_mode = mode
        
    def _update_cursor(self):
        if self.manual_point_mode_enabled or self.click_to_jump_enabled:
            self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

    def _map_orig_to_widget(self, orig_pt):
        if self._original_pixmap is None or self._original_pixmap.isNull() or not orig_pt:
            return QtCore.QPoint(0, 0)
        
        scaled_w = self._scaled_pixmap.width()
        scaled_h = self._scaled_pixmap.height()
        offset_x = (self.width() - scaled_w) / 2
        offset_y = (self.height() - scaled_h) / 2
        
        if self._original_pixmap.width() == 0: return QtCore.QPoint(0, 0)
        scale_factor = scaled_w / self._original_pixmap.width()
        
        widget_x = (orig_pt[0] * scale_factor) + offset_x
        widget_y = (orig_pt[1] * scale_factor) + offset_y
        
        return QtCore.QPoint(int(widget_x), int(widget_y))

    def _map_widget_to_orig(self, widget_pt):
        if self._original_pixmap is None or self._original_pixmap.isNull():
            return None

        scaled_w = self._scaled_pixmap.width()
        scaled_h = self._scaled_pixmap.height()
        offset_x = (self.width() - scaled_w) / 2
        offset_y = (self.height() - scaled_h) / 2
        
        if (widget_pt.x() < offset_x or widget_pt.x() > offset_x + scaled_w or
            widget_pt.y() < offset_y or widget_pt.y() > offset_y + scaled_h):
            return None
            
        if scaled_w == 0: return None
        scale_factor = self._original_pixmap.width() / scaled_w
        
        orig_x = (widget_pt.x() - offset_x) * scale_factor
        orig_y = (widget_pt.y() - offset_y) * scale_factor
        
        return (int(orig_x), int(orig_y))

    def _find_nearest_point(self, click_pos_orig):
        if not self.left_trace and not self.right_trace:
            return None, None, -1
            
        click_pt = np.array(click_pos_orig)
        if self._original_pixmap is None: # Safety check
            return None, None, -1
            
        img_width = self._original_pixmap.width()
        
        is_left_half = click_pos_orig[0] < (img_width / 2)
        find_left = (self.click_mode == "left") or (self.click_mode == "auto" and is_left_half)
        find_right = (self.click_mode == "right") or (self.click_mode == "auto" and not is_left_half)

        min_dist_left = float('inf')
        best_idx_left = -1
        if find_left and self.left_trace:
            distances_left = np.linalg.norm(np.array(self.left_trace) - click_pt, axis=1)
            min_dist_left = distances_left.min()
            best_idx_left = distances_left.argmin()

        min_dist_right = float('inf')
        best_idx_right = -1
        if find_right and self.right_trace:
            distances_right = np.linalg.norm(np.array(self.right_trace) - click_pt, axis=1)
            min_dist_right = distances_right.min()
            best_idx_right = distances_right.argmin()
            
        if find_left and (min_dist_left <= min_dist_right):
            return self.left_trace[best_idx_left], "left", best_idx_left
        elif find_right and (min_dist_right < min_dist_left):
            return self.right_trace[best_idx_right], "right", best_idx_right
            
        return None, None, -1

    def mouseMoveEvent(self, event):
        """Show magnifier when in manual or click-to-jump mode."""
        click_pos_orig = self._map_widget_to_orig(event.position().toPoint())
        if not click_pos_orig:
            self.hide_magnifier()
            super().mouseMoveEvent(event)
            return

        if self.manual_point_mode_enabled:
            self.show_magnifier(click_pos_orig)
            
        elif self.click_to_jump_enabled:
            target_pt, _, _ = self._find_nearest_point(click_pos_orig)
            if target_pt:
                self.show_magnifier(target_pt)
            else:
                self.hide_magnifier()
        
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        """Hide magnifier when mouse leaves the widget."""
        self.hide_magnifier()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        """Handle user clicking on the image."""
        click_pos_orig = self._map_widget_to_orig(event.position().toPoint())
        if click_pos_orig is None:
            super().mousePressEvent(event)
            return

        if self.manual_point_mode_enabled:
            self.pointAdded.emit(click_pos_orig)
            return

        if self.click_to_jump_enabled and (self.left_trace or self.right_trace):
            target_pt, side, index = self._find_nearest_point(click_pos_orig)
                
            if side == "left":
                self.left_idx = index
                self.endpointMoved.emit("left", index)
                point_on_line = self.left_trace[index] 
            elif side == "right":
                self.right_idx = index
                self.endpointMoved.emit("right", index)
                point_on_line = self.right_trace[index] 
            
            self.update() 
            return

        super().mousePressEvent(event)
        
    def mouseReleaseEvent(self, event):
        # Don't hide magnifier if it's an always-on mode
        if not self.manual_point_mode_enabled and not self.click_to_jump_enabled:
            self.hide_magnifier()
        super().mouseReleaseEvent(event)

    def _draw_loupe(self, painter, orig_pt, offset_x=0):
        """Helper function to draw one magnifier."""
        if not orig_pt: # Safety check
            return
            
        widget_pt = self._map_orig_to_widget(orig_pt)
        
        LOUPE_SIZE = 120
        ZOOM_FACTOR = 3
        
        loupe_rect_widget = QtCore.QRect(0, 0, LOUPE_SIZE, LOUPE_SIZE)
        center_offset = QtCore.QPoint(offset_x, -LOUPE_SIZE // 2 - 10)
        loupe_rect_widget.moveCenter(widget_pt + center_offset)

        SOURCE_SIZE = int(LOUPE_SIZE / ZOOM_FACTOR)
        source_rect_orig = QtCore.QRect(0, 0, SOURCE_SIZE, SOURCE_SIZE)
        source_rect_orig.moveCenter(QtCore.QPoint(orig_pt[0], orig_pt[1]))
        
        path = QPainterPath()
        path.addEllipse(loupe_rect_widget)
        
        painter.save()
        painter.setClipPath(path)
        
        painter.drawPixmap(loupe_rect_widget, self._original_pixmap, source_rect_orig)
        
        painter.restore() 
        
        painter.setPen(QPen(QtCore.Qt.gray, 2))
        painter.drawEllipse(loupe_rect_widget)
        
        painter.setPen(QPen(QtCore.Qt.red, 1))
        c_x = loupe_rect_widget.center().x()
        c_y = loupe_rect_widget.center().y()
        painter.drawLine(c_x, loupe_rect_widget.top(), c_x, loupe_rect_widget.bottom())
        painter.drawLine(loupe_rect_widget.left(), c_y, loupe_rect_widget.right(), c_y)

    def paintEvent(self, event):
        super().paintEvent(event)
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # --- Draw Overlays ---
        left_idx = -1
        right_idx = -1
        
        if self.manual_point_mode_enabled:
            if self.full_manual_trace:
                scaled_manual_trace = [self._map_orig_to_widget(pt) for pt in self.full_manual_trace]
                painter.setPen(QPen(QColor(255, 0, 255), 2)) # Magenta
                painter.drawPolyline(scaled_manual_trace)
                
                painter.setPen(QPen(QColor(255, 0, 255, 150), 6)) 
                for pt in self.manual_points:
                    painter.drawPoint(self._map_orig_to_widget(pt))

        elif self.left_trace and self.right_trace:
            left_idx = self.left_idx if self.left_idx != -1 else self.orig_left_len - 1
            right_idx = self.right_idx if self.right_idx != -1 else self.orig_right_len - 1
            
            # Clamp indices
            if self.left_trace:
                left_idx = max(0, min(len(self.left_trace) - 1, left_idx))
            if self.right_trace:
                right_idx = max(0, min(len(self.right_trace) - 1, right_idx))
            
            scaled_left_trace = [self._map_orig_to_widget(pt) for pt in self.left_trace]
            scaled_right_trace = [self._map_orig_to_widget(pt) for pt in self.right_trace]

            if not scaled_left_trace or not scaled_right_trace:
                painter.end()
                return

            painter.setPen(QPen(QColor(0, 255, 0), 2)) # Green
            painter.drawPolyline(scaled_left_trace[:self.orig_left_len])
            painter.drawPolyline(scaled_right_trace[:self.orig_right_len])

            painter.setPen(QPen(QColor(100, 100, 100), 2, Qt.DashLine)) # Gray, dashed
            painter.drawPolyline(scaled_left_trace[self.orig_left_len-1:])
            painter.drawPolyline(scaled_right_trace[self.orig_right_len-1:])

            pt_left = scaled_left_trace[left_idx]
            pt_right = scaled_right_trace[right_idx]

            painter.setPen(QPen(QColor(255, 0, 0), 2)) # Red
            painter.drawLine(pt_left, pt_right)
            
            painter.setPen(QPen(QColor(0, 255, 255), 2)) # Yellow
            painter.drawEllipse(pt_left, 8, 8)
            painter.setPen(QPen(QColor(255, 0, 255), 2)) # Magenta
            painter.drawEllipse(pt_right, 8, 8)
        
        # --- Draw Magnifier(s) ---
        if self._original_pixmap is None:
            painter.end()
            return
            
        if self.magnify_data is not None:
            # "Active" magnifier (mouse follow, click, or button press)
            self._draw_loupe(painter, self.magnify_data["orig_pt"])
        
        else:
            # No active loupe, draw passive ones
            if self.show_always_on_left_loupe and self.left_trace and left_idx < len(self.left_trace):
                pt_left = self.left_trace[left_idx]
                self._draw_loupe(painter, pt_left, offset_x=-70) # Offset left
            
            if self.show_always_on_right_loupe and self.right_trace and right_idx < len(self.right_trace):
                pt_right = self.right_trace[right_idx]
                self._draw_loupe(painter, pt_right, offset_x=70) # Offset right

        painter.end()


# =====================================================================
# MAIN SCREEN CLASS
# =====================================================================
class SubstrateScreen(QWidget):

    goPrev = QtCore.Signal()
    goNext = QtCore.Signal()
    
    def __init__(self, controller=None):
        super().__init__()
        self.controller = controller
        
        self.image_data = [] 
        self.current_index = 0

        self.jump_map = {
            "Large (20)": 20,
            "Fine (5)": 5,
            "Very Fine (1)": 1
        }

        self._build_ui()

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        title = QLabel("Substrate Removal")
        title.setStyleSheet("font-size: 24px; font-weight: bold;")
        main_layout.addWidget(title)

        content_layout = QHBoxLayout()
        content_layout.setSpacing(20)

        # -------- Left: Image Preview --------
        image_frame = QFrame()
        image_layout = QVBoxLayout(image_frame)
        
        self.main_image_label = ZoomLabel("Image with Substrate Line")
        self.main_image_label.setStyleSheet("border: 2px solid #0B3C5D; background: #111;")
        self.main_image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        image_layout.addWidget(self.main_image_label, stretch=1)
        
        nav_images = QHBoxLayout()
        self.prev_img_btn = QPushButton("Prev Image")
        self.next_img_btn = QPushButton("Next Image")
        self.prev_img_btn.clicked.connect(self.prev_image)
        self.next_img_btn.clicked.connect(self.next_image)
        nav_images.addWidget(self.prev_img_btn)
        nav_images.addWidget(self.next_img_btn)
        image_layout.addLayout(nav_images)

        content_layout.addWidget(image_frame, stretch=3)

        # -------- Right: Controls --------
        right_frame = QFrame()
        right_frame.setMaximumWidth(350)
        right_layout = QVBoxLayout(right_frame)
        right_layout.setSpacing(15)
        right_layout.setAlignment(Qt.AlignTop)
        
        # --- Auto-Detect Group ---
        auto_group = QGroupBox("Auto-Detection")
        auto_layout = QVBoxLayout()
        self.recalc_btn = QPushButton("Auto-Detect Substrate")
        self.recalc_btn.clicked.connect(self._run_substrate_detection_current)
        auto_layout.addWidget(self.recalc_btn)
        
        self.auto_detect_all_btn = QPushButton("Auto-Detect in All Images")
        self.auto_detect_all_btn.clicked.connect(self._run_substrate_detection_all)
        auto_layout.addWidget(self.auto_detect_all_btn)
        
        self.edm_btn = QPushButton("Auto-Detect (EDM Cut)")
        self.edm_btn.clicked.connect(self._run_edm_detection)
        auto_layout.addWidget(self.edm_btn)
        
        # --- Skip Button ---
        self.skip_btn = QPushButton("No Substrate (Use Bottom)")
        self.skip_btn.clicked.connect(self._on_skip_detection)
        auto_layout.addWidget(self.skip_btn)
        
        auto_group.setLayout(auto_layout)
        right_layout.addWidget(auto_group)

        
        # --- Manual Endpoint Adjustment Group ---
        self.adjust_group = QGroupBox("Manual Endpoint Adjustment")
        adjust_layout = QVBoxLayout()
        
        self.click_toggle = QCheckBox("Enable Click-to-Jump")
        # --- FIX: Set Click-to-Jump as default ---
        self.click_toggle.setChecked(True) 
        adjust_layout.addWidget(self.click_toggle)
        
        self.click_mode_group = QWidget()
        click_mode_layout = QHBoxLayout()
        click_mode_layout.setContentsMargins(0, 5, 0, 5)
        self.radio_auto = QRadioButton("Auto")
        self.radio_left = QRadioButton("Left")
        self.radio_right = QRadioButton("Right")
        self.radio_auto.setChecked(True)
        click_mode_layout.addWidget(self.radio_auto)
        click_mode_layout.addWidget(self.radio_left)
        click_mode_layout.addWidget(self.radio_right)
        self.click_mode_group.setLayout(click_mode_layout)
        self.click_mode_group.setVisible(True) 
        adjust_layout.addWidget(self.click_mode_group)
        
        jump_layout = QHBoxLayout()
        jump_layout.addWidget(QLabel("Jump Size:"))
        self.jump_size_combo = QComboBox()
        self.jump_size_combo.addItems(self.jump_map.keys())
        jump_layout.addWidget(self.jump_size_combo)
        adjust_layout.addLayout(jump_layout)
        
        left_layout = QHBoxLayout()
        left_layout.addWidget(QLabel("Left Point:"))
        left_layout.addStretch()
        btn_left_L = QPushButton("< (Out)")
        btn_left_R = QPushButton("> (In)")
        btn_left_L.setFixedSize(60, 40)
        btn_left_R.setFixedSize(60, 40)
        btn_left_L.setAutoRepeat(True) 
        btn_left_R.setAutoRepeat(True)
        left_layout.addWidget(btn_left_L)
        left_layout.addWidget(btn_left_R)
        adjust_layout.addLayout(left_layout)
        
        right_layout_btns = QHBoxLayout()
        right_layout_btns.addWidget(QLabel("Right Point:"))
        right_layout_btns.addStretch()
        btn_right_L = QPushButton("< (In)")
        btn_right_R = QPushButton("> (Out)")
        btn_right_L.setFixedSize(60, 40)
        btn_right_R.setFixedSize(60, 40)
        btn_right_L.setAutoRepeat(True) 
        btn_right_R.setAutoRepeat(True)
        right_layout_btns.addWidget(btn_right_L)
        right_layout_btns.addWidget(btn_right_R)
        adjust_layout.addLayout(right_layout_btns)

        self.adjust_group.setLayout(adjust_layout)
        right_layout.addWidget(self.adjust_group)
        
        # --- Manual Mode Group ---
        self.manual_group = QGroupBox("Manual Point Selection")
        manual_layout = QVBoxLayout()
        self.manual_toggle = QCheckBox("Enable Manual Point Selection")
        self.manual_toggle.setChecked(False)
        manual_layout.addWidget(self.manual_toggle)
        
        manual_btn_layout = QHBoxLayout()
        self.delete_last_manual_btn = QPushButton("Delete Last Point")
        self.delete_last_manual_btn.clicked.connect(self._on_delete_last_manual_point)
        manual_btn_layout.addWidget(self.delete_last_manual_btn)

        self.clear_manual_btn = QPushButton("Clear All")
        self.clear_manual_btn.clicked.connect(self._on_clear_manual_points)
        manual_btn_layout.addWidget(self.clear_manual_btn)
        manual_layout.addLayout(manual_btn_layout)

        
        self.manual_group.setLayout(manual_layout)
        right_layout.addWidget(self.manual_group)
        
        right_layout.addSpacing(20)

        # Apply Buttons
        self.apply_single_btn = QPushButton("Apply to Current")
        self.apply_single_btn.clicked.connect(self.save_settings_to_current)
        right_layout.addWidget(self.apply_single_btn)
        
        right_layout.addStretch()
        content_layout.addWidget(right_frame, stretch=1)
        
        main_layout.addLayout(content_layout, stretch=1)

        # --- Bottom navigation ---
        bottom = QHBoxLayout()
        self.prev_btn = QPushButton("Prev: Scale")
        self.next_btn = QPushButton("Next: Analysis Menu")
        self.prev_btn.setFixedHeight(50)
        self.next_btn.setFixedHeight(50)
        self.prev_btn.clicked.connect(self.goPrev.emit)
        self.next_btn.clicked.connect(self.goNext.emit)
        bottom.addWidget(self.prev_btn)
        bottom.addWidget(self.next_btn)
        main_layout.addLayout(bottom)
        
        # --- Connect Endpoint Buttons (Corrected Logic) ---
        btn_left_L.pressed.connect(lambda: self._on_move_endpoint("left", -1))
        btn_left_R.pressed.connect(lambda: self._on_move_endpoint("left", 1))
        btn_right_L.pressed.connect(lambda: self._on_move_endpoint("right", 1)) 
        btn_right_R.pressed.connect(lambda: self._on_move_endpoint("right", -1)) 
        
        btn_left_L.released.connect(self._on_move_endpoint_released)
        btn_left_R.released.connect(self._on_move_endpoint_released)
        btn_right_L.released.connect(self._on_move_endpoint_released)
        btn_right_R.released.connect(self._on_move_endpoint_released)
        
        self.click_toggle.toggled.connect(self._on_click_toggle)
        self.manual_toggle.toggled.connect(self._on_manual_toggle)
        
        self.radio_auto.toggled.connect(self._on_click_mode_changed)
        self.radio_left.toggled.connect(self._on_click_mode_changed)
        self.radio_right.toggled.connect(self._on_click_mode_changed)
        
        self.main_image_label.endpointMoved.connect(self._on_endpoint_jump)
        self.main_image_label.pointAdded.connect(self._on_manual_point_added)

    # =================================================================
    # OPENCV & ANALYSIS LOGIC
    # =================================================================
    
    def _run_substrate_detection_current(self):
        """Wrapper to run detection on just the current image."""
        if not self.image_data: return
        data = self.image_data[self.current_index]
        success = self._detect_substrate_for_data(data)
        if success:
            self._draw_image_with_overlay()
        else:
            self.main_image_label.setText("Error: Could not trace substrate.")

    def _run_substrate_detection_all(self):
        """Runs substrate detection for ALL images."""
        if not self.image_data: return
        
        progress = QProgressDialog("Running auto-detection...", "Cancel", 0, len(self.image_data), self)
        progress.setWindowTitle("Processing Images")
        progress.setWindowModality(Qt.WindowModal)
        progress.setValue(0)
        
        for i, data in enumerate(self.image_data):
            progress.setValue(i)
            progress.setLabelText(f"Processing image {i+1} of {len(self.image_data)}...")
            QApplication.processEvents() 

            if progress.wasCanceled():
                break

            self._detect_substrate_for_data(data)
            
        progress.setValue(len(self.image_data))
        self._draw_image_with_overlay()

    def _detect_substrate_for_data(self, data):
        """
        REVERTED to original Edge-In logic:
        1. Find starting points at edges (x=0, x=width).
        2. Trace inward until a large jump is detected.
        3. Interpolate between the two stop points.
        """
        path = data.get("path")
        image_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image_gray is None:
            print(f"Error loading image: {path}")
            return False

        h_img, w_img = image_gray.shape
        roi_x = int((data.get("roi_x_pct", 0) / 100.0) * w_img)
        roi_y = int((data.get("roi_y_pct", 0) / 100.0) * h_img)
        roi_w = int((data.get("roi_w_pct", 30) / 100.0) * w_img)
        roi_h = int((data.get("roi_h_pct", 30) / 100.0) * h_img)
        
        cv2.rectangle(image_gray, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 0, 0), -1)

        _, substrate_otsu_thresh = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        height, width = substrate_otsu_thresh.shape
        step = 5
        left_trace, right_trace = [], []
        
        try:
            # Left Trace (x=0 inward)
            x = 0
            for y in range(height):
                if substrate_otsu_thresh[y, x] == 255:
                    left_trace.append((x, y))
                    prev_y = y
                    break
            x = step
            while x < width // 2:
                white_pixels = np.where(substrate_otsu_thresh[:, x] == 255)[0]
                if len(white_pixels) == 0: break
                y = white_pixels[0]
                if y > prev_y - 100: # Jump detected
                    left_trace.append((x, y))
                    prev_y = y
                    x += step
                else: break
            
            # Right Trace (x=width inward)
            x = width - 1
            for y in range(height):
                if substrate_otsu_thresh[y, x] == 255:
                    right_trace.append((x, y))
                    prev_y = y
                    break
            x = width - 1 - step
            while x > width // 2:
                white_pixels = np.where(substrate_otsu_thresh[:, x] == 255)[0]
                if len(white_pixels) == 0: break
                y = white_pixels[0]
                if y > prev_y - 10: # Jump detected
                    right_trace.append((x, y))
                    prev_y = y
                    x -= step
                else: break
        except Exception as e:
            print(f"Boundary trace failed: {e}")
            return False

        if not left_trace or not right_trace:
            print("Could not trace substrate.")
            return False
            
        p_left = left_trace[-1]
        p_right = right_trace[-1]
        
        num_steps = int(np.linalg.norm(np.array(p_left) - np.array(p_right)))
        if num_steps < 2: 
             interface_trace = [p_left, p_right]
        else:
            x_coords = np.linspace(p_left[0], p_right[0], num_steps)
            y_coords = np.linspace(p_left[1], p_right[1], num_steps)
            interface_trace = list(zip(x_coords.astype(int), y_coords.astype(int)))
        
        full_left_path = left_trace + interface_trace[1:] 
        full_right_path = right_trace + list(reversed(interface_trace))[:-1] 
            
        data["substrate_trace_left"] = full_left_path
        data["substrate_trace_right"] = full_right_path
        data["substrate_left_endpoint_index"] = len(left_trace) - 1
        data["substrate_right_endpoint_index"] = len(right_trace) - 1
        data["substrate_orig_left_len"] = len(left_trace)
        data["substrate_orig_right_len"] = len(right_trace)
        data["substrate_is_manual"] = False 
        data["manual_substrate_points"] = [] 
        return True

    # --- EDM Cut Detection Logic (Unchanged) ---
    def _run_edm_detection(self):
        if not self.image_data: return
        data = self.image_data[self.current_index]
        path = data.get("path")
        
        image_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image_gray is None:
            self.main_image_label.setText("Error: Could not load image.")
            return
            
        # Blackout ROI first
        h_img, w_img = image_gray.shape
        roi_x = int((data.get("roi_x_pct", 0) / 100.0) * w_img)
        roi_y = int((data.get("roi_y_pct", 0) / 100.0) * h_img)
        roi_w = int((data.get("roi_w_pct", 30) / 100.0) * w_img)
        roi_h = int((data.get("roi_h_pct", 30) / 100.0) * h_img)
        cv2.rectangle(image_gray, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 0, 0), -1)

        # 1. Otsu and keep largest blob
        _, otsu = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(otsu, connectivity=8)
        
        if num_labels <= 1:
            self.main_image_label.setText("Error: No material found.")
            return
            
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        blob_mask = (labels == largest_label).astype(np.uint8) * 255
        
        # 2. Find midpoint of the blob
        bx = stats[largest_label, cv2.CC_STAT_LEFT]
        bw = stats[largest_label, cv2.CC_STAT_WIDTH]
        mid_x = bx + bw // 2
        
        # 3. Find start point (scan up from bottom at mid_x)
        # Scan column mid_x
        col = blob_mask[:, mid_x]
        white_pixels = np.where(col == 255)[0]
        if len(white_pixels) == 0:
             self.main_image_label.setText("Error: Could not find bottom at midpoint.")
             return
        start_y = white_pixels[-1] # Last white pixel (bottom-most)
        
        # 4. Trace Outward Pixel by Pixel with Lookback Check
        left_trace = []
        right_trace = []
        
        check_distance = 10 # Look back 10 pixels
        jump_limit = 5      # Stop if Y decreases (goes up) by 5 pixels
        
        # Trace Right
        curr_x = mid_x
        while curr_x < w_img:
            col = blob_mask[:, curr_x]
            white_pixels = np.where(col == 255)[0]
            
            if len(white_pixels) == 0: break 
            
            y = white_pixels[-1]
            right_trace.append((curr_x, y))
            
            if len(right_trace) > check_distance:
                prev_y = right_trace[-1 - check_distance][1]
                if (prev_y - y) >= jump_limit:
                    right_trace = right_trace[:-check_distance]
                    break
            
            curr_x += 1
            
        # Trace Left
        curr_x = mid_x - 1
        while curr_x >= 0:
            col = blob_mask[:, curr_x]
            white_pixels = np.where(col == 255)[0]
            
            if len(white_pixels) == 0: break
            
            y = white_pixels[-1]
            left_trace.append((curr_x, y))
            
            if len(left_trace) > check_distance:
                prev_y = left_trace[-1 - check_distance][1]
                if (prev_y - y) >= jump_limit:
                    left_trace = left_trace[:-check_distance]
                    break

            curr_x -= 1
            
        left_trace = left_trace[::-1]
        
        # --- FIX: Reverse right trace (Right -> Center to Center -> Right) ---
        # No wait, I constructed it Center -> Right. It is already correct for drawing.
        # But for consistency with "Right Endpoint is Index 0", 
        # the standard logic uses Right->Center (Outside->In).
        # So yes, I should reverse it.
        right_trace = right_trace[::-1]

        # --- FIX: Extend to edges ---
        if left_trace:
             leftmost_pt = left_trace[0]
             if leftmost_pt[0] > 0:
                 left_trace.insert(0, (0, leftmost_pt[1]))
                 
        if right_trace:
             rightmost_pt = right_trace[0] 
             if rightmost_pt[0] < w_img - 1:
                 right_trace.insert(0, (w_img - 1, rightmost_pt[1]))
        
        # Save to data
        data["substrate_trace_left"] = left_trace
        data["substrate_trace_right"] = right_trace
        data["substrate_left_endpoint_index"] = 1 
        data["substrate_right_endpoint_index"] = 1
        data["substrate_orig_left_len"] = len(left_trace)
        data["substrate_orig_right_len"] = len(right_trace)
        data["substrate_is_manual"] = False
        data["manual_substrate_points"] = []
        
        self._draw_image_with_overlay()

    # --- Skip Detection Logic (Flat Line) ---
    def _on_skip_detection(self):
        if not self.image_data: return
        data = self.image_data[self.current_index]
        path = data.get("path")
        
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None: return
        h, w = image.shape
        
        bottom_y = h - 1
        left_trace = [(0, bottom_y), (w-1, bottom_y)]
        right_trace = [(w-1, bottom_y), (0, bottom_y)]
        
        data["substrate_trace_left"] = left_trace
        data["substrate_trace_right"] = right_trace
        data["substrate_left_endpoint_index"] = 0
        data["substrate_right_endpoint_index"] = 0
        data["substrate_orig_left_len"] = len(left_trace)
        data["substrate_orig_right_len"] = len(right_trace)
        data["substrate_is_manual"] = False 
        data["manual_substrate_points"] = []
        
        self._draw_image_with_overlay()

    def _draw_image_with_overlay(self):
        """Draws the overlay and updates the image label"""
        if not self.image_data: return
        data = self.image_data[self.current_index]

        path = data.get("path")
        pixmap = QPixmap(path)
        if pixmap.isNull():
            self.main_image_label.setText("Error: Could not load image.")
            return
        
        self.main_image_label.setPixmap(pixmap)
        
        if data.get("substrate_is_manual", False):
            points = data.get("manual_substrate_points", [])
            self.main_image_label.set_manual_overlays(points)
        else:
            left_trace = data.get("substrate_trace_left")
            right_trace = data.get("substrate_trace_right")
            left_idx = data.get("substrate_left_endpoint_index", -1)
            right_idx = data.get("substrate_right_endpoint_index", -1)
            orig_left_len = data.get("substrate_orig_left_len", 0)
            orig_right_len = data.get("substrate_orig_right_len", 0)
            self.main_image_label.set_auto_overlays(left_trace, right_trace, left_idx, right_idx, orig_left_len, orig_right_len)

        
    # =================================================================
    # Endpoint Movement
    # =================================================================
    
    def _on_click_toggle(self, is_checked):
        """Toggles the Click-to-Jump mode."""
        self.main_image_label.set_click_to_jump_enabled(is_checked)
        self.click_mode_group.setVisible(is_checked) 
        
        if is_checked:
            self.manual_toggle.setChecked(False) 
            self.main_image_label.set_always_on_loupes(False, False)
        else:
            self.main_image_label.hide_magnifier()
            self.main_image_label.set_always_on_loupes(True, True)

    def _on_click_mode_changed(self):
        """Updates the label's click mode when a radio button is toggled."""
        if self.radio_left.isChecked():
            self.main_image_label.set_click_mode("left")
        elif self.radio_right.isChecked():
            self.main_image_label.set_click_mode("right")
        else:
            self.main_image_label.set_click_mode("auto")

    def _on_manual_toggle(self, is_checked):
        """Toggles the Manual Point Selection mode."""
        self.main_image_label.set_manual_point_mode_enabled(is_checked)
        self.recalc_btn.setEnabled(not is_checked)
        self.auto_detect_all_btn.setEnabled(not is_checked)
        self.edm_btn.setEnabled(not is_checked)
        self.skip_btn.setEnabled(not is_checked) 
        
        self.adjust_group.setEnabled(not is_checked)
        self.clear_manual_btn.setEnabled(is_checked)
        self.delete_last_manual_btn.setEnabled(is_checked)
        
        if is_checked:
            self.click_toggle.setChecked(False) 
            self.main_image_label.set_always_on_loupes(False, False)
        
        if not is_checked:
            self.main_image_label.hide_magnifier()
            if not self.click_toggle.isChecked():
                self.main_image_label.set_always_on_loupes(True, True)
            
        self._draw_image_with_overlay()


    def _on_endpoint_jump(self, side, index):
        """Called by ZoomLabel when user clicks to jump endpoint."""
        if not self.image_data: return
        data = self.image_data[self.current_index]
        
        if side == "left":
            data["substrate_left_endpoint_index"] = index
        else:
            data["substrate_right_endpoint_index"] = index
            
        self.save_settings_to_current()
    
    def _on_move_endpoint(self, side, direction_sign):
        """Move the left or right endpoint index by one step."""
        if not self.image_data: return
        
        data = self.image_data[self.current_index]
        left_trace = data.get("substrate_trace_left")
        right_trace = data.get("substrate_trace_right")
        
        if not left_trace or not right_trace:
            self.main_image_label.setText("Run Auto-Detect first.")
            return

        jump_text = self.jump_size_combo.currentText()
        jump_size = self.jump_map.get(jump_text, 1)
        direction = direction_sign * jump_size

        if side == "left":
            key = "substrate_left_endpoint_index"
            trace = left_trace
            orig_len_key = "substrate_orig_left_len"
        else: # side == "right"
            key = "substrate_right_endpoint_index"
            trace = right_trace
            orig_len_key = "substrate_orig_right_len"
            # --- FIX: Invert direction for the right trace ---
            direction = -direction
            
        idx = data.get(key, -1)
        if idx == -1: 
            idx = data.get(orig_len_key, len(trace)) - 1
            
        idx += direction 
        idx = max(0, min(len(trace) - 1, idx)) 
        
        data[key] = idx
        
        self.main_image_label.set_auto_overlays(left_trace, right_trace, 
                                           data["substrate_left_endpoint_index"], 
                                           data["substrate_right_endpoint_index"],
                                           data.get("substrate_orig_left_len", 0),
                                           data.get("substrate_orig_right_len", 0))
        
        point_to_magnify = trace[idx]
        self.main_image_label.show_magnifier(point_to_magnify)

    def _on_move_endpoint_released(self):
        """Hide the magnifier when the button is released."""
        self.main_image_label.hide_magnifier()
        self.save_settings_to_current()
        # --- FIX: Re-show always-on loupes ---
        if not self.click_toggle.isChecked():
            self.main_image_label.set_always_on_loupes(True, True)
        
    def _on_manual_point_added(self, point_coords):
        """Adds a new point from the user click to the manual list."""
        if not self.image_data: return
        data = self.image_data[self.current_index]
        
        points = data.get("manual_substrate_points", [])
        points.append(point_coords)
        points.sort(key=lambda p: p[0])
        
        data["manual_substrate_points"] = points
        data["substrate_is_manual"] = True 
        
        self._draw_image_with_overlay()
        
    def _on_delete_last_manual_point(self):
        """NEW: Deletes the last added manual point."""
        if not self.image_data: return
        data = self.image_data[self.current_index]
        
        points = data.get("manual_substrate_points", [])
        if points:
            points.pop() # Remove the last point
        
        data["manual_substrate_points"] = points
        
        self._draw_image_with_overlay()
        
    def _on_clear_manual_points(self):
        """Clears the manual points list."""
        if not self.image_data: return
        data = self.image_data[self.current_index]
        data["manual_substrate_points"] = []
        data["substrate_is_manual"] = True 
        
        self.main_image_label.set_manual_overlays([])
        self._draw_image_with_overlay()


    # =================================================================
    # DATA MODEL FUNCTIONS
    # =================================================================
    def load_image_data(self, image_data_list):
        self.image_data = image_data_list
        self.current_index = 0
        if not self.image_data:
            self.main_image_label.setText("No images loaded from main")
            return
        
        self.display_image(run_detection=False)
        self.update_nav_buttons()

    def display_image(self, run_detection=True):
        if not self.image_data: return
        data = self.image_data[self.current_index]

        # Set the state of the toggles based on loaded data
        is_manual = data.get("substrate_is_manual", False)
        self.manual_toggle.setChecked(is_manual)
        
        # --- FIX: Set Click-to-Jump default to ON ---
        self.click_toggle.setChecked(True)
        self._on_manual_toggle(is_manual) 
        self._on_click_toggle(True)
        
        self._draw_image_with_overlay() 
        
        # --- FIX: Adjust always-on loupe logic ---
        if not is_manual and not self.click_toggle.isChecked():
            self.main_image_label.set_always_on_loupes(True, True)
        else:
            self.main_image_label.set_always_on_loupes(False, False)

        if not is_manual and not data.get("substrate_trace_left") and run_detection:
            self._run_substrate_detection_current()
            
        self.update_nav_buttons()

    def save_settings_to_current(self):
        if not self.image_data: return
        data = self.image_data[self.current_index]
        
        if self.manual_toggle.isChecked():
            data["substrate_is_manual"] = True
            self.main_image_label._generate_full_manual_trace()
            full_trace = self.main_image_label.full_manual_trace
            
            if not full_trace:
                data["substrate_trace_left"] = []
                data["substrate_trace_right"] = []
                print("Saved empty manual substrate trace.")
                return

            data["substrate_trace_left"] = full_trace
            data["substrate_trace_right"] = list(reversed(full_trace))
            data["substrate_left_endpoint_index"] = 0
            data["substrate_right_endpoint_index"] = 0
            data["substrate_orig_left_len"] = len(full_trace)
            data["substrate_orig_right_len"] = len(full_trace)
            print(f"Saved MANUAL substrate trace for image {self.current_index}")
            
        else:
            data["substrate_is_manual"] = False
            if "substrate_trace_left" not in data or data["substrate_trace_left"] is None:
                print("Running detection before saving...")
                self._run_substrate_detection_current()
            
            print(f"Saved AUTO substrate trace for image {self.current_index}")

    def save_settings_to_all(self):
        # This button is gone
        pass


    # =================================================================
    # NAVIGATION
    # =================================================================
    def next_image(self):
        if not self.image_data: return
        if self.current_index < len(self.image_data) - 1:
            self.current_index += 1
            self.display_image(run_detection=True) 

    def prev_image(self):
        if not self.image_data: return
        if self.current_index > 0:
            self.current_index -= 1
            self.display_image(run_detection=False) 

    def update_nav_buttons(self):
        if not self.image_data:
            self.prev_img_btn.setEnabled(False)
            self.next_img_btn.setEnabled(False)
            return
        self.prev_img_btn.setEnabled(self.current_index > 0)
        self.next_img_btn.setEnabled(self.current_index < len(self.image_data) - 1)


# =================================================================
# TEST HARNESS
# =================================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    app.setStyle("Fusion")
    dark_palette = QtGui.QPalette()
    dark_palette.setColor(QtGui.QPalette.Window, QtGui.QColor(23, 23, 23))
    dark_palette.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
    dark_palette.setColor(QtGui.QPalette.Base, QtGui.QColor(30, 30, 30))
    dark_palette.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
    dark_palette.setColor(QtGui.QPalette.Button, QtGui.QColor(45, 45, 45))
    dark_palette.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white)
    app.setPalette(dark_palette)
    
    win = SubstrateScreen()
    
    # --- TEST HARNESS ---
    test_data = [
        {
            "path": "C:/path/to/your/image1.png", # <-- REPLACE WITH REAL PATH
            "microns_per_pixel": 2.57,
            "roi_x_pct": 0.0, "roi_y_pct": 70.0, "roi_w_pct": 30.0, "roi_h_pct": 30.0,
            "substrate_trace_left": None,
            "substrate_trace_right": None,
            "substrate_left_endpoint_index": -1,
            "substrate_right_endpoint_index": -1,
            "substrate_is_manual": False,
            "manual_substrate_points": []
        }
    ]
    valid_test_data = [d for d in test_data if os.path.exists(d["path"]) and d.get("microns_per_pixel")]
    if valid_test_data:
        win.load_image_data(valid_test_data)
    else:
        print("---")
        print("TEST HARNETT: Please edit substrate_screen.py")
        print("Add a real image path AND a 'microns_per_pixel' value to 'test_data'.")
        print("---")

    win.resize(1200, 800)
    win.show()
    sys.exit(app.exec())