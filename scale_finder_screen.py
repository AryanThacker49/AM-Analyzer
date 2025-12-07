import sys
import os
import re
import cv2
import numpy as np
import pytesseract
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QHBoxLayout, QFrame, QPushButton, 
    QGridLayout, QLineEdit, QSizePolicy, QApplication, QMessageBox, QDialog,
    QCheckBox, QGroupBox, QSlider
)
from PySide6.QtGui import QPixmap, Qt, QImage, QPainter, QPen, QColor, QBrush
from PySide6 import QtCore, QtGui

# =====================================================================
# CONFIGURATION
# =====================================================================
TESSERACT_PATH = r'C:/Users/aryan/OneDrive/Desktop/Al - Steel Research/Images/Tesseract-OCR/tesseract.exe'
if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# =====================================================================
# HELPER CLASS: AspectRatioLabel
# =====================================================================
class AspectRatioLabel(QLabel):
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setMinimumSize(1, 1)
        self.setAlignment(Qt.AlignCenter)
        self._original_pixmap = None

    def setPixmap(self, pixmap):
        self._original_pixmap = pixmap
        self._update_pixmap()

    def resizeEvent(self, event):
        self._update_pixmap()
        super().resizeEvent(event)

    def _update_pixmap(self):
        if self._original_pixmap and not self._original_pixmap.isNull():
            scaled = self._original_pixmap.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            super().setPixmap(scaled)

# =====================================================================
# HELPER CLASS: ROISelectionLabel
# =====================================================================
class ROISelectionLabel(QLabel):
    roiSelected = QtCore.Signal(tuple) 

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setMinimumSize(1, 1)
        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(True)
        self.setCursor(Qt.CrossCursor)
        
        self._original_pixmap = None
        self._scaled_pixmap = QPixmap()
        
        self.start_point = None
        self.end_point = None
        self.is_drawing = False
        self.current_roi_rect = None 

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

    def _get_image_offset(self):
        if self._scaled_pixmap.isNull(): return (0, 0)
        w_widgets = self.width()
        h_widgets = self.height()
        w_img = self._scaled_pixmap.width()
        h_img = self._scaled_pixmap.height()
        return ((w_widgets - w_img) // 2, (h_widgets - h_img) // 2)

    def _map_widget_to_orig(self, widget_pt):
        if self._original_pixmap is None: return None
        offset_x, offset_y = self._get_image_offset()
        w_scaled = self._scaled_pixmap.width()
        h_scaled = self._scaled_pixmap.height()
        
        x_rel = widget_pt.x() - offset_x
        y_rel = widget_pt.y() - offset_y
        x_rel = max(0, min(x_rel, w_scaled))
        y_rel = max(0, min(y_rel, h_scaled))
        
        if w_scaled == 0 or h_scaled == 0: return (0, 0)
        scale_x = self._original_pixmap.width() / w_scaled
        scale_y = self._original_pixmap.height() / h_scaled
        
        return (int(x_rel * scale_x), int(y_rel * scale_y))

    def _map_orig_to_widget(self, orig_x, orig_y, orig_w, orig_h):
        if self._original_pixmap is None or self._scaled_pixmap.isNull(): return None
        offset_x, offset_y = self._get_image_offset()
        w_scaled = self._scaled_pixmap.width()
        h_scaled = self._scaled_pixmap.height()
        scale_x = w_scaled / self._original_pixmap.width()
        scale_y = h_scaled / self._original_pixmap.height()
        
        x = int(orig_x * scale_x) + offset_x
        y = int(orig_y * scale_y) + offset_y
        w = int(orig_w * scale_x)
        h = int(orig_h * scale_y)
        return QtCore.QRect(x, y, w, h)

    def mousePressEvent(self, event):
        if self._original_pixmap:
            self.is_drawing = True
            self.start_point = event.position().toPoint()
            self.end_point = event.position().toPoint()
            self.update()

    def mouseMoveEvent(self, event):
        if self.is_drawing:
            self.end_point = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event):
        if self.is_drawing:
            self.is_drawing = False
            self.end_point = event.position().toPoint()
            p1 = self._map_widget_to_orig(self.start_point)
            p2 = self._map_widget_to_orig(self.end_point)
            if p1 and p2:
                x = min(p1[0], p2[0])
                y = min(p1[1], p2[1])
                w = abs(p1[0] - p2[0])
                h = abs(p1[1] - p2[1])
                if w > 5 and h > 5:
                    self.roiSelected.emit((x, y, w, h))
                    self.current_roi_rect = (x, y, w, h) 
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        if self.is_drawing and self.start_point and self.end_point:
            rect = QtCore.QRect(self.start_point, self.end_point).normalized()
            painter.setPen(QPen(QColor(0, 120, 215), 2, Qt.SolidLine))
            painter.setBrush(QBrush(QColor(0, 120, 215, 50))) 
            painter.drawRect(rect)
        elif self.current_roi_rect:
            x, y, w, h = self.current_roi_rect
            widget_rect = self._map_orig_to_widget(x, y, w, h)
            if widget_rect:
                painter.setPen(QPen(QColor(0, 255, 0), 2, Qt.SolidLine))
                painter.setBrush(QBrush(QColor(0, 255, 0, 30))) 
                painter.drawRect(widget_rect)
        painter.end()
        
    def set_current_roi(self, x, y, w, h):
        self.current_roi_rect = (x, y, w, h)
        self.update()

# =====================================================================
# MAIN SCREEN CLASS
# =====================================================================
class ScaleFinderScreen(QWidget):

    goPrev = QtCore.Signal()
    goNext = QtCore.Signal()
    moveROI = QtCore.Signal(str)
    
    def __init__(self, controller=None):
        super().__init__()
        self.controller = controller
        self.image_data = [] 
        self.current_index = 0
        self.current_microns_per_pixel = None
        self.current_detected_line_length_px = None
        self.roi_pixels = None 
        
        self.current_roi_x_pct = 0.0
        self.current_roi_y_pct = 70.0
        self.current_roi_w_pct = 30.0
        self.current_roi_h_pct = 30.0
        self._build_ui()

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        title = QLabel("Scale Finder")
        title.setStyleSheet("font-size: 24px; font-weight: bold;")
        main_layout.addWidget(title)

        center_layout = QHBoxLayout()
        center_layout.setSpacing(20)

        left_frame = QFrame()
        left_layout = QVBoxLayout(left_frame)
        left_layout.setContentsMargins(0, 0, 0, 0)
        instr_label = QLabel("1. Drag a box around the scale bar to auto-detect:")
        instr_label.setStyleSheet("color: #aaa; font-style: italic;")
        left_layout.addWidget(instr_label)
        self.image_label = ROISelectionLabel("Image Preview")
        self.image_label.setStyleSheet("border: 2px solid #444; background: #111;")
        self.image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.image_label.roiSelected.connect(self._on_roi_drawn)
        left_layout.addWidget(self.image_label, stretch=1)
        center_layout.addWidget(left_frame, stretch=3)

        right_frame = QFrame()
        right_frame.setMaximumWidth(350)
        right_layout = QVBoxLayout(right_frame)
        right_layout.setSpacing(15)
        right_layout.setAlignment(Qt.AlignTop)
        
        self.scalebar_label = AspectRatioLabel("Selected Region")
        self.scalebar_label.setStyleSheet("border: 2px solid #0B3C5D; background: #111;")
        self.scalebar_label.setFixedHeight(150)
        self.scalebar_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        right_layout.addWidget(self.scalebar_label)
        
        self.recalc_btn = QPushButton("Re-Run Detection on Selection")
        self.recalc_btn.clicked.connect(lambda: self._run_roi_detection(debug=False))
        right_layout.addWidget(self.recalc_btn)
        
        right_layout.addSpacing(20)
        right_layout.addWidget(QLabel("2. Verify or Enter Manual Scale:"))
        
        manual_layout = QHBoxLayout()
        self.manual_scale_value_input = QLineEdit()
        self.manual_scale_value_input.setPlaceholderText("e.g., 100")
        self.manual_scale_value_input.returnPressed.connect(self._on_manual_recalc)
        manual_layout.addWidget(self.manual_scale_value_input)

        self.btn_unit_mm = QPushButton("mm")
        self.btn_unit_mm.setCheckable(True)
        self.btn_unit_mm.clicked.connect(self._on_manual_recalc)
        
        self.btn_unit_um = QPushButton("µm")
        self.btn_unit_um.setCheckable(True)
        self.btn_unit_um.setChecked(True) 
        self.btn_unit_um.clicked.connect(self._on_manual_recalc)

        manual_layout.addWidget(self.btn_unit_mm)
        manual_layout.addWidget(self.btn_unit_um)
        right_layout.addLayout(manual_layout)

        self.final_scale_label = QLabel("Result: Not calculated")
        self.final_scale_label.setStyleSheet("font-size: 14px; font-weight: bold; padding: 5px; color: #4CAF50;")
        self.final_scale_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.final_scale_label)
        right_layout.addStretch()

        nav = QHBoxLayout()
        self.prev_img_btn = QPushButton("Prev Image")
        self.next_img_btn = QPushButton("Next Image")
        self.prev_img_btn.clicked.connect(self.prev_image)
        self.next_img_btn.clicked.connect(self.next_image)
        nav.addWidget(self.prev_img_btn)
        nav.addWidget(self.next_img_btn)
        right_layout.addLayout(nav)
        
        self.apply_all_btn = QPushButton("Apply this Scale to All Images")
        self.apply_all_btn.clicked.connect(self.save_settings_to_all)
        right_layout.addWidget(self.apply_all_btn)
        
        center_layout.addWidget(right_frame, stretch=1)
        main_layout.addLayout(center_layout, stretch=1)

        bottom = QHBoxLayout()
        self.prev_btn = QPushButton("Prev: Upload")
        self.next_btn = QPushButton("Next: Substrate")
        self.prev_btn.setFixedHeight(50)
        self.next_btn.setFixedHeight(50)
        self.prev_btn.clicked.connect(self.goPrev.emit)
        self.next_btn.clicked.connect(self.goNext.emit)
        bottom.addWidget(self.prev_btn)
        bottom.addWidget(self.next_btn)
        main_layout.addLayout(bottom)

    # =================================================================
    # DATA MODEL FUNCTIONS
    # =================================================================
    
    def load_image_data(self, image_data_list):
        self.image_data = image_data_list 
        self.current_index = 0
        if not self.image_data:
            self.image_label.setText("No images loaded from main")
            self.final_scale_label.setText("Result: Error")
            return
        self.display_image() 

    def load_settings_for_current_image(self):
        if not self.image_data: return
        data = self.image_data[self.current_index]
        self.current_microns_per_pixel = data["microns_per_pixel"]
        self.manual_scale_value_input.setText(data["manual_val"])
        
        if data["manual_unit"] == "mm":
            self.btn_unit_mm.setChecked(True)
            self.btn_unit_um.setChecked(False)
        else:
            self.btn_unit_um.setChecked(True)
            self.btn_unit_mm.setChecked(False)
            
        if self.current_microns_per_pixel:
            self.final_scale_label.setText(f"Result (Saved): {self.current_microns_per_pixel:.4f} µm/px")
        else:
            self.final_scale_label.setText("Result: Not calculated")
            
        path = data["path"]
        if os.path.exists(path):
             img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
             if img is not None:
                 h, w = img.shape
                 rx = int((data.get("roi_x_pct", 0) / 100.0) * w)
                 ry = int((data.get("roi_y_pct", 0) / 100.0) * h)
                 rw = int((data.get("roi_w_pct", 0) / 100.0) * w)
                 rh = int((data.get("roi_h_pct", 0) / 100.0) * h)
                 
                 if rw > 0 and rh > 0:
                     self.roi_pixels = (rx, ry, rw, rh)
                     self.image_label.set_current_roi(rx, ry, rw, rh)
                     self._display_roi_preview(img[ry:ry+rh, rx:rx+rw])
                 else:
                     self.roi_pixels = None
                     self.image_label.set_current_roi(0,0,0,0)
                     self.scalebar_label.clear() 

    def save_settings_to_current(self):
        if not self.image_data: return
        data = self.image_data[self.current_index]
        data["microns_per_pixel"] = self.current_microns_per_pixel
        data["manual_val"] = self.manual_scale_value_input.text()
        data["manual_unit"] = "mm" if self.btn_unit_mm.isChecked() else "µm"
        
        if self.roi_pixels:
            x, y, w, h = self.roi_pixels
            path = data["path"]
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                h_img, w_img = img.shape
                data["roi_x_pct"] = (x / w_img) * 100.0
                data["roi_y_pct"] = (y / h_img) * 100.0
                data["roi_w_pct"] = (w / w_img) * 100.0
                data["roi_h_pct"] = (h / h_img) * 100.0
        print(f"Saved settings for image {self.current_index}")

    def save_settings_to_all(self):
        if not self.image_data: return

        # 1. Save the current screen's UI state to the current data object first
        self.save_settings_to_current()
        
        # 2. Get the source data (from current image)
        current_data = self.image_data[self.current_index]
        
        # Validation
        if current_data["microns_per_pixel"] is None:
            self.final_scale_label.setText("Result: Save current scale first!")
            return

        # 3. Apply to ALL images in the list
        for data in self.image_data:
            # Copy Scale Information
            data["microns_per_pixel"] = current_data["microns_per_pixel"]
            data["manual_val"] = current_data["manual_val"]
            data["manual_unit"] = current_data["manual_unit"]
            
            # Copy "Black Scalebar" setting (safe get in case key missing)
            data["scalebar_is_black"] = current_data.get("scalebar_is_black", False)
            
            # --- THE REQUESTED CHANGE: COPY ROI COORDINATES ---
            data["roi_x_pct"] = current_data["roi_x_pct"]
            data["roi_y_pct"] = current_data["roi_y_pct"]
            data["roi_w_pct"] = current_data["roi_w_pct"]
            data["roi_h_pct"] = current_data["roi_h_pct"]
        
        self.final_scale_label.setText(f"Applied SCALE & ROI to all {len(self.image_data)} images.")

    # =================================================================
    # CORE LOGIC
    # =================================================================
    
    def _on_roi_drawn(self, roi_rect):
        self.roi_pixels = roi_rect 
        self.final_scale_label.setText("Analyzing selection...")
        QApplication.processEvents()
        self._run_roi_detection(debug=False)
        self.save_settings_to_current()

    def _run_roi_detection(self, debug=False):
        if not self.image_data or not self.roi_pixels:
            self.final_scale_label.setText("Select a region first.")
            return

        data = self.image_data[self.current_index]
        path = data["path"]
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None: return

        x, y, w, h = self.roi_pixels
        h_img, w_img = image.shape
        x = max(0, min(x, w_img-1))
        y = max(0, min(y, h_img-1))
        w = max(1, min(w, w_img-x))
        h = max(1, min(h, h_img-y))
        roi = image[y:y+h, x:x+w]
        
        if roi.size == 0: return

        # --- 1. THRESHOLD & FIND BLACK BLOB (The Box) ---
        _, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Invert: Now black regions (the box background) are White.
        inverted = cv2.bitwise_not(thresh)
        
        contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        final_masked_roi = thresh # Default fallback
        
        if contours:
            # Find largest blob (the box background)
            largest_blob = max(contours, key=cv2.contourArea)
            blob_mask = np.zeros_like(thresh)
            cv2.drawContours(blob_mask, [largest_blob], -1, 255, -1)
            
            # MASKING: Keep ONLY the white pixels (text/lines) that are INSIDE the blob mask
            # Original thresh has white text/line.
            # blob_mask has white box area.
            # AND them together.
            final_masked_roi = cv2.bitwise_and(thresh, blob_mask)
        
        # --- 2. LINE DETECTION (With Endstops) ---
        roi_h_sub, roi_w_sub = final_masked_roi.shape
        horizontal_lines = []
        
        for r in range(roi_h_sub):
            row_pixels = final_masked_roi[r, :]
            white_pixels = np.where(row_pixels == 255)[0] 
            if len(white_pixels) > 20: 
                start_x, end_x = white_pixels[0], white_pixels[-1]
                length = end_x - start_x + 1
                
                # CRITICAL CHECK: Are there black pixels at the ends?
                # Check left neighbor
                has_left_stop = False
                if start_x > 0:
                     if row_pixels[start_x - 1] == 0: has_left_stop = True
                else: has_left_stop = True # Edge of image counts as stop
                
                # Check right neighbor
                has_right_stop = False
                if end_x < roi_w_sub - 1:
                     if row_pixels[end_x + 1] == 0: has_right_stop = True
                else: has_right_stop = True
                
                if has_left_stop and has_right_stop:
                     # Density check for solid line
                     segment = row_pixels[start_x:end_x+1]
                     white_ratio = np.sum(segment == 255) / len(segment)
                     if white_ratio > 0.8: 
                         horizontal_lines.append({'length': length, 'y': r, 'x1': start_x, 'x2': end_x})

        self.current_detected_line_length_px = None 
        
        roi_color = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
        if horizontal_lines:
            best_line = max(horizontal_lines, key=lambda x: x['length'])
            self.current_detected_line_length_px = best_line['length']
            
            ly = best_line['y']
            lx1 = best_line['x1']
            lx2 = best_line['x2']
            cv2.line(roi_color, (lx1, ly), (lx2, ly), (255, 0, 0), 2)

        self._display_roi_preview(roi_color)

        # --- 3. OCR (On the Masked Content) ---
        scale_val, scale_unit = None, None
        
        # Prepare for Tesseract: Invert so text is Black on White
        ocr_input = cv2.bitwise_not(final_masked_roi)
        # Upscale & Pad
        roi_upscaled = cv2.resize(ocr_input, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        roi_padded = cv2.copyMakeBorder(roi_upscaled, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(255, 255, 255))

        try:
            tess_config = r'--oem 3 --psm 7'
            text = pytesseract.image_to_string(roi_padded, config=tess_config) 
            
            match = re.search(r'(\d+[\.,]?\d*)\s*([a-zA-Zµμ]+)', text.lower())
            if match:
                 val_str = match.group(1).replace(',', '.')
                 scale_val = float(val_str)
                 unit_str = match.group(2)
                 if 'mm' in unit_str:
                     scale_unit = 'mm'
                 else:
                     scale_unit = 'µm'
        except: pass

        if self.current_detected_line_length_px and scale_val and scale_unit:
            unit_factor = {'mm': 1000, 'cm': 10000, 'µm': 1}
            self.current_microns_per_pixel = (scale_val * unit_factor.get(scale_unit, 1)) / self.current_detected_line_length_px
            
            self.manual_scale_value_input.setText(str(scale_val))
            if scale_unit == 'mm':
                self.btn_unit_mm.setChecked(True)
                self.btn_unit_um.setChecked(False)
            else:
                self.btn_unit_um.setChecked(True)
                self.btn_unit_mm.setChecked(False)
            
            self.final_scale_label.setText(f"Result (Saved): {self.current_microns_per_pixel:.4f} µm/px")

        elif self.current_detected_line_length_px:
            self.current_microns_per_pixel = None
            self.final_scale_label.setText("Line found. Enter value manually.")
        else:
            self.current_microns_per_pixel = None
            self.final_scale_label.setText("No line found in selection.")

    def _on_manual_recalc(self):
        sender = self.sender()
        if sender == self.btn_unit_mm and self.btn_unit_mm.isChecked():
            self.btn_unit_um.setChecked(False)
        elif sender == self.btn_unit_um and self.btn_unit_um.isChecked():
            self.btn_unit_mm.setChecked(False)
        
        try:
            manual_val = float(self.manual_scale_value_input.text())
        except ValueError:
            self.final_scale_label.setText("Result: Invalid Number")
            return

        if not self.current_detected_line_length_px:
            self.final_scale_label.setText("Result: No line found. Select ROI.")
            return
            
        if manual_val <= 0:
            self.final_scale_label.setText("Result: Value > 0")
            return

        is_mm = self.btn_unit_mm.isChecked()
        unit_factor = 1000 if is_mm else 1 
        
        self.current_microns_per_pixel = (manual_val * unit_factor) / self.current_detected_line_length_px
        self.final_scale_label.setText(f"Result (Saved): {self.current_microns_per_pixel:.4f} µm/px")
        self.save_settings_to_current()

    def _display_roi_preview(self, cv_img_data):
        if len(cv_img_data.shape) == 2:
             h, w = cv_img_data.shape
             img_display = cv2.cvtColor(cv_img_data, cv2.COLOR_GRAY2RGB)
        else:
             h, w, ch = cv_img_data.shape
             img_display = cv_img_data

        bytes_per_line = 3 * w
        qt_img = QImage(img_display.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.scalebar_label.setPixmap(QPixmap.fromImage(qt_img))

    def display_image(self):
        data = self.image_data[self.current_index]
        path = data["path"]
        if os.path.exists(path):
            pix = QPixmap(path)
            self.image_label.setPixmap(pix)
        self.load_settings_for_current_image()
        self.update_nav_buttons()

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

    def update_nav_buttons(self):
        if not self.image_data:
            self.prev_img_btn.setEnabled(False)
            self.next_img_btn.setEnabled(False)
            return
        self.prev_img_btn.setEnabled(self.current_index > 0)
        self.next_img_btn.setEnabled(self.current_index < len(self.image_data) - 1)

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
    
    win = ScaleFinderScreen()
    win.resize(1200, 800)
    
    test_data = [
        {
             "path": "C:/Users/aryan/Downloads/TR.tif", 
             "microns_per_pixel": None,
             "roi_x_pct": 0.0, "roi_y_pct": 70.0, "roi_w_pct": 30.0, "roi_h_pct": 30.0,
             "manual_val": "", "manual_unit": "µm"
        }
    ]
    valid_test_data = [d for d in test_data if os.path.exists(d["path"])]
    if valid_test_data:
        win.load_image_data(valid_test_data)

    win.show()
    sys.exit(app.exec())