import sys
import os
import cv2
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QHBoxLayout, QFrame, QPushButton, 
    QGridLayout, QLineEdit, QSizePolicy, QApplication, QSlider,
    QProgressDialog, QFileDialog, QTableWidget, QTableWidgetItem,
    QHeaderView, QCheckBox, QScrollArea, QGroupBox, QDialog, QTabWidget
)
from PySide6.QtGui import QPixmap, Qt, QImage, QPainter, QPen, QColor
from PySide6 import QtCore, QtGui

# Matplotlib for histogram
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

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
# MAIN SCREEN CLASS
# =====================================================================
class PorosityScreen(QWidget):

    goPrev = QtCore.Signal()
    
    def __init__(self, controller=None):
        super().__init__()
        self.controller = controller
        
        self.image_data = [] 
        self.current_index = 0
        
        # Store analysis data for histograms
        self.current_pore_areas_px = [] 
        self.current_pore_circs = []    
        self.current_scale_factor = 1.0 

        self._build_ui()

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        title = QLabel("Porosity Analysis")
        title.setStyleSheet("font-size: 24px; font-weight: bold;")
        main_layout.addWidget(title)

        # --- Main content area ---
        content_layout = QHBoxLayout()
        content_layout.setSpacing(20)

        # -------- Left: Image Preview --------
        image_frame = QFrame()
        image_layout = QVBoxLayout(image_frame)
        
        self.main_image_label = AspectRatioLabel("Processing...")
        self.main_image_label.setStyleSheet("border: 2px solid #0B3C5D; background: #111;")
        self.main_image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        image_layout.addWidget(self.main_image_label, stretch=1)
        
        # Image Nav
        nav_images = QHBoxLayout()
        self.prev_img_btn = QPushButton("Prev Image")
        self.next_img_btn = QPushButton("Next Image")
        self.prev_img_btn.clicked.connect(self.prev_image)
        self.next_img_btn.clicked.connect(self.next_image)
        nav_images.addWidget(self.prev_img_btn)
        nav_images.addWidget(self.next_img_btn)
        image_layout.addLayout(nav_images)

        content_layout.addWidget(image_frame, stretch=2)

        # -------- Right: Controls & Results --------
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QFrame.NoFrame)
        right_scroll.setMaximumWidth(480) 
        
        right_content = QWidget()
        right_layout = QVBoxLayout(right_content)
        right_layout.setSpacing(15)
        right_layout.setAlignment(Qt.AlignTop)
        
        # --- Contrast Control ---
        contrast_group = QGroupBox("Image Pre-Processing")
        contrast_layout = QGridLayout()
        
        contrast_layout.addWidget(QLabel("Contrast (CLAHE):"), 0, 0)
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(0, 100) 
        self.contrast_slider.setValue(10) 
        contrast_layout.addWidget(self.contrast_slider, 0, 1)
        
        self.contrast_input = QLineEdit("1.0")
        self.contrast_input.setFixedWidth(50)
        contrast_layout.addWidget(self.contrast_input, 0, 2)
        
        self._connect_sync(self.contrast_slider, self.contrast_input, is_float=True, scale=10.0)
        
        contrast_group.setLayout(contrast_layout)
        right_layout.addWidget(contrast_group)
        
        # 1. Analysis Height
        height_group = QGroupBox("Analysis Region")
        height_layout = QGridLayout()
        
        height_layout.addWidget(QLabel("Max Height (mm):"), 0, 0)
        self.height_slider = QSlider(Qt.Horizontal)
        self.height_slider.setRange(1, 200) 
        self.height_slider.setValue(20)
        height_layout.addWidget(self.height_slider, 0, 1)
        
        self.height_input = QLineEdit("2.0")
        self.height_input.setFixedWidth(50)
        height_layout.addWidget(self.height_input, 0, 2)

        self._connect_sync(self.height_slider, self.height_input, is_float=True, scale=10.0)
        
        height_group.setLayout(height_layout)
        right_layout.addWidget(height_group)
        
        # 2. Filters
        filter_group = QGroupBox("Pore Filters")
        filter_layout = QGridLayout() 
        
        # Min Area - FIX: Set default to 0
        filter_layout.addWidget(QLabel("Min Area (px):"), 0, 0)
        self.min_size_slider = QSlider(Qt.Horizontal)
        self.min_size_slider.setRange(0, 500)
        self.min_size_slider.setValue(0) # Default 0
        filter_layout.addWidget(self.min_size_slider, 0, 1)
        self.min_size_input = QLineEdit("0") # Default 0
        self.min_size_input.setFixedWidth(50)
        filter_layout.addWidget(self.min_size_input, 0, 2)
        
        # Max Area
        filter_layout.addWidget(QLabel("Max Area (px):"), 1, 0)
        self.max_size_slider = QSlider(Qt.Horizontal)
        self.max_size_slider.setRange(0, 10000)
        self.max_size_slider.setValue(10000)
        filter_layout.addWidget(self.max_size_slider, 1, 1)
        self.max_size_input = QLineEdit("10000")
        self.max_size_input.setFixedWidth(50)
        filter_layout.addWidget(self.max_size_input, 1, 2)

        # Min Sphericity
        filter_layout.addWidget(QLabel("Min Sph.:"), 2, 0)
        self.min_circ_slider = QSlider(Qt.Horizontal)
        self.min_circ_slider.setRange(0, 100)
        self.min_circ_slider.setValue(0)
        filter_layout.addWidget(self.min_circ_slider, 2, 1)
        self.min_circ_input = QLineEdit("0.00")
        self.min_circ_input.setFixedWidth(50)
        filter_layout.addWidget(self.min_circ_input, 2, 2)
        
        # Max Sphericity
        filter_layout.addWidget(QLabel("Max Sph.:"), 3, 0)
        self.max_circ_slider = QSlider(Qt.Horizontal)
        self.max_circ_slider.setRange(0, 100)
        self.max_circ_slider.setValue(100) 
        filter_layout.addWidget(self.max_circ_slider, 3, 1)
        self.max_circ_input = QLineEdit("1.00")
        self.max_circ_input.setFixedWidth(50)
        filter_layout.addWidget(self.max_circ_input, 3, 2)
        
        # Histogram Buttons
        self.hist_size_btn = QPushButton("Plot: Size Distribution")
        self.hist_size_btn.setStyleSheet("background-color: #555; font-size: 11px; padding: 4px;")
        self.hist_size_btn.clicked.connect(lambda: self._show_histogram_popup(0)) 
        filter_layout.addWidget(self.hist_size_btn, 4, 0, 1, 3)

        self.hist_circ_btn = QPushButton("Plot: Sphericity Distribution")
        self.hist_circ_btn.setStyleSheet("background-color: #555; font-size: 11px; padding: 4px;")
        self.hist_circ_btn.clicked.connect(lambda: self._show_histogram_popup(1))
        filter_layout.addWidget(self.hist_circ_btn, 5, 0, 1, 3)
        
        # Connect Sliders & Inputs
        self._connect_sync(self.min_size_slider, self.min_size_input, is_float=False)
        self._connect_sync(self.max_size_slider, self.max_size_input, is_float=False)
        self._connect_sync(self.min_circ_slider, self.min_circ_input, is_float=True, scale=100.0)
        self._connect_sync(self.max_circ_slider, self.max_circ_input, is_float=True, scale=100.0)

        filter_group.setLayout(filter_layout)
        right_layout.addWidget(filter_group)
        
        # 3. Intervals
        interval_group = QGroupBox("Height Intervals")
        interval_layout = QHBoxLayout()
        
        self.interval_check = QCheckBox("Enable Intervals")
        self.interval_check.toggled.connect(self._on_recalc)
        interval_layout.addWidget(self.interval_check)
        
        interval_layout.addWidget(QLabel("Step (mm):"))
        self.interval_input = QLineEdit("1.0")
        self.interval_input.setFixedWidth(50)
        self.interval_input.returnPressed.connect(self._on_recalc)
        interval_layout.addWidget(self.interval_input)
        
        interval_group.setLayout(interval_layout)
        right_layout.addWidget(interval_group)
        
        # 4. Results Table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels(["Height", "Porosity %", "Area px", "Avg Sph."])
        self.results_table.verticalHeader().setVisible(False)
        header = self.results_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        
        right_layout.addWidget(QLabel("Results:"))
        right_layout.addWidget(self.results_table)
        
        right_scroll.setWidget(right_content)
        content_layout.addWidget(right_scroll, stretch=1)
        
        main_layout.addLayout(content_layout, stretch=1)

        # --- Bottom navigation ---
        bottom = QHBoxLayout()
        self.prev_btn = QPushButton("Back to Menu")
        self.export_btn = QPushButton("Export Report (CSV)")
        self.prev_btn.setFixedHeight(50)
        self.export_btn.setFixedHeight(50)
        
        self.export_btn.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                font-weight: bold;
                background-color: #0B3C5D;
                color: white;
                border: 1px solid #1E5F8A;
                border-radius: 8px;
                padding: 10px;
            }
            QPushButton:hover { background-color: #1E5F8A; }
            QPushButton:pressed { background-color: #0A324E; }
        """)
        
        self.prev_btn.clicked.connect(self.goPrev.emit)
        self.export_btn.clicked.connect(self._on_export_csv)
        
        bottom.addWidget(self.prev_btn)
        bottom.addStretch()
        bottom.addWidget(self.export_btn)
        main_layout.addLayout(bottom)

    # =================================================================
    # UI SYNC HELPERS
    # =================================================================
    def _connect_sync(self, slider, line_edit, is_float=False, scale=1.0):
        """Bi-directional sync between slider and text box."""
        
        def slider_moved(val):
            if is_float:
                text_val = f"{val/scale:.2f}"
            else:
                text_val = str(val)
            line_edit.setText(text_val)
            
        def text_changed():
            try:
                val = float(line_edit.text())
                slider_val = int(val * scale) if is_float else int(val)
                slider.blockSignals(True)
                slider.setValue(slider_val)
                slider.blockSignals(False)
            except ValueError: pass

        # Connect
        slider.valueChanged.connect(slider_moved)
        slider.sliderReleased.connect(self._on_recalc) # Only recalc on release
        line_edit.editingFinished.connect(self._on_recalc) # Recalc on enter/blur
        line_edit.textChanged.connect(text_changed) # Update slider visuals live

    def _sync_height_slider_to_text(self, value):
        self.height_input.setText(f"{value/10.0:.1f}")

    def _on_recalc(self):
        self._run_full_analysis_on_current()

    # =================================================================
    # HISTOGRAM LOGIC
    # =================================================================
    def _show_histogram_popup(self, start_tab_index=0):
        if not self.current_pore_areas_px:
            return # No data

        dlg = QDialog(self)
        dlg.setWindowTitle("Pore Distributions")
        dlg.resize(700, 500)
        layout = QVBoxLayout(dlg)
        
        tabs = QTabWidget()
        
        # --- Tab 1: Size Distribution (Microns^2) ---
        tab1 = QWidget()
        layout1 = QVBoxLayout(tab1)
        fig1 = Figure(figsize=(5, 4), dpi=100)
        canvas1 = FigureCanvas(fig1)
        ax1 = fig1.add_subplot(111)
        
        # Convert Pixels to Microns^2
        scale_sq = self.current_scale_factor ** 2
        areas_um = [a * scale_sq for a in self.current_pore_areas_px]
        
        ax1.hist(areas_um, bins=30, color='skyblue', edgecolor='black')
        ax1.set_title(f"Pore Size Distribution (n={len(areas_um)})")
        ax1.set_xlabel("Area (µm²)")
        ax1.set_ylabel("Frequency")
        ax1.grid(True, alpha=0.3)
        layout1.addWidget(canvas1)
        tabs.addTab(tab1, "Size (µm²)")
        
        # --- Tab 2: Sphericity Distribution ---
        tab2 = QWidget()
        layout2 = QVBoxLayout(tab2)
        fig2 = Figure(figsize=(5, 4), dpi=100)
        canvas2 = FigureCanvas(fig2)
        ax2 = fig2.add_subplot(111)
        
        ax2.hist(self.current_pore_circs, bins=20, range=(0.0, 1.0), color='lightgreen', edgecolor='black')
        ax2.set_title(f"Sphericity Distribution (n={len(self.current_pore_circs)})")
        ax2.set_xlabel("Sphericity (1.0 = Circle)")
        ax2.set_ylabel("Frequency")
        ax2.grid(True, alpha=0.3)
        layout2.addWidget(canvas2)
        tabs.addTab(tab2, "Sphericity")
        
        tabs.setCurrentIndex(start_tab_index)
        
        layout.addWidget(tabs)
        dlg.exec()

    # =================================================================
    # LOGIC
    # =================================================================
    
    def load_image_data(self, image_data_list):
        self.image_data = image_data_list
        self.current_index = 0
        
        if not self.image_data:
            self.main_image_label.setText("No images loaded from main")
            return

        data = self.image_data[0]
        if "pore_max_height_mm" in data:
            self.height_input.setText(str(data["pore_max_height_mm"]))
            # Fix: Set min size default to 0
            self.min_size_input.setText(str(data.get("pore_min_size", 0))) 
            self.max_size_input.setText(str(data.get("pore_max_size", 10000)))
            self.min_circ_input.setText(f"{data.get('pore_min_circ', 0.0):.2f}")
            self.max_circ_input.setText(f"{data.get('pore_max_circ', 1.0):.2f}")
            self.interval_check.setChecked(data.get("pore_use_intervals", False))
            
            contrast_val = data.get("contrast_clip_limit", 1.0)
            self.contrast_input.setText(f"{contrast_val:.1f}")
            self.contrast_slider.setValue(int(contrast_val * 10))
        
        self.display_image()
        self.update_nav_buttons()

    def _run_full_analysis_on_current(self):
        if not self.image_data: return
        data = self.image_data[self.current_index]
        
        try:
            max_height_mm = float(self.height_input.text())
            min_size = int(self.min_size_input.text())
            max_size = int(self.max_size_input.text())
            min_circ = float(self.min_circ_input.text())
            max_circ = float(self.max_circ_input.text())
            contrast_val = float(self.contrast_input.text())
            use_intervals = self.interval_check.isChecked()
            interval_mm = float(self.interval_input.text())

            data["pore_max_height_mm"] = max_height_mm
            data["pore_min_size"] = min_size
            data["pore_max_size"] = max_size
            data["pore_min_circ"] = min_circ
            data["pore_max_circ"] = max_circ
            data["pore_use_intervals"] = use_intervals
            data["pore_interval_mm"] = interval_mm
            data["contrast_clip_limit"] = contrast_val 
            
            results, overlay_img, pore_areas, pore_circs = self._calculate_porosity_for_data(data)
            
            self.current_pore_areas_px = pore_areas
            self.current_pore_circs = pore_circs
            self.current_scale_factor = data.get("microns_per_pixel", 1.0)
            
            self.main_image_label.setPixmap(self._convert_cv_to_pixmap(overlay_img))
            self._populate_table(results)
            
            data["porosity_results_list"] = results

        except Exception as e:
            print(f"Error analyzing: {e}")
            self.main_image_label.setText(f"Error: {e}")

    def _calculate_porosity_for_data(self, data):
        path = data.get("path")
        if not path or not os.path.exists(path):
            raise ValueError("Image path not found")
            
        microns_per_pixel = data.get("microns_per_pixel")
        if not microns_per_pixel: microns_per_pixel = 1.0 

        max_height_mm = data.get("pore_max_height_mm", 2.0)
        # Fix: Default to 0
        min_size = data.get("pore_min_size", 0) 
        max_size = data.get("pore_max_size", 10000)
        min_circ = data.get("pore_min_circ", 0.0)
        max_circ = data.get("pore_max_circ", 1.0)
        use_intervals = data.get("pore_use_intervals", False)
        interval_mm = data.get("pore_interval_mm", 1.0)
        
        left_trace = data.get("substrate_trace_left")
        right_trace = data.get("substrate_trace_right")
        left_idx = data.get("substrate_left_endpoint_index", -1)
        right_idx = data.get("substrate_right_endpoint_index", -1)
        is_manual = data.get("substrate_is_manual", False)

        # 1. Load & Preprocess
        image_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        image_color = cv2.imread(path, cv2.IMREAD_COLOR)
        h_img, w_img = image_gray.shape

        roi_x = int((data.get("roi_x_pct", 0) / 100.0) * w_img)
        roi_y = int((data.get("roi_y_pct", 0) / 100.0) * h_img)
        roi_w = int((data.get("roi_w_pct", 30) / 100.0) * w_img)
        roi_h = int((data.get("roi_h_pct", 30) / 100.0) * h_img)
        cv2.rectangle(image_gray, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 0, 0), -1)

        # Contrast
        clip_limit = data.get("contrast_clip_limit", 1.0)
        if clip_limit > 0.1: 
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            image_gray = clahe.apply(image_gray)

        _, otsu = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 2. Define Substrate Line
        if left_idx == -1: left_idx = len(left_trace) - 1
        if right_idx == -1: right_idx = len(right_trace) - 1
            
        if is_manual:
            substrate_line = left_trace
        else:
            substrate_line = left_trace[:left_idx+1] + right_trace[:right_idx+1][::-1]

        # 3. Find Valid Pores (Global)
        inverted_otsu = cv2.bitwise_not(otsu)
        contours, _ = cv2.findContours(inverted_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_pores = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_size or area > max_size:
                continue
                
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0: continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            if circularity < min_circ or circularity > max_circ:
                continue
                
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = cnt[0][0]
                
            valid_pores.append({
                'cnt': cnt, 'area': area, 'circ': circularity, 'cx': cX, 'cy': cY
            })
            
        # O(N) Removal of Largest Pore
        if valid_pores:
            largest_idx = -1
            max_area = -1
            for i, pore in enumerate(valid_pores):
                if pore['area'] > max_area:
                    max_area = pore['area']
                    largest_idx = i
            if largest_idx != -1:
                valid_pores.pop(largest_idx)

        collected_pore_areas = [p['area'] for p in valid_pores]
        collected_pore_circs = [p['circ'] for p in valid_pores]

        # 4. Interval Analysis
        results = []
        
        intervals = []
        if use_intervals:
            curr_h = 0.0
            while curr_h < max_height_mm:
                next_h = min(curr_h + interval_mm, max_height_mm)
                intervals.append((curr_h, next_h))
                if next_h >= max_height_mm: break
                curr_h = next_h
        else:
            intervals.append((0.0, max_height_mm))
            
        for h_start, h_end in intervals:
            off_start_px = int(round(h_start * 1000 / microns_per_pixel))
            off_end_px = int(round(h_end * 1000 / microns_per_pixel))
            
            poly_bot = [(x, max(0, y - off_start_px)) for (x, y) in substrate_line]
            poly_top = [(x, max(0, y - off_end_px)) for (x, y) in substrate_line]
            slice_poly = np.array(poly_bot + poly_top[::-1], dtype=np.int32)
            
            slice_mask = np.zeros_like(otsu)
            cv2.fillPoly(slice_mask, [slice_poly], 255)
            
            contours_mat, _ = cv2.findContours(otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            total_slice_area = 0
            if contours_mat:
                largest_mat = max(contours_mat, key=cv2.contourArea)
                global_mat_mask = np.zeros_like(otsu)
                cv2.drawContours(global_mat_mask, [largest_mat], -1, 255, -1)
                true_slice_area_mask = cv2.bitwise_and(global_mat_mask, slice_mask)
                total_slice_area = cv2.countNonZero(true_slice_area_mask)
                
            slice_pore_area = 0
            slice_pore_count = 0
            avg_circ = 0
            
            for pore in valid_pores:
                if slice_mask[pore['cy'], pore['cx']] == 255:
                    slice_pore_area += pore['area']
                    slice_pore_count += 1
                    avg_circ += pore['circ']
                    
                    cv2.drawContours(image_color, [pore['cnt']], -1, (0, 0, 255), -1)
            
            if slice_pore_count > 0:
                avg_circ /= slice_pore_count
                
            porosity_pct = 0.0
            if total_slice_area > 0:
                porosity_pct = (slice_pore_area / total_slice_area) * 100.0
                
            results.append({
                "range": f"{h_start:.1f}-{h_end:.1f} mm",
                "porosity": porosity_pct,
                "pore_area": slice_pore_area,
                "avg_circ": avg_circ
            })
            
            pts = np.array(poly_top, np.int32)
            cv2.polylines(image_color, [pts], False, (0, 255, 255), 1) 

        return (results, image_color, collected_pore_areas, collected_pore_circs)

    def _populate_table(self, results):
        self.results_table.setRowCount(len(results))
        for i, res in enumerate(results):
            self.results_table.setItem(i, 0, QTableWidgetItem(res["range"]))
            self.results_table.setItem(i, 1, QTableWidgetItem(f"{res['porosity']:.2f} %"))
            self.results_table.setItem(i, 2, QTableWidgetItem(f"{res['pore_area']:.0f}"))
            self.results_table.setItem(i, 3, QTableWidgetItem(f"{res['avg_circ']:.2f}"))

    def display_image(self):
        if not self.image_data: return
        self._run_full_analysis_on_current()

    def _convert_cv_to_pixmap(self, cv_img_bgr):
        h, w, ch = cv_img_bgr.shape
        bytes_per_line = ch * w
        qt_img = QImage(cv_img_bgr.data, w, h, bytes_per_line, QImage.Format_BGR888).rgbSwapped()
        return QPixmap.fromImage(qt_img)

    def _on_export_csv(self):
        if not self.image_data: return
        
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Report", "porosity_report.csv", "CSV Files (*.csv)")
        if not file_path: return
        
        progress = QProgressDialog("Calculating all images...", "Cancel", 0, len(self.image_data), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setValue(0)
        
        all_results = []
        
        try:
            max_h = float(self.height_input.text())
            min_s = int(self.min_size_input.text())
            max_s = int(self.max_size_input.text())
            min_c = float(self.min_circ_input.text())
            max_c = float(self.max_circ_input.text())
            contrast_val = float(self.contrast_input.text())
            use_int = self.interval_check.isChecked()
            int_mm = float(self.interval_input.text())
        except:
            return 

        for i, data in enumerate(self.image_data):
            progress.setValue(i)
            QApplication.processEvents()
            if progress.wasCanceled(): return
            
            try:
                data["pore_max_height_mm"] = max_h
                data["pore_min_size"] = min_s
                data["pore_max_size"] = max_s
                data["pore_min_circ"] = min_c
                data["pore_max_circ"] = max_c
                data["pore_use_intervals"] = use_int
                data["pore_interval_mm"] = int_mm
                data["contrast_clip_limit"] = contrast_val
                
                res_tuple, _, _, _ = self._calculate_porosity_for_data(data)
                results_list = res_tuple[0]
                all_results.append((os.path.basename(data['path']), results_list))
            except Exception as e:
                print(e)

        progress.setValue(len(self.image_data))

        try:
            with open(file_path, 'w') as f:
                f.write("Filename,Range,Porosity (%),Pore Area (px),Avg Sphericity\n")
                for filename, rows in all_results:
                    for r in rows:
                        f.write(f"{filename},{r['range']},{r['porosity']:.4f},{r['pore_area']},{r['avg_circ']:.4f}\n")
        except Exception as e:
            print(f"Export failed: {e}")

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
    
    win = PorosityScreen()
    
    test_data = [
        {
            "path": r"C:\Users\aryan\Downloads\TR.tif",
            "microns_per_pixel": 2.57,
            "contrast_clip_limit": 2.0,
            "analysis_height_mm": 5.0,
            "substrate_trace_left": [(x, 800) for x in range(0, 500)],
            "substrate_trace_right": [(x, 800) for x in range(999, 500, -1)],
            "substrate_left_endpoint_index": 499,
            "substrate_right_endpoint_index": 0,
            "roi_x_pct": 0, "roi_y_pct": 0, "roi_w_pct": 0, "roi_h_pct": 0
        }
    ]
    
    valid_data = [d for d in test_data if os.path.exists(d['path'])]
    if valid_data:
        win.load_image_data(valid_data)
    
    win.resize(1200, 800)
    win.show()
    sys.exit(app.exec())