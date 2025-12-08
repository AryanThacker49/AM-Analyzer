import sys
import os
import cv2
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QHBoxLayout, QFrame, QPushButton, 
    QGridLayout, QLineEdit, QSizePolicy, QApplication, QSlider,
    QProgressDialog, QFileDialog, QTableWidget, QTableWidgetItem,
    QHeaderView
)
from PySide6.QtGui import QPixmap, Qt, QImage, QPainter, QPen, QColor
from PySide6 import QtCore, QtGui

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
class ThicknessScreen(QWidget):

    goPrev = QtCore.Signal()
    
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

        title = QLabel("Thickness vs. Height Analysis")
        title.setStyleSheet("font-size: 24px; font-weight: bold;")
        main_layout.addWidget(title)

        # --- Main content area ---
        content_layout = QHBoxLayout()
        content_layout.setSpacing(20)

        # -------- Left: Image Preview --------
        image_frame = QFrame()
        image_layout = QVBoxLayout(image_frame)
        
        self.main_image_label = AspectRatioLabel("Image with Thickness Lines")
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

        content_layout.addWidget(image_frame, stretch=3)

        # -------- Right: Controls & Table --------
        right_frame = QFrame()
        right_frame.setMaximumWidth(400) 
        right_layout = QVBoxLayout(right_frame)
        right_layout.setSpacing(15)
        right_layout.setAlignment(Qt.AlignTop)

        # --- Input Controls ---
        control_layout = QGridLayout()
        control_layout.addWidget(QLabel("Distance between measurements:"), 0, 0)
        self.interval_input = QLineEdit("1.0")
        self.interval_input.setFixedWidth(50)
        control_layout.addWidget(self.interval_input, 0, 1)
        control_layout.addWidget(QLabel("mm"), 0, 2)
        
        control_layout.addWidget(QLabel("...for the bottom:"), 1, 0)
        self.total_height_input = QLineEdit("2.0") 
        self.total_height_input.setFixedWidth(50)
        control_layout.addWidget(self.total_height_input, 1, 1)
        control_layout.addWidget(QLabel("mm"), 1, 2)
        
        self.calc_btn = QPushButton("Calculate Thickness")
        self.calc_btn.clicked.connect(self._run_thickness_analysis_current)
        control_layout.addWidget(self.calc_btn, 2, 0, 1, 3) 
        
        # --- NEW: Wasted Material Button ---
        self.calc_waste_btn = QPushButton("Calculate Excess Deposited Material")
        self.calc_waste_btn.clicked.connect(self._run_wasted_material_analysis)
        control_layout.addWidget(self.calc_waste_btn, 3, 0, 1, 3)
        
        # --- NEW: Wasted Material Label ---
        self.wasted_label = QLabel("Efficiency (A/B): N/A")
        self.wasted_label.setStyleSheet("font-weight: bold;")
        control_layout.addWidget(self.wasted_label, 4, 0, 1, 3)
        
        right_layout.addLayout(control_layout)

        # --- Results Table ---
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(2)
        self.results_table.setHorizontalHeaderLabels(["Height (mm)", "Thickness (mm)"])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        right_layout.addWidget(self.results_table, stretch=1)
        
        content_layout.addWidget(right_frame, stretch=1)
        
        main_layout.addLayout(content_layout, stretch=1)

        # --- Bottom navigation ---
        bottom = QHBoxLayout()
        self.prev_btn = QPushButton("Prev: Analyze")
        self.export_btn = QPushButton("Export all as CSV")
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
    # OPENCV & ANALYSIS LOGIC
    # =================================================================
    
    def _log_to_file(self, message):
        """Appends a message to the debug log file."""
        log_dir = os.path.dirname(sys.executable if getattr(sys, 'frozen', False) else __file__)
        log_file_path = os.path.join(log_dir, "thickness_calc_log.txt")
        try:
            with open(log_file_path, 'a') as f:
                f.write(message)
        except Exception as e:
            print(f"Failed to write to log file: {e}")
            
    def _get_cleaned_analysis_inputs(self, data, interval_mm, total_height_mm):
        """
        REFACTORED: This is the common setup logic for both
        thickness and waste calculation.
        """
        path = data.get("path")
        microns_per_pixel = data.get("microns_per_pixel")
        left_trace = data.get("substrate_trace_left")
        right_trace = data.get("substrate_trace_right")
        left_idx = data.get("substrate_left_endpoint_index", -1)
        right_idx = data.get("substrate_right_endpoint_index", -1)
        is_manual = data.get("substrate_is_manual", False)
        clip_limit = data.get("contrast_clip_limit", 1.0)

        if not all([path, microns_per_pixel, left_trace, right_trace]):
            raise ValueError("Missing critical data (path, scale, or substrate trace).")

        image_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        image_color = cv2.imread(path, cv2.IMREAD_COLOR) 
        if image_gray is None:
            raise ValueError(f"Could not load image: {path}")

        h_img, w_img = image_gray.shape
        roi_x = int((data.get("roi_x_pct", 0) / 100.0) * w_img)
        roi_y = int((data.get("roi_y_pct", 0) / 100.0) * h_img)
        roi_w = int((data.get("roi_w_pct", 30) / 100.0) * w_img)
        roi_h = int((data.get("roi_h_pct", 30) / 100.0) * h_img)
        cv2.rectangle(image_gray, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 0, 0), -1)

        if clip_limit > 0.1: 
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            high_contrast_image = clahe.apply(image_gray)
        else:
            high_contrast_image = image_gray 
            
        _, otsu_thresh = cv2.threshold(high_contrast_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(otsu_thresh, connectivity=8)
        cleaned_otsu = np.zeros_like(otsu_thresh)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            cleaned_otsu[labels == largest_label] = 255
        else:
            cleaned_otsu = otsu_thresh.copy()
            
        return (image_color, cleaned_otsu, data, microns_per_pixel, 
                left_trace, right_trace, left_idx, right_idx, is_manual)

    def _run_thickness_analysis_current(self):
        """Runs the thickness analysis for the currently visible image."""
        if not self.image_data: return
        data = self.image_data[self.current_index]
        
        try:
            interval_mm = float(self.interval_input.text())
            total_height_mm = float(self.total_height_input.text())
        except ValueError:
            self.main_image_label.setText("Invalid interval or height value.")
            return

        try:
            (interface_result, height_results), overlay_image = self._calculate_thickness_for_data(data, interval_mm, total_height_mm)
            
            results_list = [interface_result] + height_results
            
            data["thickness_results"] = results_list
            data["thickness_interval_mm"] = interval_mm
            data["thickness_total_height_mm"] = total_height_mm 
            
            self.main_image_label.setPixmap(self._convert_cv_to_pixmap(overlay_image))
            self._populate_table(results_list)

        except Exception as e:
            self._log_to_file(f"--- ERROR during thickness analysis for {data.get('path')} ---\n{e}\n\n")
            self.main_image_label.setText(f"Error:\n{e}")

    def _calculate_thickness_for_data(self, data, interval_mm, total_height_mm):
        """
        Runs the full thickness pipeline for a single image's data.
        Returns: ( (interface_result_tuple), [height_results_list] ), overlay_image
        """
        
        try:
            (image_color, cleaned_otsu, data, microns_per_pixel, 
             left_trace, right_trace, left_idx, right_idx, is_manual) = self._get_cleaned_analysis_inputs(data, interval_mm, total_height_mm)

            interval_px = int(round((interval_mm * 1000) / microns_per_pixel))
            if interval_px == 0:
                raise ValueError("Interval is 0px. Check scale or interval value.")
                
            total_height_px = int(round((total_height_mm * 1000) / microns_per_pixel))

            height_results = []
            thickness_image = image_color.copy() 

            if left_idx == -1: left_idx = len(left_trace) - 1
            if right_idx == -1: right_idx = len(right_trace) - 1
                
            pt_left = left_trace[left_idx]
            pt_right = right_trace[right_idx]
            
            interface_y = min(pt_left[1], pt_right[1])
            stop_y = interface_y - total_height_px
            font = cv2.FONT_HERSHEY_SIMPLEX
            height_mm_counter = 0.0

            # --- Step 1 - Manually calculate and add the 0.0mm line ---
            interface_thickness_px = pt_right[0] - pt_left[0]
            interface_thickness_mm = (interface_thickness_px * microns_per_pixel) / 1000.0
            
            self._log_to_file(f"Image: {os.path.basename(data['path'])} | Interface Thickness: {interface_thickness_mm} mm\n")
            
            interface_result = (0.0, interface_thickness_mm) 
            
            # --- Step 2 - Draw the substrate lines (not the thickness line) ---
            if is_manual:
                cv2.polylines(thickness_image, [np.array(left_trace, dtype=np.int32)], isClosed=False, color=(0, 255, 0), thickness=2)
            else:
                cv2.polylines(thickness_image, [np.array(left_trace, dtype=np.int32)], isClosed=False, color=(0, 255, 0), thickness=2)
                cv2.polylines(thickness_image, [np.array(right_trace, dtype=np.int32)], isClosed=False, color=(0, 255, 0), thickness=2)
            
            cv2.line(thickness_image, (pt_left[0], interface_y), (pt_right[0], interface_y), (0, 0, 255), 2) 
            label = f"{interface_thickness_mm:.2f} mm"
            cv2.putText(thickness_image, label, (pt_right[0] + 5, interface_y + 5), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            # --- Step 3 - Start the loop *above* the interface ---
            y = interface_y - interval_px
            height_mm_counter = interval_mm
            empty_streak = 0

            while y > 0 and y > stop_y: 
                row = cleaned_otsu[y, :] 
                white_indices = np.where(row == 255)[0]
                
                if len(white_indices) < 2:
                    empty_streak += 1
                    if empty_streak >= 3:
                        break
                else:
                    empty_streak = 0
                    left = white_indices[0]
                    right = white_indices[-1]
                    thickness_px = right - left
                    thickness_mm = (thickness_px * microns_per_pixel) / 1000.0
                    
                    height_results.append((height_mm_counter, thickness_mm)) 
                    
                    cv2.line(thickness_image, (left, y), (right, y), (0, 0, 255), 2)
                    
                    label = f"{thickness_mm:.2f} mm"
                    cv2.putText(thickness_image, label, (right + 5, y + 5), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                
                y -= interval_px
                height_mm_counter += interval_mm
                
            return (interface_result, height_results), thickness_image
        
        except Exception as e:
            self._log_to_file(f"--- ERROR processing {data.get('path')} ---\n{e}\n\n")
            raise e

    # =================================================================
    # NEW: Wasted Material Calculation
    # =================================================================
    def _run_wasted_material_analysis(self):
        """
        Runs the A/B efficiency calculation and displays the result.
        """
        if not self.image_data: return
        data = self.image_data[self.current_index]
        
        try:
            # Get settings from UI
            interval_mm = float(self.interval_input.text())
            total_height_mm = float(self.total_height_input.text())
        except ValueError:
            self.main_image_label.setText("Invalid interval or height value.")
            return

        try:
            # 1. Get all the pre-processed data
            (image_color, cleaned_otsu, data, microns_per_pixel, 
             left_trace, right_trace, left_idx, right_idx, is_manual) = self._get_cleaned_analysis_inputs(data, interval_mm, total_height_mm)

            # 2. Get endpoint and height data
            if left_idx == -1: left_idx = len(left_trace) - 1
            if right_idx == -1: right_idx = len(right_trace) - 1
            
            pt_left = left_trace[left_idx]
            pt_right = right_trace[right_idx]
            
            interface_y = min(pt_left[1], pt_right[1])
            total_height_px = int(round((total_height_mm * 1000) / microns_per_pixel))
            stop_y = interface_y - total_height_px

            # 3. Define the two regions, A (Box) and B (Blob)
            
            # --- AREA A (Ideal Box) ---
            # Define the four corners of the red box
            A_box_poly = np.array([
                (pt_left[0], interface_y),
                (pt_right[0], interface_y),
                (pt_right[0], stop_y),
                (pt_left[0], stop_y)
            ], dtype=np.int32)
            
            # Calculate Area A (simple rectangle)
            A_box_area_px = (pt_right[0] - pt_left[0]) * (interface_y - stop_y)

            # --- AREA B (Actual Material) ---
            # Create the *full analysis region mask* (the one from contrast screen)
            if is_manual:
                substrate_line_trace = left_trace
            else:
                substrate_line_trace = left_trace[:left_idx+1] + right_trace[:right_idx+1][::-1]
            
            top_line_trace = [(x, max(0, y - total_height_px)) for (x, y) in substrate_line_trace]
            analysis_poly = np.array(substrate_line_trace + top_line_trace[::-1], dtype=np.int32)
            
            region_mask = np.zeros_like(cleaned_otsu)
            cv2.fillPoly(region_mask, [analysis_poly], 255)
            
            # Get the actual material (B) inside this region
            B_material_in_region = cv2.bitwise_and(cleaned_otsu, cleaned_otsu, mask=region_mask)
            B_material_area_px = cv2.countNonZero(B_material_in_region)

            # 4. Calculate Ratio
            if B_material_area_px > 0:
                efficiency = (A_box_area_px / B_material_area_px) * 100.0
                self.wasted_label.setText(f"Efficiency (A/B): {efficiency:.2f} %")
                data["wasted_material_ratio"] = efficiency
            else:
                self.wasted_label.setText("Efficiency (A/B): N/A (No material)")
                data["wasted_material_ratio"] = None

            # 5. Draw Overlays
            # Draw B (Green Blob)
            contours, _ = cv2.findContours(B_material_in_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image_color, contours, -1, (0, 255, 0), 2) # Green outline
            
            # Draw A (Red Box)
            cv2.polylines(image_color, [A_box_poly], isClosed=True, color=(0, 0, 255), thickness=2) # Red box
            
            self.main_image_label.setPixmap(self._convert_cv_to_pixmap(image_color))

        except Exception as e:
            self._log_to_file(f"--- ERROR during Wasted Material analysis for {data.get('path')} ---\n{e}\n\n")
            self.main_image_label.setText(f"Error:\n{e}")

    # =================================================================
    # DATA MODEL & NAVIGATION
    # =================================================================
    def load_image_data(self, image_data_list):
        self.image_data = image_data_list
        self.current_index = 0
        if not self.image_data:
            self.main_image_label.setText("No images loaded from main")
            return
        
        self.display_image()
        self.update_nav_buttons()

    def display_image(self):
        """
        Loads the current image and pre-saved results.
        """
        if not self.image_data: return
        data = self.image_data[self.current_index]
        
        results_list = data.get("thickness_results") 
        
        interval_mm = data.get("thickness_interval_mm", 1.0)
        total_height_mm = data.get("thickness_total_height_mm", 2.0)
        self.interval_input.setText(str(interval_mm))
        self.total_height_input.setText(str(total_height_mm))
        
        # Clear old results
        self.wasted_label.setText("Efficiency (A/B): N/A")
        
        try:
            pixmap = self._get_image_with_basic_overlay(data) 
            self.main_image_label.setPixmap(pixmap)
            
            if results_list:
                self._populate_table(results_list)
            else:
                self.results_table.setRowCount(0)
                
        except Exception as e:
            self.main_image_label.setText(f"Error drawing overlay:\n{e}")
            print(f"Error in display_image: {e}")

        self.update_nav_buttons()

    def _get_image_with_basic_overlay(self, data):
        """Just draws the substrate lines, no thickness."""
        path = data.get("path")
        microns_per_pixel = data.get("microns_per_pixel")
        left_trace = data.get("substrate_trace_left")
        right_trace = data.get("substrate_trace_right")
        left_idx = data.get("substrate_left_endpoint_index", -1)
        right_idx = data.get("substrate_right_endpoint_index", -1)
        is_manual = data.get("substrate_is_manual", False)

        if not all([path, microns_per_pixel, left_trace, right_trace]):
            raise ValueError("Missing critical data for drawing.")

        image_color = cv2.imread(path, cv2.IMREAD_COLOR) 
        if image_color is None:
            raise ValueError("Could not load image.")
            
        if left_idx == -1: left_idx = len(left_trace) - 1
        if right_idx == -1: right_idx = len(right_trace) - 1

        if is_manual:
            substrate_line_trace = left_trace
            cv2.polylines(image_color, [np.array(substrate_line_trace, dtype=np.int32)], isClosed=False, color=(0, 255, 0), thickness=2)
        else:
            cv2.polylines(image_color, [np.array(left_trace, dtype=np.int32)], isClosed=False, color=(0, 255, 0), thickness=2)
            cv2.polylines(image_color, [np.array(right_trace, dtype=np.int32)], isClosed=False, color=(0, 255, 0), thickness=2)
            pt_left = left_trace[left_idx]
            pt_right = right_trace[right_idx]
            interface_y = min(pt_left[1], pt_right[1])
            cv2.line(image_color, (pt_left[0], interface_y), (pt_right[0], interface_y), (0, 0, 255), 2) 
        
        return self._convert_cv_to_pixmap(image_color)

    def _convert_cv_to_pixmap(self, cv_img_bgr):
        """Converts BGR OpenCV image to QPixmap."""
        h, w, ch = cv_img_bgr.shape
        bytes_per_line = ch * w
        qt_img = QImage(cv_img_bgr.data, w, h, bytes_per_line, QImage.Format_BGR888).rgbSwapped()
        return QPixmap.fromImage(qt_img)

    def _populate_table(self, results):
        """Fills the QTableWidget with results."""
        self.results_table.setRowCount(len(results))
        for i, (height_mm, thickness_mm) in enumerate(results):
            
            height_item = QTableWidgetItem(f"{height_mm:.3f}")
            self.results_table.setItem(i, 0, height_item)
            
            thickness_item = QTableWidgetItem(f"{thickness_mm:.3f}")
            self.results_table.setItem(i, 1, thickness_item)

    def _on_export_csv(self):
        """Saves all calculated thickness data to a CSV file."""
        if not self.image_data:
            return
            
        try:
            interval_mm = float(self.interval_input.text())
            total_height_mm = float(self.total_height_input.text())
        except ValueError:
            self.main_image_label.setText("Invalid interval or height value.")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Thickness Report", "", "CSV Files (*.csv)")
        
        if not file_path:
            return # User cancelled

        log_dir = os.path.dirname(sys.executable if getattr(sys, 'frozen', False) else __file__)
        log_file_path = os.path.join(log_dir, "thickness_export_log.txt")
        
        try:
            with open(log_file_path, 'w') as f:
                f.write(f"--- STARTING THICKNESS EXPORT ---\n")
                f.write(f"Interval: {interval_mm} mm, Total Height: {total_height_mm} mm\n\n")
        except Exception as e:
            print(f"Failed to create log file: {e}")

        progress = QProgressDialog("Calculating Thickness for all images...", "Cancel", 0, len(self.image_data), self)
        progress.setWindowTitle("Exporting CSV")
        progress.setWindowModality(Qt.WindowModal)
        progress.setValue(0)
        
        all_results_by_file = {}
        
        for i, data in enumerate(self.image_data):
            progress.setValue(i)
            progress.setLabelText(f"Processing image {i+1} of {len(self.image_data)}...")
            QApplication.processEvents() 

            if progress.wasCanceled():
                return

            try:
                (interface_result, height_results), _ = self._calculate_thickness_for_data(data, interval_mm, total_height_mm)
                
                with open(log_file_path, 'a') as f:
                    f.write(f"--- Processing {os.path.basename(data['path'])} ---\n")
                    f.write(f"Interface: {str(interface_result)}\n")
                    f.write(f"Heights: {str(height_results)}\n\n")

                all_results_by_file[os.path.basename(data["path"])] = (interface_result, height_results)
                
            except Exception as e:
                with open(log_file_path, 'a') as f:
                    f.write(f"--- ERROR processing {os.path.basename(data['path'])} ---\n")
                    f.write(str(e) + "\n\n")
                print(f"Error processing {data['path']}: {e}")
                all_results_by_file[os.path.basename(data["path"])] = ((0.0, "ERROR"), [])

        progress.setValue(len(self.image_data))
        
        csv_lines = ["Filename,Measurement Type,Thickness (mm)"]
        for filename, (interface_result, height_results) in all_results_by_file.items():
            
            if isinstance(interface_result[1], str):
                csv_lines.append(f"{filename},Interface,{interface_result[1]}") # Append "ERROR"
            else:
                csv_lines.append(f"{filename},Interface,{interface_result[1]:.3f}") # Append formatted number
            
            for height_mm, thickness_mm in height_results:
                height_label = f"Height {height_mm:.3f}mm"
                thickness_label = f"{thickness_mm:.3f}"
                if isinstance(thickness_mm, str):
                    thickness_label = thickness_mm 
                    
                csv_lines.append(f"{filename},{height_label},{thickness_label}")
                
        csv_content = "\n".join(csv_lines)
        
        try:
            with open(file_path, 'w') as f:
                f.write(csv_content)
        except Exception as e:
            print(f"Error saving CSV: {e}")
            
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
    
    win = ThicknessScreen()
    
    # --- TEST HARNESS ---
    test_data = [
        {
            "path": "C:/path/to/your/image1.png", # <-- REPLACE WITH REAL PATH
            "microns_per_pixel": 2.57, 
            "contrast_clip_limit": 1.0,
            "analysis_height_mm": 2.0,
            "substrate_trace_left": [(0, 400), (5, 401)], # <-- ADD DUMMY TRACE
            "substrate_trace_right": [(499, 400), (494, 401)], # <-- ADD DUMMY TRACE
            "substrate_left_endpoint_index": -1,
            "substrate_right_endpoint_index": -1,
            "roi_x_pct": 0.0, "roi_y_pct": 70.0, "roi_w_pct": 30.0, "roi_h_pct": 30.0,
        }
    ]
    valid_test_data = [d for d in test_data if (
        os.path.exists(d["path"]) 
        and d.get("microns_per_pixel")
        and d.get("substrate_trace_left")
    )]
    if valid_test_data:
        win.load_image_data(valid_test_data)
    else:
        print("---")
        print("TEST HARNETT: Please edit thickness_screen.py")
        print("Add a real image path, 'microns_per_pixel', and 'substrate_trace_left/right' to 'test_data'.")
        print("---")

    win.resize(1200, 800)
    win.show()
    sys.exit(app.exec())