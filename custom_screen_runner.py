import sys
import os
import ast
import importlib.util
import inspect
import cv2
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QHBoxLayout, QFrame, QPushButton, 
    QGridLayout, QLineEdit, QSizePolicy, QApplication, QSlider,
    QProgressDialog, QFileDialog, QComboBox, QCheckBox, QScrollArea
)
from PySide6.QtGui import QPixmap, Qt, QImage, QPainter, QPen, QColor
from PySide6 import QtCore, QtGui

from custom_script_base import CustomScript

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
# SAFETY VALIDATOR
# =====================================================================
def validate_script_safety(file_path):
    """
    Parses the python script looking for dangerous imports or calls.
    Returns (True, "") if safe, or (False, error_message) if unsafe.
    """
    with open(file_path, "r") as f:
        source = f.read()

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return False, f"Syntax Error in script: {e}"

    # List of forbidden modules
    forbidden_modules = {'os', 'sys', 'subprocess', 'shutil', 'pickle', 'socket', 'http', 'urllib'}
    # List of allowed modules (whitelist) - not strictly enforced but good for reference
    # allowed = {'numpy', 'cv2', 'math', 'scipy', 'custom_script_base'}

    for node in ast.walk(tree):
        # Check imports (import x)
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split('.')[0] in forbidden_modules:
                    return False, f"Forbidden import detected: '{alias.name}'"
        
        # Check imports (from x import y)
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split('.')[0] in forbidden_modules:
                return False, f"Forbidden import detected: 'from {node.module} ...'"

        # Check for dangerous function calls (e.g., eval, exec, open)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in {'eval', 'exec', 'open', 'compile'}:
                    return False, f"Forbidden function call detected: '{node.func.id}()'"

    return True, "Safe"

# =====================================================================
# MAIN SCREEN CLASS
# =====================================================================
class CustomRunnerScreen(QWidget):

    goPrev = QtCore.Signal()
    
    def __init__(self, controller=None):
        super().__init__()
        self.controller = controller
        
        self.image_data = [] 
        self.current_index = 0
        
        self.current_script_instance = None
        self.current_ui_widgets = {} # Stores references to the dynamic input widgets
        self.analysis_results = {} # Stores results for export

        self._build_ui()

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        title = QLabel("Custom Analysis Runner")
        title.setStyleSheet("font-size: 24px; font-weight: bold;")
        main_layout.addWidget(title)

        content_layout = QHBoxLayout()
        content_layout.setSpacing(20)

        # -------- Left: Image Preview --------
        image_frame = QFrame()
        image_layout = QVBoxLayout(image_frame)
        
        self.main_image_label = AspectRatioLabel("Load a script to begin")
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

        # -------- Right: Dynamic Controls --------
        right_frame = QFrame()
        right_frame.setMaximumWidth(400) 
        right_layout = QVBoxLayout(right_frame)
        right_layout.setSpacing(15)
        right_layout.setAlignment(Qt.AlignTop)

        # Load Script Button
        self.load_script_btn = QPushButton("Load Custom Script (.py)")
        self.load_script_btn.setStyleSheet("background-color: #444; padding: 8px;")
        self.load_script_btn.clicked.connect(self._on_load_script)
        right_layout.addWidget(self.load_script_btn)
        
        # Script Name Label
        self.script_name_label = QLabel("No script loaded")
        self.script_name_label.setStyleSheet("color: #888; font-style: italic;")
        right_layout.addWidget(self.script_name_label)

        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        right_layout.addWidget(line)

        # --- Dynamic Inputs Container ---
        # We use a ScrollArea in case the user adds 50 inputs
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        
        self.dynamic_container = QWidget()
        self.dynamic_layout = QVBoxLayout(self.dynamic_container)
        self.dynamic_layout.setAlignment(Qt.AlignTop)
        self.dynamic_layout.setSpacing(10)
        
        self.scroll_area.setWidget(self.dynamic_container)
        right_layout.addWidget(self.scroll_area, stretch=1)
        
        # --- Run / Export ---
        self.run_btn = QPushButton("Run Analysis")
        self.run_btn.setEnabled(False)
        self.run_btn.setStyleSheet("background-color: #0B3C5D; font-weight: bold; padding: 10px;")
        self.run_btn.clicked.connect(self._run_analysis_current)
        right_layout.addWidget(self.run_btn)
        
        self.export_btn = QPushButton("Export Results (CSV)")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._on_export_csv)
        right_layout.addWidget(self.export_btn)

        content_layout.addWidget(right_frame, stretch=1)
        main_layout.addLayout(content_layout, stretch=1)

        # --- Bottom navigation ---
        bottom = QHBoxLayout()
        self.prev_btn = QPushButton("Prev: Menu")
        self.prev_btn.setFixedHeight(50)
        self.prev_btn.clicked.connect(self.goPrev.emit)
        bottom.addWidget(self.prev_btn)
        bottom.addStretch()
        main_layout.addLayout(bottom)
        
    # =================================================================
    # SCRIPT LOADING & UI BUILDING
    # =================================================================
    def _on_load_script(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Python Script", "", "Python Files (*.py)")
        if not file_path:
            return
            
        # 1. Run Safety Check
        is_safe, message = validate_script_safety(file_path)
        if not is_safe:
            QMessageBox.critical(self, "Security Alert", f"Script blocked for safety reasons:\n{message}")
            return

        # 2. Import the module dynamically
        try:
            spec = importlib.util.spec_from_file_location("user_script", file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # 3. Find the CustomScript subclass
            script_class = None
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, CustomScript) and obj is not CustomScript:
                    script_class = obj
                    break
            
            if script_class is None:
                QMessageBox.warning(self, "Error", "Could not find a class that inherits from 'CustomScript' in this file.")
                return
                
            # 4. Instantiate and Setup UI
            self.current_script_instance = script_class()
            self.current_script_instance.setup_ui() # This populates self.current_script_instance.inputs
            
            self._build_dynamic_ui(self.current_script_instance.inputs)
            
            self.script_name_label.setText(os.path.basename(file_path))
            self.run_btn.setEnabled(True)
            self.export_btn.setEnabled(True)
            
        except Exception as e:
            QMessageBox.critical(self, "Error Loading Script", str(e))

    def _build_dynamic_ui(self, input_defs):
        """Clears the dynamic area and builds widgets based on the script definitions."""
        # Clear old widgets
        for i in reversed(range(self.dynamic_layout.count())): 
            self.dynamic_layout.itemAt(i).widget().setParent(None)
        self.current_ui_widgets = {}

        for inp in input_defs:
            label_text = inp['label']
            default = inp['default']
            
            # Create Label
            self.dynamic_layout.addWidget(QLabel(label_text))
            
            # Create Input Widget
            if inp['type'] == 'number' or inp['type'] == 'text':
                widget = QLineEdit(str(default))
                self.current_ui_widgets[label_text] = widget
                self.dynamic_layout.addWidget(widget)
                
            elif inp['type'] == 'bool':
                widget = QCheckBox(label_text) # Label is on the box
                widget.setChecked(bool(default))
                self.current_ui_widgets[label_text] = widget
                self.dynamic_layout.addWidget(widget)
                
            elif inp['type'] == 'dropdown':
                widget = QComboBox()
                widget.addItems(inp['options'])
                widget.setCurrentIndex(int(default))
                self.current_ui_widgets[label_text] = widget
                self.dynamic_layout.addWidget(widget)
                
            elif inp['type'] == 'slider':
                h_layout = QHBoxLayout()
                slider = QSlider(Qt.Horizontal)
                slider.setRange(inp['min'], inp['max'])
                slider.setValue(int(default))
                
                # Add a little label next to it to show value
                val_label = QLabel(str(default))
                val_label.setFixedWidth(30)
                slider.valueChanged.connect(lambda v, l=val_label: l.setText(str(v)))
                
                h_layout.addWidget(slider)
                h_layout.addWidget(val_label)
                
                self.current_ui_widgets[label_text] = slider
                container = QWidget()
                container.setLayout(h_layout)
                self.dynamic_layout.addWidget(container)

        self.dynamic_layout.addStretch()

    def _get_params_from_ui(self):
        """Reads the current values from all dynamic widgets."""
        params = {}
        input_defs = self.current_script_instance.inputs
        
        for inp in input_defs:
            label = inp['label']
            widget = self.current_ui_widgets[label]
            
            if inp['type'] == 'number':
                try:
                    params[label] = float(widget.text())
                except:
                    params[label] = 0.0
            elif inp['type'] == 'text':
                params[label] = widget.text()
            elif inp['type'] == 'bool':
                params[label] = widget.isChecked()
            elif inp['type'] == 'dropdown':
                params[label] = widget.currentText()
            elif inp['type'] == 'slider':
                params[label] = widget.value()
                
        return params

    # =================================================================
    # RUNNING
    # =================================================================
    def _run_analysis_current(self):
        if not self.current_script_instance or not self.image_data:
            return
            
        data = self.image_data[self.current_index]
        path = data.get('path')
        if not path or not os.path.exists(path): return
        
        # Load Image
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        
        # Get Params
        params = self._get_params_from_ui()
        
        try:
            # RUN USER CODE
            processed_img, result_data = self.current_script_instance.run(image, data, params)
            
            # Display Image
            self.main_image_label.setPixmap(self._convert_cv_to_pixmap(processed_img))
            
            # Store results
            self.analysis_results[os.path.basename(path)] = result_data
            
            # Show feedback
            # (Ideally we'd show this in the UI, but console for now)
            print(f"Custom Analysis Result: {result_data}")
            
        except Exception as e:
            QMessageBox.critical(self, "Script Error", f"Error running script:\n{e}")

    def _convert_cv_to_pixmap(self, cv_img_bgr):
        """Converts BGR OpenCV image to QPixmap."""
        if cv_img_bgr is None: return QPixmap()
        
        # Handle grayscale
        if len(cv_img_bgr.shape) == 2:
            h, w = cv_img_bgr.shape
            qt_img = QImage(cv_img_bgr.data, w, h, w, QImage.Format_Grayscale8)
            return QPixmap.fromImage(qt_img)
            
        h, w, ch = cv_img_bgr.shape
        bytes_per_line = ch * w
        qt_img = QImage(cv_img_bgr.data, w, h, bytes_per_line, QImage.Format_BGR888).rgbSwapped()
        return QPixmap.fromImage(qt_img)

    def _on_export_csv(self):
        """Runs the script on ALL images and saves results."""
        if not self.image_data or not self.current_script_instance: return
        
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Custom Results", "", "CSV Files (*.csv)")
        if not file_path: return
        
        params = self._get_params_from_ui()
        progress = QProgressDialog("Running custom script on all images...", "Cancel", 0, len(self.image_data), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setValue(0)
        
        results_list = []
        keys = set()
        
        for i, data in enumerate(self.image_data):
            progress.setValue(i)
            QApplication.processEvents()
            if progress.wasCanceled(): return
            
            path = data.get('path')
            if not path or not os.path.exists(path): continue
            
            try:
                image = cv2.imread(path, cv2.IMREAD_COLOR)
                _, res_dict = self.current_script_instance.run(image, data, params)
                
                # Add filename to dict
                res_dict['Filename'] = os.path.basename(path)
                results_list.append(res_dict)
                keys.update(res_dict.keys())
            except Exception as e:
                print(f"Error on {path}: {e}")
                
        progress.setValue(len(self.image_data))
        
        # Write CSV
        # Ensure 'Filename' is first
        header = ['Filename'] + sorted([k for k in keys if k != 'Filename'])
        
        with open(file_path, 'w') as f:
            f.write(",".join(header) + "\n")
            for res in results_list:
                row = [str(res.get(k, "")) for k in header]
                f.write(",".join(row) + "\n")

    # =================================================================
    # DATA MODEL & NAVIGATION
    # =================================================================
    def load_image_data(self, image_data_list):
        self.image_data = image_data_list
        self.current_index = 0
        if not self.image_data: return
        
        # Load first image preview (original)
        path = self.image_data[0]['path']
        self.main_image_label.setPixmap(QPixmap(path))
        
        self.update_nav_buttons()

    def next_image(self):
        if not self.image_data: return
        if self.current_index < len(self.image_data) - 1:
            self.current_index += 1
            # Reload with current script if active
            if self.current_script_instance:
                self._run_analysis_current()
            else:
                path = self.image_data[self.current_index]['path']
                self.main_image_label.setPixmap(QPixmap(path))

    def prev_image(self):
        if not self.image_data: return
        if self.current_index > 0:
            self.current_index -= 1
            if self.current_script_instance:
                self._run_analysis_current()
            else:
                path = self.image_data[self.current_index]['path']
                self.main_image_label.setPixmap(QPixmap(path))

    def update_nav_buttons(self):
        if not self.image_data:
            self.prev_img_btn.setEnabled(False)
            self.next_img_btn.setEnabled(False)
            return
        self.prev_img_btn.setEnabled(self.current_index > 0)
        self.next_img_btn.setEnabled(self.current_index < len(self.image_data) - 1)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = CustomRunnerScreen()
    win.resize(1200, 800)
    win.show()
    sys.exit(app.exec())

    