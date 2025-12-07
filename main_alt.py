from PySide6 import QtCore, QtGui, QtWidgets
import sys 
import os 
import uuid
from PySide6.QtWidgets import QProgressDialog, QLabel, QMessageBox

# --- Import screens (EXCLUDING Login, Save, Contrast, and DB) ---
from upload_screen_pyside6 import UploadScreen
from scale_finder_screen import ScaleFinderScreen
from substrate_screen import SubstrateScreen 
from analysis_menu_screen import AnalysisMenuScreen
from porosity_screen import PorosityScreen
from thickness_screen import ThicknessScreen
from custom_runner_screen import CustomRunnerScreen
from notch_screen import NotchScreen 

# NOTE: No database_manager import here. This is the offline version.

class AMAnalyzerAppOffline(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("AM Analyzer (Offline Edition)")
        self.resize(1400, 900)

        self.stack = QtWidgets.QStackedWidget()
        self.setCentralWidget(self.stack)

        # --- Data Properties ---
        self.image_data = [] 
        # No user_id or profile data needed in offline mode

        # -----------------------------
        # CREATE SCREENS
        # -----------------------------
        self.upload_screen = UploadScreen()
        self.scale_screen = ScaleFinderScreen()
        self.substrate_screen = SubstrateScreen() 
        # Contrast Screen Removed
        self.analysis_menu_screen = AnalysisMenuScreen()
        self.porosity_screen = PorosityScreen()
        self.thickness_screen = ThicknessScreen() 
        self.custom_runner_screen = CustomRunnerScreen() 
        self.notch_screen = NotchScreen() 
        
        # Add to stack
        self.stack.addWidget(self.upload_screen)
        self.stack.addWidget(self.scale_screen)
        self.stack.addWidget(self.substrate_screen) 
        # self.stack.addWidget(self.contrast_screen) <-- Removed
        self.stack.addWidget(self.analysis_menu_screen)
        self.stack.addWidget(self.porosity_screen)
        self.stack.addWidget(self.thickness_screen)
        self.stack.addWidget(self.custom_runner_screen) 
        self.stack.addWidget(self.notch_screen) 

        # -----------------------------
        # CONNECT SIGNALS (Offline Flow)
        # -----------------------------
        
        # 1. Upload -> Scale
        self.upload_screen.goNext.connect(self.goto_scale_screen)
        
        # 2. Scale <-> Substrate
        self.scale_screen.goPrev.connect(self.goto_upload_screen)
        self.scale_screen.goNext.connect(self.goto_substrate_screen) 

        # 3. Substrate <-> Analysis Menu (SKIPPING CONTRAST & SAVE)
        self.substrate_screen.goPrev.connect(self.goto_scale_screen) 
        self.substrate_screen.goNext.connect(self.goto_analysis_menu) # <-- Direct jump
        
        # 4. Analysis Menu Back Button
        self.analysis_menu_screen.goPrev.connect(self.goto_substrate_screen) # <-- Points back to Substrate
        
        # 5. Tools
        self.analysis_menu_screen.goToPorosity.connect(self.goto_porosity_screen) 
        self.analysis_menu_screen.goToThickness.connect(self.goto_thickness_screen)
        self.analysis_menu_screen.goToCustom.connect(self.goto_custom_runner_screen) 
        self.analysis_menu_screen.goToNotch.connect(self.goto_notch_screen) 
        
        # 6. Tool Back Buttons
        self.porosity_screen.goPrev.connect(self.goto_analysis_menu)
        self.thickness_screen.goPrev.connect(self.goto_analysis_menu) 
        self.custom_runner_screen.goPrev.connect(self.goto_analysis_menu) 
        self.notch_screen.goPrev.connect(self.goto_analysis_menu) 

        # Start at Upload (Skip Login)
        self.stack.setCurrentWidget(self.upload_screen)

    def show_critical_error(self, message):
        QMessageBox.critical(self, "Error", message)

    # -------------------------------------------------------
    # NAVIGATION FUNCTIONS
    # -------------------------------------------------------
    
    def goto_upload_screen(self):
        self.stack.setCurrentWidget(self.upload_screen)

    def goto_scale_screen(self):
        if self.sender() == self.upload_screen:
            if not self.upload_screen.files():
                QtWidgets.QMessageBox.warning(
                    self, "No Images Loaded", "Please load images before continuing."
                )
                return
            
            # Reset Data
            self.image_data = []
            
            for i, path in enumerate(self.upload_screen.files()):
                self.image_data.append({
                    "path": path, 
                    "microns_per_pixel": None,
                    # Default ROI: Top Right (Matches main app)
                    "roi_x_pct": 70.0, "roi_y_pct": 0.0, "roi_w_pct": 30.0, "roi_h_pct": 10.0, 
                    "manual_val": "",
                    "manual_unit": "µm",
                    "contrast_clip_limit": 1.0,
                    "analysis_height_mm": 2.0,
                    "substrate_trace_left": None,
                    "substrate_trace_right": None,
                    "substrate_left_endpoint_index": -1,
                    "substrate_right_endpoint_index": -1,
                    "substrate_is_manual": False,
                    "manual_substrate_points": [],
                    "porosity_percent": None,
                    "total_pore_area": None,
                    "total_matrix_area": None,
                    "thickness_results": None,
                    "thickness_interval_mm": 1.0,
                    "thickness_total_height_mm": 2.0,
                    "wasted_material_ratio": None,
                    # Offline Defaults
                    "project_name": "Offline Project",
                    "lab_name": "",
                    "school_name": "",
                    "user_name": "Local User",
                    "image_name": os.path.basename(path),
                    "image_desc": ""
                })
            
            self.scale_screen.load_image_data(self.image_data)
        self.stack.setCurrentWidget(self.scale_screen)

    def goto_substrate_screen(self):
        self.substrate_screen.load_image_data(self.image_data)
        self.stack.setCurrentWidget(self.substrate_screen)
        
    def goto_analysis_menu(self):
        # In offline mode, we just load the data directly
        self.analysis_menu_screen.load_image_data(self.image_data)
        self.stack.setCurrentWidget(self.analysis_menu_screen)

    def goto_porosity_screen(self):
        self.porosity_screen.load_image_data(self.image_data)
        self.stack.setCurrentWidget(self.porosity_screen)
        
    def goto_thickness_screen(self):
        self.thickness_screen.load_image_data(self.image_data)
        self.stack.setCurrentWidget(self.thickness_screen)
        
    def goto_custom_runner_screen(self): 
        self.custom_runner_screen.load_image_data(self.image_data)
        self.stack.setCurrentWidget(self.custom_runner_screen)
        
    def goto_notch_screen(self): 
        self.notch_screen.load_image_data(self.image_data)
        self.stack.setCurrentWidget(self.notch_screen)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    app.setStyle("Fusion")
    dark_palette = QtGui.QPalette()
    dark_palette.setColor(QtGui.QPalette.Window, QtGui.QColor(23, 23, 23))
    dark_palette.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
    dark_palette.setColor(QtGui.QPalette.Base, QtGui.QColor(30, 30, 30))
    dark_palette.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
    dark_palette.setColor(QtGui.QPalette.Button, QtGui.QColor(45, 45, 45))
    dark_palette.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white)
    app.setPalette(dark_palette)

    win = AMAnalyzerAppOffline()
    win.show()
    sys.exit(app.exec())