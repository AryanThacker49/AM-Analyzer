# upload_screen_pyside6.py
from PySide6 import QtCore, QtGui, QtWidgets
import os
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

class UploadScreen(QtWidgets.QWidget):
    goNext = QtCore.Signal()

    filesChanged = QtCore.Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._files = []
        self._init_ui()
        self.setAcceptDrops(True)

    def _init_ui(self):
        main_l = QtWidgets.QVBoxLayout(self)
        main_l.setContentsMargins(12,12,12,12) # Margins on sides
        main_l.setSpacing(12) # Vertical spacing between elements
        # FORCE the central section to expand in stacked widget mode
        main_l.setStretch(1, 2)   # big middle area (drop zone)
        main_l.setStretch(2, 1)   # file list area


        title = QtWidgets.QLabel("AM Analyzer — Load Images")
        title.setObjectName("titleLabel")
        title.setAlignment(QtCore.Qt.AlignCenter)
        # Apply title-specific styling if needed, but not colors
        title.setStyleSheet("font-size: 24px; font-weight: bold; padding: 10px;")
        main_l.addWidget(title)

        centre_container = QtWidgets.QWidget()
        centre_layout = QtWidgets.QHBoxLayout(centre_container)
        centre_layout.addStretch(1) #Where is the center dashed box located
        centre_container.setContentsMargins(12,12,12,12)

        self.drop_zone = QtWidgets.QLabel("Drag & Drop Images Here\nor Click to Browse")
        self.drop_zone.setAlignment(QtCore.Qt.AlignCenter)
        self.drop_zone.setObjectName("dropZone")
        # Style the drop zone specifically, as it's a custom element
        self.drop_zone.setStyleSheet("""
            #dropZone {
                background: #2E2E2E;
                border: 2px dashed #555;
                border-radius: 8px;
                padding: 12px;
                font-size: 16px;
            }
            #dropZone:hover {
                background: #3A3A3A;
                border-color: #777;
            }
        """)
        self.drop_zone.setMinimumHeight(180)
        self.drop_zone.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.drop_zone.mousePressEvent = self._on_dropzone_clicked

        centre_layout.addWidget(self.drop_zone)
        centre_layout.addStretch(1)
        main_l.addWidget(centre_container, stretch=1)

        file_list_container = QtWidgets.QWidget()
        fl = QtWidgets.QHBoxLayout(file_list_container)

        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.setIconSize(QtCore.QSize(96,64))
        fl.addWidget(self.list_widget, stretch=1)

        side = QtWidgets.QVBoxLayout()
        btn_remove = QtWidgets.QPushButton("Remove Selected")
        btn_remove.clicked.connect(self.remove_selected)
        btn_clear = QtWidgets.QPushButton("Clear All")
        btn_clear.clicked.connect(self.clear)
        btn_next = QtWidgets.QPushButton("Next ➜")
        
        # --------------------------------------------------
        # --- FIX 1: Connect the 'Next' button
        # --------------------------------------------------
        btn_next.clicked.connect(self._on_next_clicked)


        side.addWidget(btn_remove)
        side.addWidget(btn_clear)
        side.addStretch()
        side.addWidget(btn_next)
        fl.addLayout(side)

        main_l.addWidget(file_list_container)

        self.status = QtWidgets.QLabel("No images loaded.")
        self.status.setObjectName("statusLabel")
        main_l.addWidget(self.status)
    
    # --- FIX 2: Removed _apply_styles() method ---
    # The global palette in main.py will handle styling.

    def files(self): return list(self._files)

    def clear(self):
        self._files = []
        self.list_widget.clear()
        self.status.setText("No images loaded.")
        self.filesChanged.emit(self.files())

    def remove_selected(self):
        for it in self.list_widget.selectedItems():
            path = it.data(QtCore.Qt.UserRole)
            if path in self._files:
                self._files.remove(path)
            self.list_widget.takeItem(self.list_widget.row(it))
        self._update_status()
        self.filesChanged.emit(self.files())

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            for u in e.mimeData().urls():
                if Path(u.toLocalFile()).suffix.lower() in IMAGE_EXTS:
                    e.acceptProposedAction()
                    return
        e.ignore()

    def dragMoveEvent(self, e): e.acceptProposedAction()

    def dropEvent(self, e):
        for u in e.mimeData().urls():
            p = u.toLocalFile()
            if Path(p).suffix.lower() in IMAGE_EXTS:
                self._add_file(p)
        self.filesChanged.emit(self.files())

    def _add_file(self, path):
        path = os.path.abspath(path)
        if path in self._files: return
        self._files.append(path)

        it = QtWidgets.QListWidgetItem(os.path.basename(path))
        it.setData(QtCore.Qt.UserRole, path)
        pix = QtGui.QPixmap(path)
        if not pix.isNull():
            pix = pix.scaled(192,128, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            it.setIcon(QtGui.QIcon(pix))
        self.list_widget.addItem(it)
        self._update_status()

    def _update_status(self):
        n = len(self._files)
        self.status.setText(f"{n} images loaded." if n else "No images loaded.")

    def _on_dropzone_clicked(self, event):
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Select images")
        for f in files: self._add_file(f)
        self.filesChanged.emit(self.files())

    def _on_next_clicked(self):
        self.goNext.emit()


# test harness
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    
    # Use the same dark palette from main.py for testing
    app.setStyle("Fusion")
    dark_palette = QtGui.QPalette()
    dark_palette.setColor(QtGui.QPalette.Window, QtGui.QColor(23, 23, 23))
    dark_palette.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
    dark_palette.setColor(QtGui.QPalette.Base, QtGui.QColor(30, 30, 30))
    dark_palette.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
    dark_palette.setColor(QtGui.QPalette.Button, QtGui.QColor(45, 45, 45))
    dark_palette.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white)
    app.setPalette(dark_palette)
    
    win = QtWidgets.QMainWindow()
    win.setCentralWidget(UploadScreen())
    win.resize(1100,700)
    win.show()
    sys.exit(app.exec())