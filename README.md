# AM Analyzer 🔬

**Automated Material Analysis Platform for Additive Manufacturing**

## 📋 Overview

- 🎥 [Watch this video for the motivation behind this tool](https://youtu.be/SgbJ0etzegI?si=UcMJq1X7ylWcnvDw)
- 🎬 [Watch this video for a demo of the tool](https://youtu.be/kPJHWkSzm0k?si=EoiFM9cYG4g-TIni)

AM Analyzer is a modular desktop platform designed to standardize and automate the analysis of microscopy images for 3D-printed (Additive Manufacturing) metals.

Traditionally, researchers spend hours manually cropping substrates, calibrating scale bars, and measuring porosity using generic tools like ImageJ. AM Analyzer replaces this manual workflow with a computer vision pipeline that processes images in seconds.

This repository contains the **Offline Edition** of the platform, allowing researchers to run the full analysis suite locally on their machines.

---

## 🚀 Key Features

### 1. Automated Pre-Processing

- **Smart Scale Detection**: Uses OCR (Tesseract) and contour analysis to automatically find the scale bar, read the text (e.g., "200 µm"), and calibrate the image pixel-to-micron ratio.

- **Intelligent Substrate Removal**:
  - **Standard Mode**: Detects the "jump" where the material meets the substrate using a bottom-up scan.
  - **EDM Cut Mode**: Uses a center-out pixel scanning algorithm to trace cut surfaces.

### 2. Advanced Analysis Tools

- **Porosity Quantification**:
  - Automated segmentation using CLAHE (Contrast Limited Adaptive Histogram Equalization) and Otsu's Binarization.
  - Filters for pore size and sphericity to remove noise.
  - Height-interval slicing (e.g., measure porosity every 1mm).

- **Thickness Profiling**: Measures the build height relative to the substrate across the entire width of the sample.

- **Notch/Waisting Detection**: Uses derivative analysis (1st, 2nd, and 3rd derivatives of the edge profile) to mathematically identify geometric inconsistencies and notches.

### 3. Extensible Plugin Engine 🔌

The **"Custom Script"** feature allows researchers to extend the platform without modifying the source code.

- Users can upload raw Python scripts (`.py`).
- The app automatically generates a GUI for the script (sliders, dropdowns, inputs) based on the code structure.
- Scripts receive pre-processed, calibrated image data instantly.

---

## 🛠️ Installation

### Prerequisites

- **Python 3.10+**
- **Tesseract OCR** installed on your system:
  - **Windows**: [Download Installer](https://github.com/UB-Mannheim/tesseract/wiki) (Add to PATH)
  - **Mac**: `brew install tesseract`
  - **Linux**: `sudo apt-get install tesseract-ocr`

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/AM-Analyzer.git
   cd AM-Analyzer
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
   **Note**: Key libraries include `PySide6`, `opencv-python`, `numpy`, `scipy`, `matplotlib`, and `pytesseract`.

---

## 🖥️ Usage

Run the offline controller to start the application:

```bash
python main_alt.py
```

### The Workflow

1. **Upload**: Drag and drop your `.tif`, `.png`, or `.jpg` microscopy images.

2. **Scale**: The app will attempt to auto-detect the scale bar. You can verify it or draw a manual box to re-run OCR on a specific region.

3. **Substrate**: Auto-detect the interface line. Use the "Manual Adjustment" tools to fine-tune endpoints if the sample is irregular.

4. **Contrast**: Adjust the contrast slider to highlight pores against the metal matrix.

5. **Analysis Menu**: Choose your tool:
   - **Porosity**: Get % density, pore count, and shape factors.
   - **Thickness**: View height maps.
   - **Notch Detection**: Analyze edge consistency.
   - **Custom Script**: Run your own Python logic.

---

## 🧩 Writing Custom Scripts

AM Analyzer enables **"Low-Code"** tool creation. Create a python file that inherits from `CustomScript`:

```python
from custom_script_base import CustomScript
import cv2

class MyCustomTool(CustomScript):
    def setup_ui(self):
        # This automatically creates a Slider in the App UI
        self.add_slider("Blur Strength", 1, 20, 5)

    def run(self, image, metadata, params):
        # Your logic here
        val = params["Blur Strength"]
        blurred = cv2.GaussianBlur(image, (val, val), 0)
        
        # Return image to display and dict of results to export
        return blurred, {"Blur Used": val}
```

---

## 🏗️ Architecture

- **Frontend**: PySide6 (Qt for Python) with custom painted widgets for high-performance image interaction.
- **Image Processing**: OpenCV (`cv2`) and NumPy for vectorized matrix operations.
- **Math**: SciPy for signal processing (peak finding in notch detection).
- **Graphs**: Matplotlib backend embedded directly into Qt widgets.

---

## 🔮 Future Roadmap

- **Cloud Integration**: (Currently available in the Enterprise version) Syncing metadata and images to Google Firebase / Firestore.
- **Foundation Models**: Integrating Segment Anything Model (SAM) or YOLO for zero-shot defect detection without manual thresholding.
- **API**: Releasing a headless Python wrapper for batch-processing images on clusters.

---


**Built by Aryan Thacker for the advancement of Additive Manufacturing research.**
