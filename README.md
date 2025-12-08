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
  - **EDM Cut Mode**: Uses a center-out pixel scanning algorithm to trace a flat bottom surface
  - **Manual Mode**: Manually select points along the substrate to define a substrate line

### 2. Advanced Analysis Tools

- **Porosity Quantification**:
  - Binarized using Otsu's binarization, can adjust contrast using CLAHE
  - Can filter pores by size, sphericity
  - Height-interval slicing (e.g., measure porosity every 1mm).

- **Thickness Profiling**:
  - Measures the thickness of the build against hight. For example, you can measure the thickness of the sample every 0.2 mm up to 5 mm
  - Can measure excess deposited material: If you were to machine the sample down to a uniform thickness, what percent of the material deposited would be wasted?

- **Notch/Waisting Detection** (Not complete):
  - Tries to detect notches in the side of the deposit

### 3. Extensible Plugin Engine 🔌

The **"Custom Script"** feature allows researchers to extend the platform without modifying the source code.

- Users can upload raw Python scripts (`.py`).
- The app automatically generates a GUI for the script (sliders, dropdowns, inputs) based on the code structure.
- Documentation included on the page
- Scripts receive pre-processed, calibrated image data instantly.

---

## 🛠️ Installation

### Prerequisites

**Python 3.10+**

### Installing Tesseract OCR (Crucial Step) ⚠️

This application uses Tesseract to read scale bars from microscopy images. **You must install the OCR engine separately from the Python libraries.** Installing `pytesseract` via pip is NOT enough - it's just a Python wrapper.

**For Windows Users:**

1. **Download the Installer**: Go to [UB Mannheim's GitHub](https://github.com/UB-Mannheim/tesseract/wiki) and download the latest `tesseract-ocr-w64-setup.exe`.

2. **Run the Installer**.

3. **⚠️ IMPORTANT**: During installation, make sure to:
   - Either check the box that says **"Add Tesseract to PATH"**
   - OR ensure the install location is the default: `C:\Program Files\Tesseract-OCR`
   
4. If you installed it to a custom location and did not add it to your PATH, the app may not find it automatically and will show an error.

**For Mac Users (Homebrew):**

```bash
brew install tesseract
```

**For Linux Users:**

```bash
sudo apt-get install tesseract-ocr
```

**Verification**: After installation, open a new terminal/command prompt and type:
```bash
tesseract --version
```
If you see version information, Tesseract is correctly installed!

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

- **Cloud Integration**: Syncing metadata and images to Google Firebase / Firestore.
- **Foundation Models**: Integrating Segment Anything Model (SAM) or YOLO for zero-shot defect detection without manual thresholding.
- **API**: Releasing a headless Python wrapper for batch-processing images on clusters.

---


**Built by Aryan Thacker for the advancement of Additive Manufacturing research.**
