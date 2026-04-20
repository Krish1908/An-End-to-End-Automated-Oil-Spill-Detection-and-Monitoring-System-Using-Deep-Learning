# 🛢️ An End-to-End Automated Oil Spill Detection and Monitoring System Using Deep Learning

> **An end-to-end deep learning pipeline for oil spill detection, segmentation, and visualization using satellite imagery**

---

## 🎯 Overview

This project implements a **multi-stage deep learning framework** for automated oil spill detection in satellite imagery. The system combines **CNN classification**, **U-Net segmentation**, and **YOLO object detection** to provide accurate, real-time oil spill analysis for both **RGB (optical)** and **SAR (radar)** imagery.

### Key Capabilities
- ✅ Binary classification (oil spill vs clean water)
- ✅ Pixel-level segmentation
- ✅ Object detection with bounding boxes
- ✅ Dual-mode support (RGB/SAR)
- ✅ Interactive web dashboard
- ✅ Downloadable analysis reports

---

## 🚀 Quick Start

### Prerequisites
- Python 3.12
- CUDA-compatible GPU (optional, for faster inference)

### Installation
```bash
# Clone repository
git clone <repo-url>
cd OIL-SPILL

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Running the Dashboard
```bash
streamlit run src/dashboard/app-st.py
```
Access the app at `http://localhost:8501`

---

## 🏗️ Architecture

![WorkFlow Architecture](https://github.com/Krish1908/An-End-to-End-Automated-Oil-Spill-Detection-and-Monitoring-System-Using-Deep-Learning/blob/main/OIL-SPILL.png)

---

## 🎨 Dashboard Features

### Interactive Controls
- **Model Selection**: Toggle between RGB and SAR models
- **Threshold Adjustment**: Fine-tune detection sensitivity
- **Visualization Options**: Toggle boundaries and overlays
- **Real-time Analysis**: Upload and process images instantly

### Output Components
1. **Original Image** - Uploaded satellite imagery
2. **Segmentation Mask** - U-Net pixel-level output
3. **Overlay Visualization** - Red fill + boundary outlines
4. **Metrics Dashboard** - Oil coverage, confidence scores
5. **Downloadable Report** - PNG with analysis summary

### Color Legend
| Element | Meaning |
|---------|---------|
| 🟥 Red Fill | Oil spill region |
| ⬜ White Mask | Segmentation output |
| ⬛ Black Outline | Oil boundary |

---

## 📂 Project Structure

```
oil-spill/
├── 📄 README.md                 # This file
├── 📄 requirements.txt          # Python dependencies
├── 📁 src/
│   ├── 📁 dashboard/
│   │   └── 📄 app-st.py        # Streamlit web interface
│   ├── 📁 data/
│   │   ├── 📄 dataloader.py    # Standard data loader
│   │   ├── 📄 preprocess.py    # Image preprocessing
│   │   └── 📄 domain_adaptation.py
│   ├── 📁 models/
│   │   ├── 📄 cnn.py           # CNN architecture
│   │   └── 📄 unet.py          # U-Net architecture
│   ├── 📁 post-processing/
│   │   └── 📄 predict_pipeline.py
│   ├── 📁 training/
│   │   ├── 📄 train_cnn.py
│   │   ├── 📄 train_unet.py
│   │   └── 📄 train_unet_sar.py
│   └── 📁 testing/
│       ├── 📄 test_cnn.py
│       └── 📄 test_unet.py
├── 📁 models-rgb/              # Trained RGB models
├── 📁 models-sar/              # Trained SAR models
```

---

## 🛠️ Usage Examples

### Web Interface (Recommended)
```bash
# Launch dashboard
streamlit run src/dashboard/app-st.py

# Features:
# 1. Select RGB or SAR model
# 2. Upload satellite image
# 3. Adjust detection thresholds
# 4. View real-time analysis
# 5. Download reports
```

### Training Custom Models
```bash
# Train CNN classifier
python src/training/train_cnn.py \
    --data-dir dataset_1/ \
    --epochs 50 \
    --batch-size 32

# Train U-Net segmentation
python src/training/train_unet.py \
    --data-dir dataset_1/ \
    --epochs 100 \
    --batch-size 16
```

---

## 📊 Datasets

### Supported Datasets
- **Dataset 1**: Primary RGB optical dataset (1920×1080)
- **Dataset 2**: Secondary dataset for augmentation
- **Dataset 3**: SAR (Synthetic Aperture Radar) imagery
- **Dataset 4**: Additional SAR dataset

### Data Source
This project uses publicly available research datasets:
- [Annotated RGB Dataset](https://zenodo.org/records/10555314)
- [Sentinel-1 SAR Data](https://zenodo.org/records/15298010/)

---

## ⚙️ Configuration

### Detection Thresholds (Dashboard)
```python
CNN_THRESHOLD = 0.5        # Classification confidence
UNET_THRESHOLD = 0.4       # Segmentation threshold
YOLO_CONF = 0.5           # Detection confidence
MIN_AREA = 600            # Minimum spill size (pixels)
```

### Visualization Settings
```python
ALPHA = 0.4               # Overlay transparency
SHOW_OUTLINE = True       # Boundary outline toggle
OUTLINE_COLOR = (0,0,0)   # Black outline (day SAR)
```

---

## 📈 Results & Analysis

### Output Metrics
For each analyzed image, the system provides:

1. **Oil Coverage (%)** - Percentage of image covered by oil
2. **CNN Confidence** - Probability of oil spill presence
3. **Spill Regions** - Number of distinct oil areas
4. **Largest Region** - Size of biggest detected spill
5. **YOLO Detections** - Count of bounding box detections

### Sample Output


![Sample Output - 1](https://github.com/Krish1908/An-End-to-End-Automated-Oil-Spill-Detection-and-Monitoring-System-Using-Deep-Learning/blob/main/oil_spill_report_sar.png)

![Sample Output - 2](https://github.com/Krish1908/An-End-to-End-Automated-Oil-Spill-Detection-and-Monitoring-System-Using-Deep-Learning/blob/main/oil_spill_report_rgb.png)

---
