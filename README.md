# LAVR Prediction

This repository provides code for the prognostic analysis of **Left Atrial Volume and Remodeling (LAVR)** using **Cardiac Magnetic Resonance (CMR)** imaging. The pipeline includes preprocessing, inference with a pretrained deep learning model, and quantitative assessment of LA remodeling.

> 📌 **Note**: Large files (e.g., sample CMR image and model weights) are managed via **Git LFS**. See [Data & Model](#-data--model) for details.

---

## 🛠️ Environment Setup

### 1. Python Version Requirement
This project is developed and tested on **Python 3.8 – 3.11**.  
✅ **Python 3.11.5 is strongly recommended**.  
❌ Not compatible with Python ≤ 3.7 (due to `torch>=2.1.1` dependency).

### 2. Create Virtual Environment (Recommended)
Using conda
conda create -n cmr_lavr python=3.11.5
conda activate cmr_lavr

Or using venv (if you prefer)
python -m venv cmr_lavr_env
source cmr_lavr_env/bin/activate  # Linux/macOS
cmr_lavr_env\Scripts\activate   # Windows

### 3. Install Dependencies
pip install -r requirements.txt

### 4. Data & Model (Git LFS)
Install Git LFS (if not already installed):
macOS
brew install git-lfs

Ubuntu/Debian
sudo apt-get install git-lfs

Clone the repository (LFS files auto-download):
git lfs install          # Run once per machine
git clone https://github.com/qutaiping/CMR-LVAR-Prediction.git
cd CMR-LVAR-Prediction

After cloning, verify file sizes:
ls -lh main/demo/img/1.nii.gz      # Should be ~64 MB
ls -lh train/checkpoints/model.pth # Should be ~152 MB

### 5. Quick Start
python main/main.py

### 6.  Project Structure
CMR-LVAR-Prediction/
├── main/
│   ├── main.py                 # Entry point for inference
│   └── demo/
│       └── img/                # Sample CMR data (.nii.gz)
├── train/
│   └── checkpoints/            # Pretrained model (.pth)
├── requirements.txt            # Python dependencies
└── README.md
