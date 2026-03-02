# LAVR Prediction
The code is used for the prognostic analysis of Left Atrial Volume and Remodeling (LAVR) in cardiac magnetic resonance (CMR) imaging.

## Environment Setup

### 1. Python Version Requirement
This project is developed and tested based on **Python 3.8 ~ 3.11** (Python 3.11.5 is strongly recommended). It is **not compatible** with Python 2.x and Python 3.7 or lower versions (core dependencies such as torch 2.1.1 have limited support for Python 3.7).

### 2. Quick Dependency Installation
```bash
# 1. Verify Python version (virtual environment is recommended first)
python --version  # Expected output: 3.8.x/3.9.x/3.10.x/3.11.5

# 2. Create a virtual environment (optional but recommended)
conda create -n cmr_lavr python=3.11.5
conda activate cmr_lavr

# 3. Install dependencies
pip install -r requirements.txt

### Try Example
python main/main.py


