# Skin Cancer Classifier (Benign vs. Malignant)

## Overview

This project is a **machine‑learning binary classifier** that determines whether a dermoscopic image shows **benign (0)** or **malignant (1)** skin lesions. It evaluates a simple CNN baseline and four transfer‑learning backbones (VGG16, ResNet50, InceptionV3, DenseNet121) on the **same, consistent preprocessing and split strategy**.

The goal is to compare model families under identical conditions and establish a clean, reproducible baseline for future improvements (augmentation, fine‑tuning, explainability, etc.).

## Tech Stack

- **Language**: Python 3.9
- **Deep Learning**: TensorFlow / Keras
- **Data & Utils**: NumPy, scikit‑learn, OpenCV
- **Visualization**: matplotlib, seaborn
- **Environment**: Jupyter Notebook

## Dataset

- **Source**: [Skin Cancer MNIST: HAM10000 (Kaggle)](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- **Description**: The **HAM10000 ("Human Against Machine with 10000 training images")** dataset is a large collection of multi-source dermatoscopic images of common pigmented skin lesions.
- **Content**:
    - **Total images**: 10,015 RGB dermatoscopic images
    - **Image size**: 600×450600 \times 450600×450 pixels (original)
    - **Labels**: 7 distinct diagnostic categories, including both benign and malignant types

## Directory Structure

```
.
├── notebooks/
│   ├── CNN.ipynb             # Simple CNN baseline
│   ├── VGG16.ipynb           # Transfer learning (frozen backbone)
│   ├── ResNet50.ipynb        # Transfer learning (frozen backbone)
│   ├── InceptionV3.ipynb     # Transfer learning (frozen backbone)
│   └── DenseNet121.ipynb     # Transfer learning (frozen backbone)
│
├── datasets/
│   └── skin-cancer/
│       ├── BENIGN/...
│       └── MALIGNANT/...
│
├── models/                   # Generated at runtime (saved weights)
│   ├── best_CNN_model.h5
│   ├── best_vgg_model.h5
│   ├── best_resnet_model.h5
│   ├── best_inception_model.h5
│   └── best_densenet_model.h5
└── README.md
```

> The models/ directory is created by ModelCheckpoint during training.
> 

## How It Works

### 1) Data Loading & Preprocessing (shared across notebooks)

- **Input shape**: `128 × 128 × 3`
- **Pipeline (OpenCV)**:
    1. Read image (BGR)
    2. **Denoise**: `cv2.fastNlMeansDenoisingColored`
    3. Convert to **YUV**; **histogram equalization** on Y channel
    4. Convert back to **RGB**
    5. **Resize** to `(128, 128)`
    6. **Normalize** to `[0, 1]`
- **Split**: Stratified **80/10/10** (train/val/test)

### 2) Models

- **Simple CNN (baseline)**
    - `Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Conv2D(128) → Flatten → Dense(64) → Dense(1, sigmoid)`
    - **Optimizer**: `Adam` (default)
    - **Loss**: `binary_crossentropy`
    - **Metric**: `accuracy`
- **Transfer Learning (frozen feature extractor)**
    - Backbones: **VGG16**, **ResNet50**, **InceptionV3**, **DenseNet121**
    - Config: `include_top=False`, `weights='imagenet'`, `input_shape=(128,128,3)`
    - Head: `GlobalAveragePooling2D → Dense(1024, relu) → Dense(1, sigmoid)`
    - **Optimizer**: `Adam(learning_rate=1e-4)`
    - **Loss**: `binary_crossentropy`
    - **Metric**: `accuracy`

### 3) Training Routine (all notebooks)

- **Epochs**: `50`, **Batch size**: `32`
- **Callbacks**:
    - `EarlyStopping(patience=10, restore_best_weights=True)` (monitor `val_loss`)
    - `ModelCheckpoint(..., save_best_only=True)`
        - For `CNN.ipynb`: monitor `val_accuracy`
        - Weight files saved as `best_*.h5`
- After training, the **best weights** are loaded and **test accuracy** is printed.

## Results (Test Accuracy)

Best observed test accuracies across runs captured in each notebook:

| Model | Best Test Acc. | Runs Observed |
| --- | --- | --- |
| **DenseNet121** | **0.8550** | 3 |
| **InceptionV3** | 0.8300 | 3 |
| **VGG16** | 0.8200 | 3 |
| **CNN (baseline)** | 0.8100 | 3 |
| **ResNet50** | 0.7300 | 3 |

> Notes
> 
> 
> • All models use **frozen** backbones (no fine‑tuning).
> 
> • No explicit **data augmentation** is applied in the current notebooks.
> 
> • Each notebook reports `Test Accuracy: ...` after re‑loading the best checkpoint.
> 

## How to Run Locally

1. **Prepare environment**
    
    ```bash
    pip install tensorflow opencv-python numpy scikit-learn matplotlib seaborn
    ```
    
2. **Place data**
    
    ```
    ../datasets/skin-cancer/
    ├── BENIGN/      # images
    └── MALIGNANT/   # images
    ```
    
3. **Launch notebooks**
    
    ```bash
    jupyter notebook
    ```
    
    Open any notebook in `notebooks/` (e.g., `DenseNet121.ipynb`) and run all cells.
    

> Tip: If your data lives elsewhere, update base_dir inside the notebook(s).
> 

## Features / Main Logic

- **Consistent preprocessing** for fair model comparison (denoise → Y‑channel equalization → RGB → resize → normalize)
- **Stratified 80/10/10 splits** to preserve class balance across train/val/test
- **Transfer‑learning baselines** with frozen ImageNet backbones
- **Reproducible checkpoints** via `ModelCheckpoint` and `EarlyStopping`
- **Simple, comparable metric** (`accuracy`) for quick benchmarking

## Future Work

- Add **data augmentation** (flips, rotations, color jitter) to improve robustness
- **Unfreeze & fine‑tune** top layers of the backbone(s)
- Track additional metrics (**precision/recall/F1**, **ROC‑AUC**), plus **confusion matrices**
- Integrate **Grad‑CAM** or similar methods for interpretability
- Package an **inference API** (FastAPI/Flask) and a lightweight **UI**
- Replace fixed “first 1,000 files” sampling with **shuffled sampling** to avoid ordering bias

## Motivation / Impact

- Establishes a **clean baseline** for skin‑lesion binary classification across popular architectures
- Creates a **repeatable training/evaluation loop** with saved best weights
- Provides a solid foundation for **future clinical‑research‑oriented improvements** (augmentation, fine‑tuning, explainability)