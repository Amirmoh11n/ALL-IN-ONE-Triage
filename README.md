# Triage Level Classification with PyTorch

A lightweight deep learning project for **multi-class emergency department triage level classification** using structured/tabular patient data and **pure PyTorch**.

The project covers the complete workflow from **dataset preparation and preprocessing** to model training and evaluation, with a focus on understanding how neural networks can be applied to tabular clinical data.

> **Best Validation Accuracy: 82.50%**

## Overview

Emergency department triage assigns patients to different urgency levels based on their clinical condition and available information.

This project trains a fully connected neural network to classify patients into **five triage levels (1–5)** using demographic, arrival, clinical, and vital-sign features.

The implementation is intentionally lightweight and relies primarily on:

* **PyTorch** — model development and training
* **Pandas** — data processing
* **NumPy** — numerical operations

The project is intended as an **educational machine learning experiment**, not as a clinical decision-support system.

---

## Dataset

The dataset contains approximately **200 patient records**.

### Input Features

| Category     | Features                                                                                 |
| ------------ | ---------------------------------------------------------------------------------------- |
| Demographics | `age`, `sex`                                                                             |
| Arrival      | `arrival_mode`                                                                           |
| Clinical     | `pain_score`, `mental_status`, `complaint_category`, `gcs_total`                         |
| Vital Signs  | `systolic_bp`, `diastolic_bp`, `heart_rate`, `respiratory_rate`, `temperature_c`, `spo2` |

### Target

`target_level`

| Level | Meaning           |
| ----- | ----------------- |
| 1     | Highest urgency   |
| 2     | Very high urgency |
| 3     | Moderate urgency  |
| 4     | Lower urgency     |
| 5     | Lowest urgency    |

The dataset is relatively small and contains class imbalance, particularly among the less frequent triage levels.

---

## Machine Learning Pipeline

The project follows a simple end-to-end tabular classification pipeline:

```text
Raw Dataset
     │
     ▼
Data Loading
     │
     ▼
Preprocessing
(Label Encoding + Standardization)
     │
     ▼
Train / Validation Split
        80 / 20
     │
     ▼
PyTorch Tensor Conversion
     │
     ▼
Neural Network
     │
     ▼
Weighted Cross-Entropy Loss
     │
     ▼
Training
     │
     ▼
Validation
     │
     ▼
Accuracy + Confusion Matrix
```

### Preprocessing

The dataset is prepared before training using:

* Categorical feature encoding
* Numerical feature standardization
* Train/validation splitting
* Conversion to PyTorch tensors
* Class-weight calculation to reduce the effect of class imbalance

---

## Model Architecture

The classifier is implemented as a fully connected neural network:

```text
Input (13 features)
        │
      Linear
    13 → 512
        │
      ReLU
        │
     Dropout
        │
      Linear
    512 → 256
        │
      ReLU
        │
     Dropout
        │
      Linear
    256 → 128
        │
      ReLU
        │
     Dropout
        │
      Linear
     128 → 64
        │
      ReLU
        │
     Dropout
        │
      Linear
      64 → 32
        │
      ReLU
        │
     Dropout
        │
      Linear
       32 → 5
        │
        ▼
  Triage Level (1–5)
```

### Training Configuration

| Component              | Configuration                  |
| ---------------------- | ------------------------------ |
| Framework              | PyTorch                        |
| Architecture           | Fully Connected Neural Network |
| Output Classes         | 5                              |
| Loss                   | `CrossEntropyLoss`             |
| Class Imbalance        | Weighted loss                  |
| Optimizer              | Adam                           |
| Learning Rate          | `0.001`                        |
| Weight Decay           | `1e-4`                         |
| Dropout                | `0.5`                          |
| Epochs                 | 60                             |
| Train/Validation Split | 80/20                          |

---

## Results

### Validation Performance

| Metric                   |                      Result |
| ------------------------ | --------------------------: |
| Best Validation Accuracy |                  **82.50%** |
| Training Samples         |                        ~160 |
| Validation Samples       |                         ~40 |
| Baseline Accuracy        |                      77.50% |
| Improvement              | **+5.00 percentage points** |

The use of **class-weighted loss** improved the model's handling of minority classes compared with the baseline configuration.

### Observations

* The most frequent class, **Triage Level 3**, is classified relatively well.
* Rare classes, particularly **Levels 1 and 5**, remain more difficult to predict.
* The small dataset size limits the reliability and generalization of the results.
* Accuracy alone is not sufficient for evaluating an imbalanced multi-class classification problem; the confusion matrix provides additional insight into class-specific behavior.

---

## Project Structure

```text
ALL-IN-ONE-Triage/
│
├── ALL_IN_ONE_Tab_V0/
│   ├── notebook.ipynb
│   └── teriage_dataset.csv
│
├── LICENSE
└── README.md
```

---

## Requirements

* Python 3.x
* PyTorch
* Pandas
* NumPy
* Jupyter Notebook

Install the required packages with:

```bash
pip install torch pandas numpy jupyter
```

---

## Getting Started

Clone the repository:

```bash
git clone https://github.com/Amirmoh11n/ALL-IN-ONE-Triage
cd ALL-IN-ONE-Triage
```

Launch the notebook:

```bash
jupyter notebook
```

Then open:

```text
ALL_IN_ONE_Tab_V0/notebook.ipynb
```

The notebook contains the dataset preparation, preprocessing, model definition, training, and evaluation workflow.

---
## Author

**AmirMohammad Nashalji**

Machine / Deep Learning Project
