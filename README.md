# Pneumonia Detection from Chest X-Rays

A deep learning project that uses Convolutional Neural Networks (CNN) to detect pneumonia from chest X-ray images.

## 📋 Project Overview

This project implements a CNN-based image classifier to distinguish between normal chest X-rays and those showing signs of pneumonia. The model achieves 90% accuracy on the test set.

## 🎯 Results

- **Validation Accuracy**: 97.51%
- **Test Accuracy**: 90.06%
- **Precision**: 90.06%
- **Recall**: 90.06%
- **AUC**: 96.33%

## 🏗️ Model Architecture

- **Input**: 256×256 grayscale images
- **Convolutional Blocks**: 3 blocks with increasing filters (32 → 64 → 128)
- **Pooling**: MaxPooling after each block
- **Regularization**: Dropout (0.25 and 0.5)
- **Dense Layers**: 256 → 128 neurons
- **Output**: 2 classes (NORMAL/PNEUMONIA) with softmax
- **Total Parameters**: 33,874,274

## 📊 Dataset

- **Source**: Chest X-Ray Images (Pneumonia)
- **Total Images**: ~5,856
- **Classes**: NORMAL (1,584), PNEUMONIA (4,272)
- **Split**: 
  - Training: 4,187 images
  - Validation: 1,045 images
  - Test: 624 images

**Note**: Dataset not included in repository due to size. Download from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8+
TensorFlow 2.19.0
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Mokshaa7/ai-powered-medical-image-analysis.git
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset and organize as:

data/
└── chest_xray/
    ├── train/                # Data used for training the model
    │   ├── NORMAL/           # Images of healthy lungs
    │   └── PNEUMONIA/        # Images of lungs with pneumonia
    ├── val/                  # Data used for validation during training
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    └── test/                 # Data used for final model evaluation
        ├── NORMAL/
        └── PNEUMONIA/


### Training

Using Jupyter Notebook** (Recommended for exploration)
```bash
jupyter notebook pneumonia_detection_training.ipynb
```

### Configuration

Edit `config.py` to modify:
- Image size
- Batch size
- Learning rate
- Augmentation parameters
- Number of epochs

## 📁 Project Structure
AIPoweredMedicalImageAnalysis/
├── README.md
├── requirements.txt
├── .gitignore
├── pneumonia_detection_training.ipynb  # Training notebook

## 🔧 Training Details

- **Platform**: Google Colab (T4 GPU)
- **Training Time**: ~30-60 minutes
- **Framework**: TensorFlow/Keras
- **Optimizer**: Adam (lr=0.001)
- **Loss**: Categorical Crossentropy
- **Callbacks**: Early Stopping, ReduceLROnPlateau, ModelCheckpoint

## 📈 Data Augmentation

- Rotation: ±15 degrees
- Width/Height Shift: ±10%
- Zoom: ±10%
- Horizontal Flip: Enabled
- Brightness: 80%-120%


## ⚠️ Disclaimer

This model is for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis. Always consult qualified healthcare professionals for medical advice.


## 👤 Author

[Moksha Shah]
- GitHub: [@Mokshaa7](https://github.com/Mokshaa7)
- LinkedIn: [Moksha Shah](www.linkedin.com/in/moksha-shah-558518343)

## 🙏 Acknowledgments

- Dataset: Kaggle
- TensorFlow/Keras team
- Google Colab for free GPU access
