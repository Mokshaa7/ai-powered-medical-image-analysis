# 🧠 AI-Powered Pneumonia Detection System

An end-to-end deep learning system that detects pneumonia from chest X-ray images using a custom Convolutional Neural Network (CNN).
The project integrates model development, uncertainty-aware predictions, and a user-facing web interface for real-world usability.

---

## 🚀 Overview

This project focuses on building a **medical imaging classification system** that prioritizes **high sensitivity** while handling uncertainty in predictions.

It goes beyond basic classification by introducing:

* Confidence-based decision thresholds
* An **“Inconclusive” prediction class**
* A full-stack deployment (Flask + Gradio)

---

## 🧩 Features

* 🧠 Custom CNN model trained on chest X-ray images
* ⚠️ Uncertainty-aware prediction system (reduces overconfident errors)
* 📊 Probability breakdown for each prediction
* 🌐 Web interface for real-time inference
* 🤖 Gradio demo deployed on Hugging Face
* 🐳 Dockerized backend (for production-ready deployment)

---

## 🏗️ Model Architecture

The model is a **custom CNN** with three convolutional blocks:

* Conv2D → Conv2D → MaxPooling → Dropout (×3)
* Flatten → Dense(256) → Dense(128) → Softmax (2 classes)

**Input:** 256 × 256 grayscale images
**Output:** Probability distribution over:

* NORMAL
* PNEUMONIA

---

## 📊 Performance Metrics

| Metric                           | Value |
| -------------------------------- | ----- |
| Accuracy                         | 88.9% |
| AUC                              | 0.96  |
| Sensitivity (Recall - Pneumonia) | 98%   |
| Specificity (Recall - Normal)    | 73%   |

### 🔍 Key Insight

* The model is **highly sensitive** → rarely misses pneumonia
* Trade-off: higher false positives for normal cases

---

## ⚠️ Uncertainty Handling (Core Contribution)

Instead of forcing every prediction into a binary class, the system introduces a third outcome:

| Pneumonia Probability | Output       |
| --------------------- | ------------ |
| ≥ 85%                 | PNEUMONIA    |
| ≤ 40%                 | NORMAL       |
| 40% – 85%             | INCONCLUSIVE |

### 🎯 Why this matters

* Prevents overconfident misclassification
* Reflects real-world medical ambiguity
* Improves trustworthiness of AI predictions

---

## 🌐 Application Interfaces

### 🔹 Flask Web App

* Image upload interface
* Prediction + confidence score
* Risk indicator and warnings
* Probability breakdown

### 🔹 Gradio Demo (Live)

* Simplified UI for quick testing
* Public deployment via Hugging Face

---

## 🐳 Deployment

### ✔️ Hugging Face (Completed)

* Gradio-based deployment
* Public and shareable demo

### ⚙️ Docker (Prepared)

* Containerized backend using Flask
* Gunicorn-based production server
* Compatible with cloud platforms (Render, Railway)

> Note: Deployment on some platforms may require environment-specific adjustments due to TensorFlow compatibility constraints.

---

## 📁 Project Structure

```
root/
├── app.py
├── best_pneumonia_cnn.keras
├── requirements.txt
├── Dockerfile
├── templates/
├── static/
├── training/
```

---

## ⚙️ Installation & Setup

### 1. Clone Repository

```bash
git clone https://github.com/your-username/pneumonia-detection.git
cd pneumonia-detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Locally

```bash
python app.py
```

---

## 🧠 Key Learnings

* Confidence ≠ Accuracy in deep learning models
* Neural networks can be overconfident on ambiguous inputs
* Importance of **model calibration and uncertainty handling**
* Real-world trade-offs between sensitivity and specificity
* Challenges in deploying ML models across environments

##

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**.
It is **not intended for clinical or diagnostic use**.

---

## 👤 Author

**Moksha Shah**
Computer Science Engineering Student

link: https://huggingface.co/spaces/mokiiii/ai_powered_pneumonia_detection
---

## ⭐ Acknowledgements

* Open-source medical imaging datasets
* TensorFlow & Keras community
* Hugging Face Spaces for deployment support

---

## 📌 Summary

This project demonstrates:

* End-to-end ML system design
* Real-world model evaluation trade-offs
* Deployment and UI integration
* Responsible AI practices through uncertainty handling

---
