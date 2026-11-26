# 📰 Bilingual Fake News Detection System (বাংলা ও English)

A state-of-the-art bilingual Fake News Detection web application using **BanglaBERT (`csebuetnlp/banglabert`)**, **Optimized SVM**, and **BiLSTM** models built with Flask.

---

## 🌟 Key Features

- **Bilingual Support:** High-accuracy classification for both **Bangla (বাংলা)** and **English** news articles.
- **State-of-the-Art BanglaBERT:** Powered by fine-tuned ELECTRA-based `BanglaBERT` achieving **75.15% Test Accuracy** and **82.42% Precision** on held-out Bangla news articles.
- **Optimized SVM Fallback:** High-speed Classical ML pipeline with Unicode-aware tokenization and character n-gram extraction.
- **Modern Responsive Web UI:** Real-time prediction with probability confidence scores, detailed decision thresholds, and history logging.
- **Model Benchmark & Performance Hub:** Interactive model comparison dashboard evaluating held-out test benchmarks.

---

## 📊 Model Performance on Held-Out Test Sets

| Language | Model Architecture | Accuracy | Fake Precision | Fake Recall | Fake F1 | Threshold | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Bangla** | **BanglaBERT (ELECTRA Transformer)** | **75.15%** | **82.42%** | 39.29% | 0.532 | 0.50 | **Active Serving Engine** |
| **Bangla** | **Optimized SVM (Unicode-aware)** | **73.74%** | **65.80%** | 56.24% | 0.606 | 0.00 | Fallback ML Engine |
| **English** | **BiLSTM (Deep Learning)** | **98.03%** | **97.54%** | 98.34% | 0.979 | 0.43 | Active DL Engine |
| **English** | **Logistic Regression** | **97.47%** | **98.31%** | 96.33% | 0.973 | 0.50 | Fallback Engine |

---

## 🛠️ Project Structure

```text
├── app.py                      # Flask Application with Bilingual Routing & Prediction
├── requirements.txt            # Python Dependencies
├── dataset/
│   ├── ben_fnd_16k.csv         # Master Dataset
│   ├── cleaned/                # Preprocessed Datasets (Bangla & English)
│   └── splits/                 # Train / Test Data Splits
├── models/
│   ├── bangla_bert/            # Fine-tuned BanglaBERT Model Config & Weights
│   ├── bangla_optimized/       # Improved Bangla SVM & TF-IDF Vectorizers
│   └── deep_learning/          # English BiLSTM & LSTM Keras Models
├── notebooks/                  # Colab Training Notebooks
├── results/                    # Confusion Matrices, Classification Reports & Benchmarks
├── static/                     # CSS Stylesheets & Assets
├── templates/                  # Jinja2 HTML Templates (index, performance, history)
└── src/                        # Data Cleaning, EDA, Model Training & Evaluation Scripts
```

---

## 🚀 Quickstart Guide

### 1. Clone the Repository
```bash
git clone https://github.com/Nirob-2002/Fake_News_Detector.git
cd Fake_News_Detector
```

### 2. Create and Activate Virtual Environment
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Flask Web App
```bash
python app.py
```
Open your browser and navigate to: `http://127.0.0.1:5000`

---

## 👥 Authors
- **Nirob Dewan Limon** ([@Nirob-2002](https://github.com/Nirob-2002))
