# 📰 Bilingual Fake News Detection System (বাংলা ও English)

A bilingual fake news detection application with a **Streamlit** interface and an optional Flask web interface. It uses **BanglaBERT (`csebuetnlp/banglabert`)**, an **Optimized SVM**, and **BiLSTM** models when their weights and dependencies are available.

---

## 🌟 Key Features

- **Bilingual Support:** High-accuracy classification for both **Bangla (বাংলা)** and **English** news articles.
- **BanglaBERT with SVM fallback:** Uses the fine-tuned transformer when its model weights are installed; otherwise, it falls back to the optimized SVM.
- **Optimized SVM Fallback:** High-speed Classical ML pipeline with Unicode-aware tokenization and character n-gram extraction.
- **Modern Responsive Web UI:** Real-time prediction with probability confidence scores, detailed decision thresholds, and history logging.
- **Model Benchmark & Performance Hub:** Interactive model comparison dashboard evaluating held-out test benchmarks.

---

## 📊 Model Performance on Held-Out Test Sets

| Language | Model Architecture | Accuracy | Fake Precision | Fake Recall | Fake F1 | Threshold | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Bangla** | **BanglaBERT (ELECTRA Transformer)** | **75.15%** | **82.42%** | 39.29% | 0.532 | 0.50 | Optional; weights not bundled |
| **Bangla** | **Optimized SVM (Unicode-aware)** | **73.74%** | **65.80%** | 56.24% | 0.606 | 0.00 | Active when BanglaBERT is unavailable |
| **English** | **BiLSTM (Deep Learning)** | **98.03%** | **97.54%** | 98.34% | 0.979 | 0.43 | Active when TensorFlow is available |
| **English** | **Logistic Regression** | **97.47%** | **98.31%** | 96.33% | 0.973 | 0.50 | Fallback when TensorFlow is unavailable |

---

## 🛠️ Project Structure

```text
├── app.py                      # Streamlit application entry point
├── flask_app.py                # Optional Flask application and shared prediction engine
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

### 4. Run the Streamlit App
```bash
streamlit run app.py
```
Streamlit opens the app at `http://localhost:8501`. Streamlit Community Cloud should use `app.py` as the app's main file.

To run the optional Flask interface instead:
```bash
python flask_app.py
```
Open `http://127.0.0.1:5000`.

The fine-tuned BanglaBERT weight file is larger than GitHub's 100 MiB file limit and is not included in this repository. Without that file, Bangla analysis uses the optimized SVM fallback. English analysis uses BiLSTM when TensorFlow is available and otherwise falls back to Logistic Regression.

---

## 👥 Authors
- **Nirob Dewan Limon** ([@Nirob-2002](https://github.com/Nirob-2002))
