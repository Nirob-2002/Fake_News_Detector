
import os
import re
import json
import math
import sqlite3
import unicodedata

from pathlib import Path
from datetime import datetime

import joblib
from flask import Flask, render_template, request, redirect, url_for, flash
from scipy.sparse import hstack

try:
    import tensorflow as tf
except ModuleNotFoundError as error:
    if error.name != "tensorflow":
        raise
    tf = None


# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "models" / "deep_learning"
RESULT_DIR = BASE_DIR / "results" / "deep_learning"
BANGLA_SVM_DIR = BASE_DIR / "models" / "bangla_optimized"
BANGLA_SVM_RESULT_DIR = BASE_DIR / "results" / "bangla_analysis"
BANGLA_BERT_DIR = BASE_DIR / "models" / "bangla_bert"
BANGLA_BERT_RESULT_DIR = BASE_DIR / "results" / "bangla_bert"

IS_CLOUD_RUN = bool(os.environ.get("K_SERVICE"))
HISTORY_ENABLED = os.environ.get(
    "ENABLE_HISTORY",
    "true"
).strip().lower() in {"1", "true", "yes", "on"}
DATABASE = Path(os.environ.get(
    "DATABASE_PATH",
    "/tmp/news-lens-history.db" if IS_CLOUD_RUN else str(BASE_DIR / "database.db")
))

app = Flask(__name__)
secret_key = os.environ.get("FLASK_SECRET_KEY")
if IS_CLOUD_RUN and not secret_key:
    raise RuntimeError(
        "Set FLASK_SECRET_KEY with Google Secret Manager before deploying."
    )
app.secret_key = secret_key or "fake-news-detection-development-key"

# ==========================================
# 2. TEXT PREPROCESSING
# Same function used during model training
# ==========================================

def clean_text(text, language):

    if not text:
        return ""

    text = str(text)

    text = unicodedata.normalize("NFC", text)

    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    text = re.sub(
        r"<[^>]+>",
        " ",
        text
    )

    # Preserve Bangla ZWNJ and ZWJ
    text = re.sub(
        r"[\u200b\ufeff]",
        "",
        text
    )

    if language == "English":
        text = text.lower()

    text = re.sub(
        r"[^\w\s\u0980-\u09FF]",
        " ",
        text,
        flags=re.UNICODE
    )

    text = text.replace("_", " ")

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ==========================================
# 3. LOAD TRAINED MODELS
# ==========================================

print("Loading trained models...")

models = {}
thresholds = {}

if tf is not None:
    for language in ["english", "bangla"]:

        model_path = (
            MODEL_DIR /
            f"{language}_bilstm.keras"
        )

        threshold_path = (
            RESULT_DIR /
            f"{language}_bilstm_threshold.json"
        )

        if not model_path.exists():
            raise FileNotFoundError(model_path)

        if not threshold_path.exists():
            raise FileNotFoundError(threshold_path)

        models[language] = tf.keras.models.load_model(
            model_path,
            compile=False
        )

        with open(
            threshold_path,
            "r",
            encoding="utf-8"
        ) as file:

            threshold_data = json.load(file)

        thresholds[language] = float(
            threshold_data["threshold"]
        )

        print(
            language.upper(),
            "model loaded!",
            "Threshold:",
            thresholds[language]
        )

    print("All models loaded successfully!")
else:
    print(
        "TensorFlow is unavailable; English predictions will use "
        "the Logistic Regression model."
    )

print("Loading tuned Bangla SVM...")

bangla_svm = joblib.load(
    BANGLA_SVM_DIR / "bangla_optimized_svm.joblib"
)

bangla_word_vectorizer = joblib.load(
    BANGLA_SVM_DIR / "word_vectorizer.joblib"
)

bangla_char_vectorizer = joblib.load(
    BANGLA_SVM_DIR / "char_vectorizer.joblib"
)

with open(
    BANGLA_SVM_RESULT_DIR / "svm_threshold.json",
    "r",
    encoding="utf-8"
) as file:
    bangla_svm_threshold = float(json.load(file)["threshold"])

if set(bangla_svm.classes_) != {"Fake", "Real"}:
    raise ValueError(
        "Bangla SVM must be trained on exactly the Fake and Real classes."
    )

expected_features = (
    len(bangla_word_vectorizer.vocabulary_)
    + len(bangla_char_vectorizer.vocabulary_)
)

if bangla_svm.n_features_in_ != expected_features:
    raise ValueError(
        "Bangla SVM feature dimensions do not match its saved vectorizers."
    )

if not math.isfinite(bangla_svm_threshold):
    raise ValueError("Bangla SVM threshold must be a finite number.")

print(
    "Bangla tuned SVM loaded!",
    "Threshold:",
    bangla_svm_threshold
)

english_model = None
english_vectorizer = None

if tf is None:
    english_model = joblib.load(
        BASE_DIR / "models" / "english_logisticregression.joblib"
    )
    english_vectorizer = joblib.load(BASE_DIR / "models" / "english_tfidf.joblib")

    if set(english_model.classes_) != {"Fake", "Real"}:
        raise ValueError(
            "English Logistic Regression must be trained on exactly "
            "the Fake and Real classes."
        )

    if english_model.n_features_in_ != len(english_vectorizer.vocabulary_):
        raise ValueError(
            "English Logistic Regression feature dimensions do not match "
            "its saved vectorizer."
        )

    print("English Logistic Regression fallback loaded!")

# Optional BanglaBERT Transformer Model
bangla_bert_model = None
bangla_bert_tokenizer = None

if (BANGLA_BERT_DIR / "config.json").exists():
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        print("Loading fine-tuned BanglaBERT from", BANGLA_BERT_DIR)
        bangla_bert_tokenizer = AutoTokenizer.from_pretrained(str(BANGLA_BERT_DIR))
        bangla_bert_model = AutoModelForSequenceClassification.from_pretrained(str(BANGLA_BERT_DIR))
        bangla_bert_model.eval()
        print("BanglaBERT loaded successfully!")
    except Exception as exc:
        print(f"Notice: Could not load BanglaBERT ({exc}). Using Bangla SVM as active engine.")

bangla_bert_threshold = 0.35
if (BANGLA_BERT_RESULT_DIR / "banglabert_metrics.json").exists():
    try:
        with open(BANGLA_BERT_RESULT_DIR / "banglabert_metrics.json", "r", encoding="utf-8") as f:
            bangla_bert_threshold = float(json.load(f).get("threshold", 0.35))
    except Exception:
        bangla_bert_threshold = 0.35


# ==========================================
# 4. DATABASE
# ==========================================

def get_db_connection():

    connection = sqlite3.connect(DATABASE)

    connection.row_factory = sqlite3.Row

    return connection


def initialize_database():

    connection = get_db_connection()

    connection.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            language TEXT NOT NULL,
            news_text TEXT NOT NULL,
            prediction TEXT NOT NULL,
            fake_probability REAL,
            confidence REAL,
            created_at TEXT NOT NULL,
            model_name TEXT NOT NULL DEFAULT 'BiLSTM',
            raw_score REAL,
            threshold REAL
        )
    """)

    columns = {
        row[1]: row
        for row in connection.execute("PRAGMA table_info(history)")
    }
    allowed_columns = {
        "id",
        "language",
        "news_text",
        "prediction",
        "fake_probability",
        "confidence",
        "created_at",
        "model_name",
        "raw_score",
        "threshold",
    }

    unexpected_columns = set(columns) - allowed_columns
    if unexpected_columns:
        connection.close()
        raise RuntimeError(
            "Cannot safely migrate detection history with unexpected "
            f"columns: {sorted(unexpected_columns)}"
        )

    required_columns = {
        "id",
        "language",
        "news_text",
        "prediction",
        "fake_probability",
        "confidence",
        "created_at",
        "model_name",
        "raw_score",
        "threshold",
    }
    needs_migration = (
        not required_columns.issubset(columns)
        or columns["fake_probability"][3] != 0
        or columns["confidence"][3] != 0
    )

    if needs_migration:
        model_name = (
            "COALESCE(model_name, 'BiLSTM')"
            if "model_name" in columns
            else "'BiLSTM'"
        )
        raw_score = "raw_score" if "raw_score" in columns else "NULL"
        threshold = "threshold" if "threshold" in columns else "NULL"

        connection.execute("ALTER TABLE history RENAME TO history_legacy")
        connection.execute("""
            CREATE TABLE history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                language TEXT NOT NULL,
                news_text TEXT NOT NULL,
                prediction TEXT NOT NULL,
                fake_probability REAL,
                confidence REAL,
                created_at TEXT NOT NULL,
                model_name TEXT NOT NULL DEFAULT 'BiLSTM',
                raw_score REAL,
                threshold REAL
            )
        """)
        connection.execute(f"""
            INSERT INTO history (
                id, language, news_text, prediction, fake_probability,
                confidence, created_at, model_name, raw_score, threshold
            )
            SELECT
                id, language, news_text, prediction, fake_probability,
                confidence, created_at, {model_name}, {raw_score}, {threshold}
            FROM history_legacy
        """)
        connection.execute("DROP TABLE history_legacy")

    connection.commit()

    connection.close()


if HISTORY_ENABLED:
    initialize_database()


# ==========================================
# 5. PREDICTION FUNCTION
# ==========================================

def predict_news(news_text, language, bangla_model="auto"):

    language_key = language.lower()

    processed_text = clean_text(
        news_text,
        language
    )

    if not processed_text:
        raise ValueError("News text is empty.")

    if language_key == "bangla":
        # Check if BanglaBERT Transformer is available
        if (
            bangla_bert_model is not None
            and bangla_bert_tokenizer is not None
            and bangla_model in ["auto", "bert", "banglabert"]
        ):
            import torch
            inputs = bangla_bert_tokenizer(
                processed_text,
                truncation=True,
                padding=True,
                max_length=256,
                return_tensors="pt"
            )
            with torch.no_grad():
                outputs = bangla_bert_model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1).numpy()[0]
                fake_prob = float(probs[1])
                pred = "Fake" if fake_prob >= bangla_bert_threshold else "Real"
                conf = fake_prob if pred == "Fake" else (1.0 - fake_prob)
                raw_score = float(outputs.logits[0][1] - outputs.logits[0][0])
                return {
                    "prediction": pred,
                    "model_name": "BanglaBERT",
                    "fake_probability": fake_prob,
                    "confidence": conf,
                    "threshold": bangla_bert_threshold,
                    "raw_score": raw_score,
                }

        # Optimized SVM Engine
        word_features = bangla_word_vectorizer.transform([processed_text])
        char_features = bangla_char_vectorizer.transform([processed_text])
        features = hstack([word_features, char_features]).tocsr()

        raw_score = float(
            bangla_svm.decision_function(features).reshape(-1)[0]
        )

        if bangla_svm.classes_[1] != "Fake":
            raw_score = -raw_score

        prediction = (
            "Fake" if raw_score >= bangla_svm_threshold else "Real"
        )

        # Calibrated probability via sigmoid of raw decision score
        try:
            fake_prob = 1.0 / (1.0 + math.exp(-raw_score))
        except OverflowError:
            fake_prob = 1.0 if raw_score > 0 else 0.0

        confidence = fake_prob if prediction == "Fake" else (1.0 - fake_prob)

        return {
            "prediction": prediction,
            "model_name": "SVM",
            "fake_probability": fake_prob,
            "confidence": confidence,
            "raw_score": raw_score,
            "threshold": bangla_svm_threshold,
        }

    if language_key == "english" and english_model is not None:
        features = english_vectorizer.transform([processed_text])
        probabilities = english_model.predict_proba(features)[0]
        fake_class_index = list(english_model.classes_).index("Fake")
        fake_probability = float(probabilities[fake_class_index])
        threshold = 0.5
        prediction = "Fake" if fake_probability >= threshold else "Real"
        confidence = (
            fake_probability
            if prediction == "Fake"
            else 1.0 - fake_probability
        )
        return {
            "prediction": prediction,
            "model_name": "Logistic Regression",
            "fake_probability": fake_probability,
            "confidence": confidence,
            "threshold": threshold,
        }

    input_tensor = tf.constant(
        [processed_text],
        dtype=tf.string
    )

    output = models[language_key](
        input_tensor,
        training=False
    )

    fake_probability = float(
        output.numpy().reshape(-1)[0]
    )

    threshold = thresholds[language_key]

    if fake_probability >= threshold:

        prediction = "Fake"

        confidence = fake_probability

    else:

        prediction = "Real"

        confidence = 1 - fake_probability

    return {
        "prediction": prediction,
        "model_name": "BiLSTM",
        "fake_probability": fake_probability,
        "confidence": confidence,
        "threshold": threshold
    }


# ==========================================
# 6. SAVE DETECTION HISTORY
# ==========================================

def save_history(
    language,
    news_text,
    prediction_result
):

    if not HISTORY_ENABLED:
        return

    connection = get_db_connection()

    connection.execute(
        """
        INSERT INTO history
        (
            language,
            news_text,
            prediction,
            fake_probability,
            confidence,
            model_name,
            raw_score,
            threshold,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,

        (
            language,
            news_text,
            prediction_result["prediction"],
            prediction_result.get("fake_probability"),
            prediction_result.get("confidence"),
            prediction_result.get("model_name", "BiLSTM"),
            prediction_result.get("raw_score"),
            prediction_result.get("threshold"),
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )
    )

    connection.commit()

    connection.close()


# ==========================================
# 7. HOME PAGE
# ==========================================

@app.route("/", methods=["GET", "POST"])
def index():

    result = None

    news_text = ""

    selected_language = "English"
    selected_bangla_model = "svm"

    if request.method == "POST":

        news_text = request.form.get(
            "news_text",
            ""
        ).strip()

        selected_language = request.form.get(
            "language",
            "English"
        )

        if selected_language not in ["English", "Bangla"]:

            flash("Invalid language selection.")

        elif not news_text:

            flash("Please enter news content.")

        elif len(news_text) > 30000:

            flash("News text must be under 30,000 characters.")

        else:

            try:

                result = predict_news(
                    news_text,
                    selected_language,
                    "auto"
                )

                save_history(
                    selected_language,
                    news_text,
                    result
                )

            except Exception as error:

                app.logger.exception(
                    "Prediction failed"
                )

                flash(
                    "Prediction failed. Please check the application logs."
                )

    return render_template(
        "index.html",
        result=result,
        news_text=news_text,
        selected_language=selected_language
    )


# ==========================================
# 8. DETECTION HISTORY
# ==========================================

@app.route("/history")
def history():

    connection = get_db_connection()

    records = connection.execute(
        """
        SELECT *
        FROM history
        ORDER BY id DESC
        LIMIT 100
        """
    ).fetchall()

    connection.close()

    summary = {
        "total": len(records),
        "fake": sum(record["prediction"] == "Fake" for record in records),
        "real": sum(record["prediction"] == "Real" for record in records),
    }

    return render_template(
        "history.html",
        records=records,
        summary=summary
    )


# ==========================================
# 9. MODEL PERFORMANCE
# ==========================================

@app.route("/performance")
def performance():
    return render_template(
        "performance.html",
        records=get_performance_records()
    )


def get_performance_records():
    comparison_file = (
        RESULT_DIR /
        "deep_learning_comparison.csv"
    )

    import pandas as pd

    records = []

    if comparison_file.exists():

        df = pd.read_csv(
            comparison_file
        )

        records = df.to_dict(
            orient="records"
        )

        for record in records:
            record["Evaluation_Set"] = "Held-out test"

    bert_metrics_file = BANGLA_BERT_RESULT_DIR / "banglabert_metrics.json"
    if bert_metrics_file.exists():
        with open(bert_metrics_file, "r", encoding="utf-8") as file:
            bert_metrics = json.load(file)

        records.append({
            "Language": "Bangla",
            "Model": "BanglaBERT (Active)" if bangla_bert_model is not None else "BanglaBERT",
            "Accuracy": float(bert_metrics["accuracy"]),
            "Precision_Fake": float(bert_metrics["fake_precision"]),
            "Recall_Fake": float(bert_metrics["fake_recall"]),
            "F1_Fake": float(bert_metrics["fake_f1"]),
            "Threshold": float(bert_metrics.get("threshold", 0.50)),
            "Evaluation_Set": "Held-out test",
            "Is_Active": bangla_bert_model is not None,
        })

    svm_predictions_file = (
        BANGLA_SVM_RESULT_DIR /
        "tuned_svm_test_predictions.csv"
    )

    if svm_predictions_file.exists():
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
        )

        svm_df = pd.read_csv(
            svm_predictions_file,
            encoding="utf-8-sig"
        )

        required_columns = {"label", "predicted_label"}
        if not required_columns.issubset(svm_df.columns):
            raise ValueError(
                "Bangla SVM test predictions are missing required labels."
            )

        with open(
            BANGLA_SVM_RESULT_DIR / "svm_threshold.json",
            "r",
            encoding="utf-8"
        ) as file:
            svm_threshold = float(json.load(file)["threshold"])

        records.append({
            "Language": "Bangla",
            "Model": "Optimized SVM (Fallback)" if bangla_bert_model is not None else "Optimized SVM (Active)",
            "Accuracy": accuracy_score(
                svm_df["label"],
                svm_df["predicted_label"]
            ),
            "Precision_Fake": precision_score(
                svm_df["label"],
                svm_df["predicted_label"],
                pos_label="Fake",
                zero_division=0
            ),
            "Recall_Fake": recall_score(
                svm_df["label"],
                svm_df["predicted_label"],
                pos_label="Fake",
                zero_division=0
            ),
            "F1_Fake": f1_score(
                svm_df["label"],
                svm_df["predicted_label"],
                pos_label="Fake",
                zero_division=0
            ),
            "Threshold": svm_threshold,
            "Evaluation_Set": "Held-out test",
            "Is_Active": bangla_bert_model is None,
        })

    return records


# ==========================================
# 10. RUN APPLICATION
# ==========================================

if __name__ == "__main__":

    print("\nStarting Fake News Detection System (Flask)...")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
