
import json
from pathlib import Path

import joblib
import pandas as pd
from scipy.sparse import hstack

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models" / "bangla_optimized"
RESULT_DIR = BASE_DIR / "results" / "bangla_analysis"
TEST_FILE = BASE_DIR / "dataset" / "splits" / "bangla_test.csv"

print("Loading saved Bangla SVM model...")

model = joblib.load(
    MODEL_DIR / "bangla_optimized_svm.joblib"
)

word_vectorizer = joblib.load(
    MODEL_DIR / "word_vectorizer.joblib"
)

char_vectorizer = joblib.load(
    MODEL_DIR / "char_vectorizer.joblib"
)

with open(
    RESULT_DIR / "svm_threshold.json",
    "r",
    encoding="utf-8"
) as file:
    threshold_data = json.load(file)

threshold = float(threshold_data["threshold"])

print("Selected threshold:", threshold)

print("Loading final test dataset...")

df = pd.read_csv(TEST_FILE, encoding="utf-8-sig")

X_test = df["clean_text"].fillna("").astype(str)
y_test = df["label"].astype(str).str.strip()

word_features = word_vectorizer.transform(X_test)
char_features = char_vectorizer.transform(X_test)

X_features = hstack(
    [word_features, char_features]
).tocsr()

if X_features.shape[1] != model.n_features_in_:
    raise ValueError("SVM feature dimension mismatch.")

print("Generating predictions...")

scores = model.decision_function(X_features)

# Ensure higher scores always mean more likely Fake
if model.classes_[1] != "Fake":
    scores = -scores

predictions = [
    "Fake" if score >= threshold else "Real"
    for score in scores
]

print("\n" + "=" * 55)
print("TUNED BANGLA SVM - FINAL TEST RESULTS")
print("=" * 55)

print("Threshold:", threshold)

print("Accuracy:", round(
    accuracy_score(y_test, predictions), 4
))

print("Fake Precision:", round(
    precision_score(
        y_test, predictions,
        pos_label="Fake",
        zero_division=0
    ), 4
))

print("Fake Recall:", round(
    recall_score(
        y_test, predictions,
        pos_label="Fake",
        zero_division=0
    ), 4
))

print("Fake F1:", round(
    f1_score(
        y_test, predictions,
        pos_label="Fake",
        zero_division=0
    ), 4
))

print("\nConfusion Matrix [Fake, Real]:")
print(
    confusion_matrix(
        y_test,
        predictions,
        labels=["Fake", "Real"]
    )
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=["Fake", "Real"],
        zero_division=0
    )
)

output = df.copy()
output["predicted_label"] = predictions
output["svm_fake_score"] = scores

output.to_csv(
    RESULT_DIR / "tuned_svm_test_predictions.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nPredictions saved successfully.")
