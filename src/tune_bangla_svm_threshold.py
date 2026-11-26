
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "dataset/splits/bangla_train.csv"
MODEL_DIR = BASE_DIR / "models/bangla_optimized"
RESULT_DIR = BASE_DIR / "results/bangla_analysis"

RESULT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading training dataset...")

df = pd.read_csv(DATA_FILE, encoding="utf-8-sig")

X = df["clean_text"].fillna("").astype(str)
y = df["label"].astype(str).str.strip()

# Validation data must remain unseen during training
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))

print("\nCreating TF-IDF features...")

word_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    max_features=30000,
    min_df=2,
    sublinear_tf=True
)

char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 5),
    max_features=30000,
    min_df=2,
    sublinear_tf=True
)

word_train = word_vectorizer.fit_transform(X_train)
word_val = word_vectorizer.transform(X_val)

char_train = char_vectorizer.fit_transform(X_train)
char_val = char_vectorizer.transform(X_val)

X_train_features = hstack(
    [word_train, char_train]
).tocsr()

X_val_features = hstack(
    [word_val, char_val]
).tocsr()

print("Training SVM...")

# Inspect saved model configuration for C value.
saved_model = joblib.load(
    MODEL_DIR / "bangla_optimized_svm.joblib"
)

model = LinearSVC(
    C=saved_model.C,
    class_weight="balanced",
    random_state=42,
    max_iter=5000
)

model.fit(X_train_features, y_train)

print("Evaluating validation scores...")

scores = model.decision_function(X_val_features)

# LinearSVC positive scores correspond to classes_[1]
if model.classes_[1] != "Fake":
    scores = -scores

y_true = (y_val == "Fake").astype(int).to_numpy()

results = []

for threshold in np.arange(-1.0, 1.001, 0.02):
    predictions = (scores >= threshold).astype(int)

    results.append({
        "threshold": round(float(threshold), 3),
        "accuracy": accuracy_score(y_true, predictions),
        "fake_precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "fake_recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "fake_f1": f1_score(
            y_true, predictions, zero_division=0
        )
    })

results_df = pd.DataFrame(results)

results_df.to_csv(
    RESULT_DIR / "svm_threshold_comparison.csv",
    index=False,
    encoding="utf-8-sig"
)

# Prioritize Fake F1 while maintaining precision >= 0.60
eligible = results_df[
    results_df["fake_precision"] >= 0.60
]

if eligible.empty:
    best = results_df.sort_values(
        ["fake_f1", "fake_recall"],
        ascending=False
    ).iloc[0]
    print("\nWarning: No threshold achieved 60% precision.")
else:
    best = eligible.sort_values(
        ["fake_f1", "fake_recall"],
        ascending=False
    ).iloc[0]

best_threshold = float(best["threshold"])

final_predictions = (
    scores >= best_threshold
).astype(int)

print("\n" + "=" * 55)
print("LEAKAGE-FREE VALIDATION RESULTS")
print("=" * 55)

print(best.to_string())

print("\nConfusion Matrix [Fake, Real]:")
print(
    confusion_matrix(
        y_true,
        final_predictions,
        labels=[1, 0]
    )
)

threshold_data = {
    "model": "Bangla Optimized SVM",
    "threshold": best_threshold,
    "validation_fake_precision": float(best["fake_precision"]),
    "validation_fake_recall": float(best["fake_recall"]),
    "validation_fake_f1": float(best["fake_f1"]),
    "positive_class": "Fake",
    "selection_method": "held_out_validation",
    "precision_floor": 0.60
}

with open(
    RESULT_DIR / "svm_threshold.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        threshold_data,
        file,
        indent=4,
        ensure_ascii=False
    )

print("\nThreshold saved successfully.")
print("Existing production SVM model was not overwritten.")
