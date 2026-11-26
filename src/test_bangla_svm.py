
import pandas as pd
import joblib

from pathlib import Path
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
TEST_FILE = BASE_DIR / "dataset" / "splits" / "bangla_test.csv"

print("Loading Bangla SVM model...")

model = joblib.load(
    MODEL_DIR / "bangla_optimized_svm.joblib"
)

word_vectorizer = joblib.load(
    MODEL_DIR / "word_vectorizer.joblib"
)

char_vectorizer = joblib.load(
    MODEL_DIR / "char_vectorizer.joblib"
)

print("Loading test dataset...")

df = pd.read_csv(TEST_FILE, encoding="utf-8-sig")

X_test = df["clean_text"].fillna("").astype(str)
y_test = df["label"]

print("Preparing TF-IDF features...")

word_features = word_vectorizer.transform(X_test)
char_features = char_vectorizer.transform(X_test)

combined_features = hstack(
    [word_features, char_features]
).tocsr()

# Verify that the saved SVM expects combined features.
if combined_features.shape[1] != model.n_features_in_:
    raise ValueError(
        "Feature mismatch: saved SVM may use word-only "
        "or character-only features. "
        "Check validation_comparison.csv before proceeding."
    )

print("Predicting Bangla news...")

predictions = model.predict(combined_features)

print("\n" + "=" * 55)
print("BANGLA OPTIMIZED SVM TEST RESULTS")
print("=" * 55)

print("Accuracy:", round(
    accuracy_score(y_test, predictions), 4
))

print("Fake Precision:", round(
    precision_score(
        y_test, predictions,
        pos_label="Fake", zero_division=0
    ), 4
))

print("Fake Recall:", round(
    recall_score(
        y_test, predictions,
        pos_label="Fake", zero_division=0
    ), 4
))

print("Fake F1:", round(
    f1_score(
        y_test, predictions,
        pos_label="Fake", zero_division=0
    ), 4
))

print("\nConfusion Matrix [Fake, Real]:")
print(confusion_matrix(
    y_test, predictions,
    labels=["Fake", "Real"]
))

print("\nClassification Report:")
print(classification_report(
    y_test, predictions,
    labels=["Fake", "Real"],
    zero_division=0
))





# ======================================
# SAVE MISCLASSIFIED FAKE NEWS
# ======================================

results_dir = BASE_DIR / "results" / "bangla_analysis"
results_dir.mkdir(parents=True, exist_ok=True)

analysis_df = df.copy()
analysis_df["predicted_label"] = predictions

false_negatives = analysis_df[
    (analysis_df["label"] == "Fake") &
    (analysis_df["predicted_label"] == "Real")
].copy()

output_path = results_dir / "svm_false_negatives.csv"

false_negatives.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)

print("\nFalse Negatives:", len(false_negatives))
print("Saved to:", output_path)
