
import pandas as pd
import joblib

from pathlib import Path
from scipy.sparse import hstack
from sklearn.metrics import confusion_matrix

# Paths
DATA_DIR = Path("dataset/splits")
MODEL_DIR = Path("models/bangla_optimized")
RESULT_DIR = Path("results/error_analysis")

RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Load test dataset
df = pd.read_csv(
    DATA_DIR / "bangla_test.csv",
    encoding="utf-8-sig"
)

X = df["clean_text"].fillna("").astype(str)
y = df["label"]

# Load optimized model
model = joblib.load(
    MODEL_DIR / "bangla_optimized_svm.joblib"
)

# Load vectorizers
word_vectorizer = joblib.load(
    MODEL_DIR / "word_vectorizer.joblib"
)

char_vectorizer = joblib.load(
    MODEL_DIR / "char_vectorizer.joblib"
)

# Transform text
word_features = word_vectorizer.transform(X)
char_features = char_vectorizer.transform(X)

X_features = hstack(
    [word_features, char_features]
).tocsr()

# Predict
predictions = model.predict(X_features)

# Create result dataframe
df["predicted_label"] = predictions

# False Negative: Fake predicted as Real
false_negative = df[
    (df["label"] == "Fake") &
    (df["predicted_label"] == "Real")
].copy()

# False Positive: Real predicted as Fake
false_positive = df[
    (df["label"] == "Real") &
    (df["predicted_label"] == "Fake")
].copy()

# Correct predictions
correct = df[
    df["label"] == df["predicted_label"]
]

# Save error records
false_negative.to_csv(
    RESULT_DIR / "false_negative.csv",
    index=False,
    encoding="utf-8-sig"
)

false_positive.to_csv(
    RESULT_DIR / "false_positive.csv",
    index=False,
    encoding="utf-8-sig"
)

# Confusion matrix
cm = confusion_matrix(
    y,
    predictions,
    labels=["Fake", "Real"]
)

print("=" * 55)
print("BANGLA MODEL ERROR ANALYSIS")
print("=" * 55)

print("\nConfusion Matrix:")
print(cm)

print("\nCorrect predictions:", len(correct))
print("False Negatives:", len(false_negative))
print("False Positives:", len(false_positive))

print("\nFalse Negative percentage:",
      round(len(false_negative) / len(df) * 100, 2))

print("\nFalse Positive percentage:",
      round(len(false_positive) / len(df) * 100, 2))

print("\nSample False Negatives:")

for _, row in false_negative.head(5).iterrows():
    print("\nNews:", row["text"][:300])
    print("Actual:", row["label"])
    print("Predicted:", row["predicted_label"])

print("\nSample False Positives:")

for _, row in false_positive.head(5).iterrows():
    print("\nNews:", row["text"][:300])
    print("Actual:", row["label"])
    print("Predicted:", row["predicted_label"])

print("\nError analysis files saved!")
