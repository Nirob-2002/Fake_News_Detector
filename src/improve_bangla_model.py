
import pandas as pd
import joblib

from pathlib import Path
from scipy.sparse import hstack

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# =====================================
# 1. PATHS
# =====================================

SPLIT_DIR = Path("dataset/splits")
MODEL_DIR = Path("models/bangla_optimized")

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# =====================================
# 2. LOAD DATA
# =====================================

train_df = pd.read_csv(
    SPLIT_DIR / "bangla_train.csv",
    encoding="utf-8-sig"
)

test_df = pd.read_csv(
    SPLIT_DIR / "bangla_test.csv",
    encoding="utf-8-sig"
)

X = train_df["clean_text"].fillna("").astype(str)
y = train_df["label"]

X_test = test_df["clean_text"].fillna("").astype(str)
y_test = test_df["label"]

# Validation split from training data only
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("=" * 55)
print("BANGLA MODEL OPTIMIZATION")
print("=" * 55)

print("Training:", len(X_train))
print("Validation:", len(X_val))
print("Final Test:", len(X_test))

# =====================================
# 3. FEATURE CONFIGURATIONS
# =====================================

def create_features(feature_type):

    word_vectorizer = None
    char_vectorizer = None

    if feature_type in ["word", "combined"]:

        word_vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            max_features=30000,
            min_df=2,
            sublinear_tf=True
        )

        word_train = word_vectorizer.fit_transform(X_train)
        word_val = word_vectorizer.transform(X_val)

    if feature_type in ["char", "combined"]:

        char_vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            max_features=30000,
            min_df=2,
            sublinear_tf=True
        )

        char_train = char_vectorizer.fit_transform(X_train)
        char_val = char_vectorizer.transform(X_val)

    if feature_type == "word":
        return word_train, word_val

    if feature_type == "char":
        return char_train, char_val

    if feature_type == "combined":
        return (
            hstack([word_train, char_train]).tocsr(),
            hstack([word_val, char_val]).tocsr()
        )


# =====================================
# 4. VALIDATION EXPERIMENTS
# =====================================

results = []

best_score = -1
best_config = None

for feature_type in ["word", "char", "combined"]:

    print("\nFeature type:", feature_type)

    X_train_features, X_val_features = create_features(
        feature_type
    )

    for c_value in [0.5, 1.0, 2.0]:

        model = LinearSVC(
            C=c_value,
            class_weight="balanced",
            random_state=42,
            max_iter=5000
        )

        model.fit(X_train_features, y_train)

        predictions = model.predict(X_val_features)

        precision = precision_score(
            y_val,
            predictions,
            pos_label="Fake",
            zero_division=0
        )

        recall = recall_score(
            y_val,
            predictions,
            pos_label="Fake",
            zero_division=0
        )

        f1 = f1_score(
            y_val,
            predictions,
            pos_label="Fake",
            zero_division=0
        )

        accuracy = accuracy_score(y_val, predictions)

        results.append({
            "Features": feature_type,
            "C": c_value,
            "Accuracy": accuracy,
            "Fake_Precision": precision,
            "Fake_Recall": recall,
            "Fake_F1": f1
        })

        print(
            f"C={c_value} | "
            f"Accuracy={accuracy:.4f} | "
            f"Fake Recall={recall:.4f} | "
            f"Fake F1={f1:.4f}"
        )

        if f1 > best_score:
            best_score = f1
            best_config = (feature_type, c_value)


# =====================================
# 5. SHOW VALIDATION RESULTS
# =====================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    MODEL_DIR / "validation_comparison.csv",
    index=False
)

print("\nVALIDATION COMPARISON")
print(results_df.to_string(index=False))

print("\nBest configuration:", best_config)
print("Best validation Fake F1:", round(best_score, 4))


# =====================================
# 6. REFIT ON FULL TRAINING DATA
# =====================================

feature_type, c_value = best_config

word_vectorizer = None
char_vectorizer = None

if feature_type in ["word", "combined"]:

    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=30000,
        min_df=2,
        sublinear_tf=True
    )

    train_word = word_vectorizer.fit_transform(X)
    test_word = word_vectorizer.transform(X_test)

if feature_type in ["char", "combined"]:

    char_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        max_features=30000,
        min_df=2,
        sublinear_tf=True
    )

    train_char = char_vectorizer.fit_transform(X)
    test_char = char_vectorizer.transform(X_test)

if feature_type == "word":
    X_train_final = train_word
    X_test_final = test_word

elif feature_type == "char":
    X_train_final = train_char
    X_test_final = test_char

else:
    X_train_final = hstack(
        [train_word, train_char]
    ).tocsr()

    X_test_final = hstack(
        [test_word, test_char]
    ).tocsr()

# =====================================
# 7. FINAL MODEL
# =====================================

final_model = LinearSVC(
    C=c_value,
    class_weight="balanced",
    random_state=42,
    max_iter=5000
)

final_model.fit(X_train_final, y)

test_predictions = final_model.predict(X_test_final)

print("\n" + "=" * 55)
print("FINAL TEST RESULTS")
print("=" * 55)

print("Accuracy:", accuracy_score(y_test, test_predictions))

print("Fake Precision:", precision_score(
    y_test, test_predictions,
    pos_label="Fake"
))

print("Fake Recall:", recall_score(
    y_test, test_predictions,
    pos_label="Fake"
))

print("Fake F1:", f1_score(
    y_test, test_predictions,
    pos_label="Fake"
))

print("\nClassification Report:")

print(classification_report(
    y_test,
    test_predictions,
    labels=["Fake", "Real"]
))

# =====================================
# 8. SAVE MODEL AND VECTORIZERS
# =====================================

joblib.dump(
    final_model,
    MODEL_DIR / "bangla_optimized_svm.joblib"
)

if word_vectorizer is not None:
    joblib.dump(
        word_vectorizer,
        MODEL_DIR / "word_vectorizer.joblib"
    )

if char_vectorizer is not None:
    joblib.dump(
        char_vectorizer,
        MODEL_DIR / "char_vectorizer.joblib"
    )

print("\nOptimized Bangla model saved!")
