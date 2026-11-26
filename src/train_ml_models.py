
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

SPLIT_DIR = Path("dataset/splits")
MODEL_DIR = Path("models")
RESULT_DIR = Path("results/ml_models")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

LANGUAGES = ["English", "Bangla"]

MODEL_CONFIGS = {
    "LogisticRegression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "NaiveBayes": MultinomialNB(),

    "SVM": LinearSVC(
        random_state=42,
        max_iter=3000
    )
}

all_results = []

# ==========================================
# 2. TRAINING FUNCTION
# ==========================================

def train_language_models(language):

    print("\n" + "=" * 55)
    print(f"TRAINING MODELS: {language.upper()}")
    print("=" * 55)

    # Load training and testing datasets
    train_file = SPLIT_DIR / f"{language.lower()}_train.csv"
    test_file = SPLIT_DIR / f"{language.lower()}_test.csv"

    train_df = pd.read_csv(train_file, encoding="utf-8-sig")
    test_df = pd.read_csv(test_file, encoding="utf-8-sig")

    X_train = train_df["clean_text"].fillna("").astype(str)
    X_test = test_df["clean_text"].fillna("").astype(str)

    y_train = train_df["label"]
    y_test = test_df["label"]

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # ======================================
    # 3. TF-IDF FEATURE EXTRACTION
    # ======================================

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=50000,
        min_df=2,
        sublinear_tf=True
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("\nTF-IDF Features:", X_train_tfidf.shape[1])

    # Save vectorizer separately
    vectorizer_path = (
        MODEL_DIR / f"{language.lower()}_tfidf.joblib"
    )

    joblib.dump(vectorizer, vectorizer_path)

    # ======================================
    # 4. TRAIN AND EVALUATE MODELS
    # ======================================

    for model_name, model in MODEL_CONFIGS.items():

        print("\n" + "-" * 45)
        print("Model:", model_name)

        # Train model
        model.fit(X_train_tfidf, y_train)

        # Prediction
        predictions = model.predict(X_test_tfidf)

        # Evaluation metrics
        accuracy = accuracy_score(y_test, predictions)

        precision = precision_score(
            y_test,
            predictions,
            pos_label="Fake",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            pos_label="Fake",
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            pos_label="Fake",
            zero_division=0
        )

        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")

        print("\nClassification Report:")
        print(
            classification_report(
                y_test,
                predictions,
                labels=["Fake", "Real"],
                zero_division=0
            )
        )

        # Save trained model
        model_path = (
            MODEL_DIR /
            f"{language.lower()}_{model_name.lower()}.joblib"
        )

        joblib.dump(model, model_path)

        # ==================================
        # 5. CONFUSION MATRIX
        # ==================================

        cm = confusion_matrix(
            y_test,
            predictions,
            labels=["Fake", "Real"]
        )

        plt.figure(figsize=(6, 5))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Fake", "Real"],
            yticklabels=["Fake", "Real"]
        )

        plt.title(f"{language} - {model_name}")
        plt.xlabel("Predicted Label")
        plt.ylabel("Actual Label")

        plt.tight_layout()

        plt.savefig(
            RESULT_DIR /
            f"{language.lower()}_{model_name.lower()}_cm.png"
        )

        plt.close()

        # Store results
        all_results.append({
            "Language": language,
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision_Fake": precision,
            "Recall_Fake": recall,
            "F1_Fake": f1
        })

        print("Model saved:", model_path)

# ==========================================
# 6. TRAIN BOTH LANGUAGES
# ==========================================

print("\nBILINGUAL FAKE NEWS DETECTION")
print("MACHINE LEARNING MODEL TRAINING")

for language in LANGUAGES:
    train_language_models(language)

# ==========================================
# 7. SAVE COMPARISON REPORT
# ==========================================

results_df = pd.DataFrame(all_results)

results_df = results_df.sort_values(
    by=["Language", "F1_Fake"],
    ascending=[True, False]
)

print("\n" + "=" * 55)
print("FINAL MODEL COMPARISON")
print("=" * 55)

print(results_df.to_string(index=False))

results_df.to_csv(
    RESULT_DIR / "model_comparison.csv",
    index=False
)

print("\nAll models trained successfully!")
print("Models directory:", MODEL_DIR)
print("Results directory:", RESULT_DIR)
