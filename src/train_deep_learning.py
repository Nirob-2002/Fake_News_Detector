import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

import sys
import json
import gc
import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ==========================================
# 1. CONFIGURATION
# ==========================================

# Configure UTF-8 output safely (works in VS Code, terminal, and Colab).
for stream in (sys.stdout, sys.stderr):
    if stream is not None and hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass

SEED = 42
tf.keras.utils.set_random_seed(SEED)

# Limit CPU thread usage where possible.
try:
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except RuntimeError:
    print("TensorFlow already initialized. Using default thread settings.")

# Project directories (run this script from the project root).
SPLIT_DIR = Path("dataset/splits")
MODEL_DIR = Path("models/deep_learning")
RESULT_DIR = Path("results/deep_learning")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Model configuration
MAX_TOKENS = 12000
MAX_LENGTH = 250
EMBEDDING_DIM = 64
LSTM_UNITS = 32

BATCH_SIZE = 16
EPOCHS = 8
VALIDATION_SIZE = 0.20

# Search thresholds from 0.10 to 0.90, inclusive, in steps of 0.01.
THRESHOLDS = np.round(np.arange(0.10, 0.901, 0.01), 2)

# A threshold is preferred only when both Fake precision and Fake recall
# are at least this value. If none qualifies, the best Fake F1 is used.
MIN_PRECISION_RECALL = 0.60

LANGUAGES = ["English", "Bangla"]
MODEL_TYPES = ["LSTM", "BiLSTM"]

results = []

print("=" * 60)
print("BILINGUAL DEEP LEARNING")
print("LSTM AND BiLSTM TRAINING")
print("=" * 60)
print("TensorFlow Version:", tf.__version__)
print("Available Devices:", tf.config.list_physical_devices())


# ==========================================
# 2. MODEL BUILDER
# ==========================================

def build_model(vectorizer, model_type):
    inputs = tf.keras.Input(
        shape=(),
        dtype=tf.string,
        name="news_text",
    )

    # Convert text into integer token sequences.
    x = vectorizer(inputs)

    # Word embedding.
    x = tf.keras.layers.Embedding(
        input_dim=MAX_TOKENS,
        output_dim=EMBEDDING_DIM,
        mask_zero=True,
        name="embedding",
    )(x)

    # LSTM / BiLSTM layer.
    if model_type == "LSTM":
        x = tf.keras.layers.LSTM(
            LSTM_UNITS,
            name="lstm",
        )(x)

    elif model_type == "BiLSTM":
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                LSTM_UNITS,
                name="forward_backward_lstm",
            ),
            name="bilstm",
        )(x)

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Dropout for regularization.
    x = tf.keras.layers.Dropout(0.5)(x)

    # Fully connected layer.
    x = tf.keras.layers.Dense(
        32,
        activation="relu",
    )(x)

    # Binary classification output: probability of Fake (class 1).
    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="prediction",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.0003,
        ),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    return model


# ==========================================
# 3. DATA VALIDATION
# ==========================================

def validate_dataset(df, file_path):
    required_columns = {"clean_text", "label"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Missing required columns in {file_path}. "
            f"Required: {required_columns}. "
            f"Found: {df.columns.tolist()}"
        )

    if df.empty:
        raise ValueError(f"Dataset is empty: {file_path}")

    # Avoid modifying the original dataframe unexpectedly.
    df = df.copy()

    df["clean_text"] = df["clean_text"].fillna("").astype(str)
    df["label"] = df["label"].astype(str).str.strip()

    valid_labels = {"Fake", "Real"}
    found_labels = set(df["label"].unique())

    if not found_labels.issubset(valid_labels):
        raise ValueError(
            f"Invalid labels in {file_path}: {found_labels}. "
            "Labels must be exactly 'Fake' and 'Real'."
        )

    if found_labels != valid_labels:
        raise ValueError(
            f"Both Fake and Real classes are required in {file_path}. "
            f"Found: {found_labels}"
        )

    if (df["clean_text"].str.strip() == "").all():
        raise ValueError(f"All text entries are empty in {file_path}")

    return df


# ==========================================
# 4. BALANCED CLASS WEIGHTS
# ==========================================

def calculate_class_weights(y_train):
    class_counts = np.bincount(
        np.asarray(y_train, dtype=int),
        minlength=2,
    )

    if np.any(class_counts == 0):
        raise ValueError(
            "Both classes must be present in training data "
            "to calculate balanced class weights."
        )

    total_samples = int(np.sum(class_counts))

    # Class mapping: Real = 0, Fake = 1.
    class_weights = {
        0: total_samples / (2.0 * class_counts[0]),
        1: total_samples / (2.0 * class_counts[1]),
    }

    return class_weights


# ==========================================
# 5. IMPROVED VALIDATION THRESHOLD TUNING
# ==========================================

def find_best_threshold(y_true, probabilities):
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float).ravel()

    if len(y_true) != len(probabilities):
        raise ValueError(
            "y_true and probabilities must have the same number of samples."
        )

    best_threshold = 0.50
    best_f1 = -1.0
    threshold_results = []

    # First pass: evaluate all candidate thresholds.
    for threshold in THRESHOLDS:
        predictions = (probabilities >= threshold).astype(int)

        precision = precision_score(
            y_true,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_true,
            predictions,
            zero_division=0,
        )

        threshold_results.append({
            "Threshold": float(threshold),
            "Precision": float(precision),
            "Recall": float(recall),
            "F1": float(f1),
        })

        # Prefer thresholds with both precision and recall >= 0.60.
        if (
            precision >= MIN_PRECISION_RECALL
            and recall >= MIN_PRECISION_RECALL
            and f1 > best_f1
        ):
            best_f1 = float(f1)
            best_threshold = float(threshold)

    # Fallback: if no threshold meets both constraints, choose the
    # threshold with the highest Fake-class F1.
    if best_f1 < 0:
        for item in threshold_results:
            if item["F1"] > best_f1:
                best_f1 = float(item["F1"])
                best_threshold = float(item["Threshold"])

    return best_threshold, best_f1, threshold_results


# ==========================================
# 6. LANGUAGE TRAINING
# ==========================================

def train_language(language):
    print("\n" + "=" * 60)
    print("LANGUAGE:", language.upper())
    print("=" * 60)

    train_path = SPLIT_DIR / f"{language.lower()}_train.csv"
    test_path = SPLIT_DIR / f"{language.lower()}_test.csv"

    # Check dataset files.
    if not train_path.exists():
        raise FileNotFoundError(f"Training dataset not found: {train_path}")

    if not test_path.exists():
        raise FileNotFoundError(f"Testing dataset not found: {test_path}")

    # Load datasets.
    train_df = pd.read_csv(train_path, encoding="utf-8-sig")
    test_df = pd.read_csv(test_path, encoding="utf-8-sig")

    # Validate datasets.
    train_df = validate_dataset(train_df, train_path)
    test_df = validate_dataset(test_df, test_path)

    # Training features and labels.
    X = train_df["clean_text"]
    y = (train_df["label"] == "Fake").astype("int32")

    # Final test features and labels.
    X_test = test_df["clean_text"].to_numpy()
    y_test = (
        (test_df["label"] == "Fake")
        .astype("int32")
        .to_numpy()
    )

    # Create validation data from training data only.
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=VALIDATION_SIZE,
        random_state=SEED,
        stratify=y,
    )

    # Convert pandas objects to NumPy arrays.
    X_train = X_train.to_numpy()
    X_val = X_val.to_numpy()
    y_train = y_train.to_numpy()
    y_val = y_val.to_numpy()

    print("\nDataset Information:")
    print("Training Samples:", len(X_train))
    print("Validation Samples:", len(X_val))
    print("Final Test Samples:", len(X_test))
    print("Training Fake:", int(np.sum(y_train == 1)))
    print("Training Real:", int(np.sum(y_train == 0)))
    print("Validation Fake:", int(np.sum(y_val == 1)))
    print("Validation Real:", int(np.sum(y_val == 0)))
    print("Testing Fake:", int(np.sum(y_test == 1)))
    print("Testing Real:", int(np.sum(y_test == 0)))

    # ======================================
    # 7. CLASS WEIGHT CALCULATION
    # ======================================

    class_weights = calculate_class_weights(y_train)

    print("\nBalanced Class Weights:")
    print(class_weights)

    # ======================================
    # 8. TRAIN LSTM AND BiLSTM
    # ======================================

    for model_type in MODEL_TYPES:
        print("\n" + "-" * 50)
        print("LANGUAGE:", language)
        print("TRAINING MODEL:", model_type)
        print("-" * 50)

        # Clear previous model state.
        tf.keras.backend.clear_session()
        gc.collect()
        tf.keras.utils.set_random_seed(SEED)

        # Create a fresh vectorizer for each model.
        vectorizer = tf.keras.layers.TextVectorization(
            max_tokens=MAX_TOKENS,
            output_mode="int",
            output_sequence_length=MAX_LENGTH,
            standardize=None,
            name="text_vectorizer",
        )

        # Adapt vocabulary using training text only (no validation/test leakage).
        text_dataset = tf.data.Dataset.from_tensor_slices(
            X_train
        ).batch(128)

        vectorizer.adapt(text_dataset)

        print("\nVocabulary Size:", len(vectorizer.get_vocabulary()))

        # Build model.
        model = build_model(vectorizer, model_type)
        model.summary()

        # Best checkpoint path.
        model_path = (
            MODEL_DIR
            / f"{language.lower()}_{model_type.lower()}.keras"
        )

        # Callbacks.
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=2,
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(model_path),
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=False,
                verbose=1,
            ),
        ]

        # ==================================
        # 9. MODEL TRAINING
        # ==================================

        print("\nStarting Training...")

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=2,
            shuffle=True,
        )

        print("\nTraining Completed:", model_type)

        # ==================================
        # 10. LOAD BEST CHECKPOINT
        # ==================================

        print("\nLoading Best Model Checkpoint...")

        if not model_path.exists():
            raise FileNotFoundError(
                f"Best model checkpoint not found: {model_path}"
            )

        # The saved model contains its TextVectorization layer.
        model = tf.keras.models.load_model(model_path)

        # ==================================
        # 11. VALIDATION THRESHOLD TUNING
        # ==================================

        print("\nPerforming Improved Validation Threshold Tuning...")

        val_probabilities = model.predict(
            X_val,
            batch_size=BATCH_SIZE,
            verbose=0,
        ).ravel()

        (
            best_threshold,
            best_val_f1,
            threshold_results,
        ) = find_best_threshold(
            y_val,
            val_probabilities,
        )

        selected_row = next(
            item for item in threshold_results
            if item["Threshold"] == best_threshold
        )

        print("\nVALIDATION THRESHOLD RESULTS:")
        print("Best Threshold:", round(best_threshold, 2))
        print("Validation Fake Precision:", round(selected_row["Precision"], 4))
        print("Validation Fake Recall:", round(selected_row["Recall"], 4))
        print("Validation Fake F1:", round(best_val_f1, 4))

        if (
            selected_row["Precision"] < MIN_PRECISION_RECALL
            or selected_row["Recall"] < MIN_PRECISION_RECALL
        ):
            print(
                "Note: No threshold met both minimum precision and recall "
                f"requirements ({MIN_PRECISION_RECALL:.2f}); "
                "the fallback highest-F1 threshold was selected."
            )

        # Save selected threshold and validation metrics.
        threshold_path = (
            RESULT_DIR
            / f"{language.lower()}_{model_type.lower()}_threshold.json"
        )

        threshold_data = {
            "language": language,
            "model": model_type,
            "threshold": best_threshold,
            "validation_precision_fake": selected_row["Precision"],
            "validation_recall_fake": selected_row["Recall"],
            "validation_fake_f1": best_val_f1,
            "minimum_precision_recall_constraint": MIN_PRECISION_RECALL,
            "positive_class": "Fake",
            "negative_class": "Real",
        }

        with open(threshold_path, "w", encoding="utf-8") as file:
            json.dump(
                threshold_data,
                file,
                ensure_ascii=False,
                indent=4,
            )

        # Save all threshold search results.
        threshold_df = pd.DataFrame(threshold_results)
        threshold_search_path = (
            RESULT_DIR
            / f"{language.lower()}_{model_type.lower()}_threshold_search.csv"
        )
        threshold_df.to_csv(
            threshold_search_path,
            index=False,
            encoding="utf-8-sig",
        )

        print("Threshold Saved:", threshold_path)
        print("Threshold Search Saved:", threshold_search_path)

        # ==================================
        # 12. FINAL TEST EVALUATION
        # ==================================

        print("\nPerforming Final Test Evaluation...")

        # Test set is used only here, after threshold selection is complete.
        probabilities = model.predict(
            X_test,
            batch_size=BATCH_SIZE,
            verbose=0,
        ).ravel()

        predictions = (probabilities >= best_threshold).astype(int)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(
            y_test,
            predictions,
            zero_division=0,
        )
        recall = recall_score(
            y_test,
            predictions,
            zero_division=0,
        )
        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0,
        )

        print("\n" + "=" * 50)
        print("FINAL TEST RESULTS:", model_type)
        print("=" * 50)
        print("Threshold:", round(best_threshold, 4))
        print("Accuracy:", round(accuracy, 4))
        print("Precision (Fake):", round(precision, 4))
        print("Recall (Fake):", round(recall, 4))
        print("F1-Score (Fake):", round(f1, 4))

        # ==================================
        # 13. CLASSIFICATION REPORT
        # ==================================

        print("\nClassification Report:")

        report = classification_report(
            y_test,
            predictions,
            labels=[1, 0],
            target_names=["Fake", "Real"],
            zero_division=0,
        )
        print(report)

        report_path = (
            RESULT_DIR
            / f"{language.lower()}_{model_type.lower()}_report.txt"
        )

        with open(report_path, "w", encoding="utf-8") as file:
            file.write(report)

        # ==================================
        # 14. CONFUSION MATRIX
        # ==================================

        cm = confusion_matrix(
            y_test,
            predictions,
            labels=[1, 0],
        )

        print("\nConfusion Matrix:")
        print(cm)

        cm_df = pd.DataFrame(
            cm,
            index=["Actual Fake", "Actual Real"],
            columns=["Predicted Fake", "Predicted Real"],
        )

        cm_path = (
            RESULT_DIR
            / f"{language.lower()}_{model_type.lower()}_cm.csv"
        )
        cm_df.to_csv(cm_path, encoding="utf-8-sig")

        # ==================================
        # 15. SAVE MODEL RESULTS
        # ==================================

        results.append({
            "Language": language,
            "Model": model_type,
            "Accuracy": accuracy,
            "Precision_Fake": precision,
            "Recall_Fake": recall,
            "F1_Fake": f1,
            "Threshold": best_threshold,
            "Validation_Precision_Fake": selected_row["Precision"],
            "Validation_Recall_Fake": selected_row["Recall"],
            "Validation_F1": best_val_f1,
            "Training_Samples": len(X_train),
            "Validation_Samples": len(X_val),
            "Test_Samples": len(X_test),
        })

        print("\nSaved Model:", model_path)
        print("Saved Confusion Matrix:", cm_path)
        print("Saved Classification Report:", report_path)

        # ==================================
        # 16. MEMORY CLEANUP
        # ==================================

        del model
        del vectorizer
        del history
        del probabilities
        del predictions
        del val_probabilities
        del text_dataset

        gc.collect()
        tf.keras.backend.clear_session()

        print("\nCompleted:", language, model_type)


# ==========================================
# 17. TRAIN BOTH LANGUAGES
# ==========================================

for language in LANGUAGES:
    train_language(language)


# ==========================================
# 18. SAVE FINAL COMPARISON REPORT
# ==========================================

results_df = pd.DataFrame(results)

comparison_path = RESULT_DIR / "deep_learning_comparison.csv"

results_df.to_csv(
    comparison_path,
    index=False,
    encoding="utf-8-sig",
)


# ==========================================
# 19. FINAL SUMMARY
# ==========================================

print("\n" + "=" * 60)
print("DEEP LEARNING FINAL COMPARISON")
print("=" * 60)

if not results_df.empty:
    print(results_df.to_string(index=False))

print("\n" + "=" * 60)
print("ALL DEEP LEARNING EXPERIMENTS COMPLETED!")
print("=" * 60)

print("\nModels Directory:")
print(MODEL_DIR)

print("\nResults Directory:")
print(RESULT_DIR)

print("\nComparison Report:")
print(comparison_path)
