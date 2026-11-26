"""
BanglaBERT Fine-Tuning Pipeline for Bangla Fake News Detection
Model: csebuetnlp/banglabert
This script trains a state-of-the-art Transformer model on the split Bangla dataset.
Can be executed locally (CPU/GPU) or inside Google Colab / Kaggle.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)

# -------------------------------------------------------------
# 1. DIRECTORY CONFIGURATION
# -------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dataset" / "splits"
OUTPUT_MODEL_DIR = BASE_DIR / "models" / "bangla_bert"
OUTPUT_RESULT_DIR = BASE_DIR / "results" / "bangla_bert"

OUTPUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_RESULT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "csebuetnlp/banglabert"
MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5

# -------------------------------------------------------------
# 2. DATASET LOADER
# -------------------------------------------------------------
class NewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

# -------------------------------------------------------------
# 3. METRICS COMPUTATION
# -------------------------------------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    acc = accuracy_score(labels, preds)
    precision_fake = precision_score(labels, preds, pos_label=1, zero_division=0)
    recall_fake = recall_score(labels, preds, pos_label=1, zero_division=0)
    f1_fake = f1_score(labels, preds, pos_label=1, zero_division=0)

    return {
        "accuracy": acc,
        "precision_fake": precision_fake,
        "recall_fake": recall_fake,
        "f1_fake": f1_fake,
    }

# -------------------------------------------------------------
# 4. MAIN TRAINING ROUTINE
# -------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("BANGLA FAKE NEWS DETECTION - BANGLABERT FINE-TUNING")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"Base Model: {MODEL_NAME}")
    print("=" * 60)

    train_file = DATA_DIR / "bangla_train.csv"
    test_file = DATA_DIR / "bangla_test.csv"

    print(f"Loading training data: {train_file}")
    train_df = pd.read_csv(train_file, encoding="utf-8-sig")
    print(f"Loading test data: {test_file}")
    test_df = pd.read_csv(test_file, encoding="utf-8-sig")

    # Map labels: Real -> 0, Fake -> 1
    label2id = {"Real": 0, "Fake": 1}
    id2label = {0: "Real", 1: "Fake"}

    train_texts = train_df["clean_text"].fillna("").astype(str).tolist()
    train_labels = [label2id[str(lbl).strip()] for lbl in train_df["label"]]

    test_texts = test_df["clean_text"].fillna("").astype(str).tolist()
    test_labels = [label2id[str(lbl).strip()] for lbl in test_df["label"]]

    # Validation split from training (15%)
    from sklearn.model_selection import train_test_split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts,
        train_labels,
        test_size=0.15,
        random_state=42,
        stratify=train_labels
    )

    print(f"Train samples: {len(train_texts)}")
    print(f"Validation samples: {len(val_texts)}")
    print(f"Test samples: {len(test_texts)}")

    # Load Tokenizer
    print("\nLoading BanglaBERT Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Tokenizing train texts...")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=MAX_LENGTH)
    print("Tokenizing validation texts...")
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=MAX_LENGTH)
    print("Tokenizing test texts...")
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=MAX_LENGTH)

    train_dataset = NewsDataset(train_encodings, train_labels)
    val_dataset = NewsDataset(val_encodings, val_labels)
    test_dataset = NewsDataset(test_encodings, test_labels)

    # Load Pre-trained BanglaBERT Model
    print("\nLoading BanglaBERT Sequence Classification Model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label=id2label,
        label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_MODEL_DIR / "checkpoints"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        warmup_ratio=0.1,
        weight_decay=0.01,
        learning_rate=LEARNING_RATE,
        logging_dir=str(OUTPUT_RESULT_DIR / "logs"),
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_fake",
        greater_is_better=True,
        save_total_limit=1,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    print("\nStarting BanglaBERT Fine-Tuning...")
    trainer.train()

    print("\nEvaluating on Test Set...")
    test_results = trainer.predict(test_dataset)
    test_logits = test_results.predictions
    test_preds = np.argmax(test_logits, axis=1)

    acc = accuracy_score(test_labels, test_preds)
    prec_fake = precision_score(test_labels, test_preds, pos_label=1, zero_division=0)
    rec_fake = recall_score(test_labels, test_preds, pos_label=1, zero_division=0)
    f1_f = f1_score(test_labels, test_preds, pos_label=1, zero_division=0)
    cm = confusion_matrix(test_labels, test_preds)
    report = classification_report(test_labels, test_preds, target_names=["Real", "Fake"], digits=4)

    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS (BanglaBERT)")
    print("=" * 60)
    print(f"Accuracy:       {acc:.4f}")
    print(f"Fake Precision: {prec_fake:.4f}")
    print(f"Fake Recall:    {rec_fake:.4f}")
    print(f"Fake F1:        {f1_f:.4f}")
    print("\nConfusion Matrix [[TN, FP], [FN, TP]]:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    # Save Final Model and Tokenizer
    print(f"\nSaving best fine-tuned model and tokenizer to: {OUTPUT_MODEL_DIR}")
    model.save_pretrained(OUTPUT_MODEL_DIR)
    tokenizer.save_pretrained(OUTPUT_MODEL_DIR)

    # Save metrics
    metrics = {
        "model": "csebuetnlp/banglabert",
        "accuracy": float(acc),
        "fake_precision": float(prec_fake),
        "fake_recall": float(rec_fake),
        "fake_f1": float(f1_f),
        "confusion_matrix": cm.tolist()
    }
    with open(OUTPUT_RESULT_DIR / "banglabert_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    with open(OUTPUT_RESULT_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    # Save detailed predictions
    test_df["predicted_label"] = [id2label[p] for p in test_preds]
    # Softmax probabilities
    probs = torch.softmax(torch.tensor(test_logits), dim=1).numpy()
    test_df["fake_probability"] = probs[:, 1]
    test_df["confidence"] = np.max(probs, axis=1)
    test_df.to_csv(OUTPUT_RESULT_DIR / "banglabert_test_predictions.csv", index=False, encoding="utf-8-sig")

    print("\nBanglaBERT training & evaluation completed successfully!")

if __name__ == "__main__":
    main()
