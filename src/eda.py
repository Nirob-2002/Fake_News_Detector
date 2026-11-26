
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# File locations
DATASET_FILE = Path("dataset/cleaned/cleaned_master.csv")
RESULTS_DIR = Path("results")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Check dataset
if not DATASET_FILE.exists():
    raise FileNotFoundError(
        "Cleaned dataset not found. Run clean_dataset.py first."
    )

# Load cleaned dataset
df = pd.read_csv(DATASET_FILE, encoding="utf-8-sig")

# Prepare text length
df["text"] = df["text"].fillna("").astype(str)
df["text_length"] = df["text"].str.len()

sns.set_theme(style="whitegrid")

print("=" * 50)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 50)

print("\nTotal records:", len(df))

# 1. Language distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="language")
plt.title("News Distribution by Language")
plt.xlabel("Language")
plt.ylabel("Number of News")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "language_distribution.png")
plt.close()

# 2. Overall label distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="label")
plt.title("Fake vs Real News Distribution")
plt.xlabel("News Label")
plt.ylabel("Number of News")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "label_distribution.png")
plt.close()

# 3. Language-wise label distribution
plt.figure(figsize=(9, 5))
sns.countplot(data=df, x="language", hue="label")
plt.title("Fake vs Real News by Language")
plt.xlabel("Language")
plt.ylabel("Number of News")
plt.legend(title="Label")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "language_label_comparison.png")
plt.close()

# 4. Text length distribution
plt.figure(figsize=(9, 5))
sns.histplot(
    data=df,
    x="text_length",
    hue="language",
    bins=50,
    element="step",
    stat="density",
    common_norm=False
)
plt.title("News Text Length Distribution")
plt.xlabel("Number of Characters")
plt.ylabel("Density")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "text_length_distribution.png")
plt.close()

# 5. Language-wise label percentages
print("\nLanguage-wise label percentages:")
percentage_table = pd.crosstab(
    df["language"],
    df["label"],
    normalize="index"
) * 100

print(percentage_table.round(2))

# 6. Text length statistics
print("\nText length statistics:")
print(
    df.groupby("language")["text_length"]
    .describe()
    .round(2)
)

print("\nGraphs saved in results folder:")
print("1. language_distribution.png")
print("2. label_distribution.png")
print("3. language_label_comparison.png")
print("4. text_length_distribution.png")

print("\nEDA COMPLETED SUCCESSFULLY!")
