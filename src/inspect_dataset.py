
import pandas as pd
from pathlib import Path

# Dataset file location
DATASET_PATH = Path("dataset/ben_fnd_16k.csv")

print("=" * 55)
print("BILINGUAL FAKE NEWS DETECTION - DATASET INSPECTION")
print("=" * 55)

# Check whether the file exists
if not DATASET_PATH.exists():
    print("ERROR: Dataset file not found!")
    print("Expected location:", DATASET_PATH)
    raise SystemExit

# Load dataset
df = pd.read_csv(DATASET_PATH, encoding="utf-8-sig")

print("\n1. Dataset loaded successfully!")
print("Total rows:", len(df))
print("Total columns:", len(df.columns))

# Show column names
print("\n2. Available columns:")
print(df.columns.tolist())

# Required columns
required_columns = [
    "title",
    "news_details",
    "language",
    "label"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    print("\nERROR: Required columns are missing:")
    print(missing_columns)
    raise SystemExit

# Missing values
print("\n3. Missing values:")
print(df[required_columns].isnull().sum())

# Language distribution
print("\n4. Language distribution:")
print(df["language"].value_counts(dropna=False))

# Fake vs Real distribution
print("\n5. Label distribution:")
print(df["label"].value_counts(dropna=False))

# Duplicate rows
print("\n6. Duplicate rows:", df.duplicated().sum())

# Text length analysis
df["text_length"] = (
    df["title"].fillna("").astype(str)
    + " "
    + df["news_details"].fillna("").astype(str)
).str.len()

print("\n7. Text length statistics:")
print(df["text_length"].describe())

# Show sample records
print("\n8. Sample news records:")
print(
    df[["title", "language", "label"]]
    .head(5)
    .to_string(index=False)
)

print("\nDataset inspection completed!")
