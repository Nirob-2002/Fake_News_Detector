
import pandas as pd
from pathlib import Path

# File paths
INPUT_FILE = Path("dataset/ben_fnd_16k.csv")
OUTPUT_DIR = Path("dataset/cleaned")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 55)
print("FAKE NEWS DATASET CLEANING")
print("=" * 55)

# Load original dataset
df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

print("\nOriginal records:", len(df))

# 1. Clean column text and normalize missing values
for col in ["title", "news_details", "language", "label"]:
    df[col] = df[col].replace(r"^\s*$", pd.NA, regex=True)

# 2. Normalize language and label
df["language"] = df["language"].str.strip().str.title()
df["label"] = df["label"].str.strip().str.title()

# 3. Remove records without language or label
before = len(df)

df = df.dropna(subset=["language", "label"])

print("Removed records with missing language/label:",
      before - len(df))

# 4. Keep only supported languages and labels
df = df[
    df["language"].isin(["English", "Bangla"])
    & df["label"].isin(["Fake", "Real"])
].copy()

# 5. Missing title/details:
# Keep a record if at least one text field is available
df["title"] = df["title"].fillna("").str.strip()
df["news_details"] = (
    df["news_details"].fillna("").str.strip()
)

before = len(df)

df = df[
    (df["title"] != "") | (df["news_details"] != "")
].copy()

print("Removed records with no news text:", before - len(df))

# 6. Remove duplicate news based on language and text
before = len(df)

df = df.drop_duplicates(
    subset=["language", "title", "news_details"]
).copy()

print("Removed duplicate news:", before - len(df))

# 7. Create combined text for future model training
df["text"] = (
    df["title"] + " " + df["news_details"]
).str.strip()

# 8. Save cleaned master dataset
master_file = OUTPUT_DIR / "cleaned_master.csv"

df.to_csv(master_file, index=False, encoding="utf-8-sig")

# 9. Save separate language datasets
english_df = df[df["language"] == "English"].copy()
bangla_df = df[df["language"] == "Bangla"].copy()

english_df.to_csv(
    OUTPUT_DIR / "english_cleaned.csv",
    index=False,
    encoding="utf-8-sig"
)

bangla_df.to_csv(
    OUTPUT_DIR / "bangla_cleaned.csv",
    index=False,
    encoding="utf-8-sig"
)

# 10. Final report
print("\n========== CLEANING REPORT ==========")

print("Final total records:", len(df))

print("\nLanguage distribution:")
print(df["language"].value_counts())

print("\nLabel distribution:")
print(df["label"].value_counts())

print("\nLanguage-wise label distribution:")
print(pd.crosstab(df["language"], df["label"]))

print("\nEnglish records:", len(english_df))
print("Bangla records:", len(bangla_df))

print("\nSaved files:")
print(master_file)
print(OUTPUT_DIR / "english_cleaned.csv")
print(OUTPUT_DIR / "bangla_cleaned.csv")

print("\nDATA CLEANING COMPLETED SUCCESSFULLY!")
