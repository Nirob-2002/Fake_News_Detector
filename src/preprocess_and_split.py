
import pandas as pd
import re
import unicodedata

from pathlib import Path
from sklearn.model_selection import train_test_split

# ----------------------------------
# 1. File paths
# ----------------------------------

INPUT_FILE = Path("dataset/cleaned/cleaned_master.csv")

PROCESSED_DIR = Path("dataset/processed")
SPLIT_DIR = Path("dataset/splits")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------
# 2. Text cleaning function
# ----------------------------------

def clean_text(text, language):

    if pd.isna(text):
        return ""

    text = str(text)

    # Normalize Unicode characters
    text = unicodedata.normalize("NFC", text)

    # Remove URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    # Remove HTML tags
    text = re.sub(
        r"<[^>]+>",
        " ",
        text
    )

   # Remove only zero-width space and BOM.
    # Preserve ZWNJ and ZWJ used in Unicode text.
    text = re.sub(
    r"[\u200b\ufeff]",
    "",
    text
)

    # Convert English letters to lowercase
    if language == "English":
        text = text.lower()

    # Preserve Bangla Unicode characters and combining marks
    text = re.sub(
    r"[^\w\s\u0980-\u09FF]",
    " ",
    text,
    flags=re.UNICODE
)

    # Remove underscores
    text = text.replace("_", " ")

    # Normalize multiple spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ----------------------------------
# 3. Load cleaned dataset
# ----------------------------------

print("=" * 55)
print("BILINGUAL TEXT PREPROCESSING")
print("=" * 55)

df = pd.read_csv(
    INPUT_FILE,
    encoding="utf-8-sig"
)

print("\nOriginal records:", len(df))

# ----------------------------------
# 4. Create cleaned text
# ----------------------------------

df["clean_text"] = df.apply(
    lambda row: clean_text(
        row["text"],
        row["language"]
    ),
    axis=1
)

# Remove records with empty cleaned text
before = len(df)

df = df[df["clean_text"].str.strip() != ""].copy()

print("Empty records removed:", before - len(df))

# ----------------------------------
# 5. Save processed master dataset
# ----------------------------------

processed_file = PROCESSED_DIR / "processed_master.csv"

df.to_csv(
    processed_file,
    index=False,
    encoding="utf-8-sig"
)

# ----------------------------------
# 6. Split each language separately
# ----------------------------------

for language in ["English", "Bangla"]:

    language_df = df[
        df["language"] == language
    ].copy()

    train_df, test_df = train_test_split(
        language_df,
        test_size=0.20,
        random_state=42,
        stratify=language_df["label"]
    )

    # Save training data
    train_file = SPLIT_DIR / f"{language.lower()}_train.csv"

    train_df.to_csv(
        train_file,
        index=False,
        encoding="utf-8-sig"
    )

    # Save testing data
    test_file = SPLIT_DIR / f"{language.lower()}_test.csv"

    test_df.to_csv(
        test_file,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\n{language} Dataset:")
    print("Total:", len(language_df))
    print("Training:", len(train_df))
    print("Testing:", len(test_df))

    print("\nTraining label distribution:")
    print(train_df["label"].value_counts())

    print("\nTesting label distribution:")
    print(test_df["label"].value_counts())

# ----------------------------------
# 7. Show sample cleaned text
# ----------------------------------

print("\n========== CLEANING EXAMPLES ==========")

for language in ["English", "Bangla"]:

    sample = df[df["language"] == language].head(1)

    if not sample.empty:

        print(f"\n{language}:")

        print("Original:")
        print(sample.iloc[0]["text"][:300])

        print("\nCleaned:")
        print(sample.iloc[0]["clean_text"][:300])

# ----------------------------------
# 8. Final report
# ----------------------------------

print("\n========== FINAL REPORT ==========")

print("Processed records:", len(df))

print("\nFiles saved:")
print(processed_file)
print(SPLIT_DIR / "english_train.csv")
print(SPLIT_DIR / "english_test.csv")
print(SPLIT_DIR / "bangla_train.csv")
print(SPLIT_DIR / "bangla_test.csv")

print("\nTEXT PREPROCESSING COMPLETED!")
