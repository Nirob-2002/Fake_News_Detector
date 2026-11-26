
import pandas as pd
from pathlib import Path

INPUT = Path("dataset/cleaned/cleaned_master.csv")
OUTPUT = Path("results/label_audit")

OUTPUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT, encoding="utf-8-sig")

# Randomly select 25 records from each language-label group
audit_samples = []

for language in ["English", "Bangla"]:
    for label in ["Fake", "Real"]:

        group = df[
            (df["language"] == language) &
            (df["label"] == label)
        ]

        sample = group.sample(
            n=min(25, len(group)),
            random_state=42
        )

        audit_samples.append(sample)

audit_df = pd.concat(audit_samples)

# Keep useful columns
columns = [
    col for col in
    ["id", "title", "news_details", "language",
     "category", "label"]
    if col in audit_df.columns
]

audit_df[columns].to_csv(
    OUTPUT / "random_label_audit.csv",
    index=False,
    encoding="utf-8-sig"
)

print("=" * 50)
print("DATASET LABEL QUALITY AUDIT")
print("=" * 50)

print("Total audit samples:", len(audit_df))

print("\nLanguage and label distribution:")
print(
    pd.crosstab(
        audit_df["language"],
        audit_df["label"]
    )
)

print("\nAudit file saved:")
print(OUTPUT / "random_label_audit.csv")
