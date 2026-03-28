"""
create_urgency_labels.py
─────────────────────────
Adds rule-based urgency labels to the cleaned train/test CSVs.

Reads  : data/processed/train_clean.csv  |  data/processed/test_clean.csv
Saves  : data/processed/train_with_urgency.csv  |  data/processed/test_with_urgency.csv

Usage:
    python src/create_urgency_labels.py
"""

import os
import sys

import pandas as pd

# Allow importing sibling modules when run as a script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from urgency_rule_based import detect_urgency_rule

BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed")


def main() -> None:
    print("Loading cleaned datasets…")
    train_df = pd.read_csv(os.path.join(PROCESSED_PATH, "train_clean.csv"))
    test_df  = pd.read_csv(os.path.join(PROCESSED_PATH, "test_clean.csv"))
    print(f"  train: {len(train_df):,}  |  test: {len(test_df):,}")

    print("Generating urgency labels (rule-based)…")
    train_df["urgency"] = train_df["clean_text"].apply(detect_urgency_rule)
    test_df["urgency"]  = test_df["clean_text"].apply(detect_urgency_rule)

    train_df.to_csv(os.path.join(PROCESSED_PATH, "train_with_urgency.csv"), index=False)
    test_df.to_csv(os.path.join(PROCESSED_PATH, "test_with_urgency.csv"),   index=False)
    print("✅  Urgency datasets saved.")


if __name__ == "__main__":
    main()
