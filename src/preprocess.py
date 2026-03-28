"""
preprocess.py
─────────────
Loads the HuggingFace email dataset, cleans text, and saves processed CSVs.

Usage:
    python src/preprocess.py
"""

import os
import re

import nltk
import pandas as pd
from datasets import load_dataset
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ── NLTK resources ─────────────────────────────────────────────────────────────
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet",   quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# ── Text cleaner ───────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Lower-case, remove noise, tokenise, remove stop-words, lemmatise."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)      # URLs
    text = re.sub(r"\S+@\S+",        " ", text)       # email addresses
    text = re.sub(r"[^a-zA-Z\s]",    " ", text)       # special chars / digits
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    print("Loading dataset from HuggingFace…")
    dataset  = load_dataset("jason23322/high-accuracy-email-classifier")
    train_df = pd.DataFrame(dataset["train"])
    test_df  = pd.DataFrame(dataset["test"])
    print(f"  train: {len(train_df):,}  |  test: {len(test_df):,}")

    print("Cleaning text…")
    train_df["clean_text"] = train_df["text"].apply(clean_text)
    test_df["clean_text"]  = test_df["text"].apply(clean_text)

    BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_path = os.path.join(BASE_DIR, "data", "processed")
    os.makedirs(processed_path, exist_ok=True)

    train_df.to_csv(os.path.join(processed_path, "train_clean.csv"), index=False)
    test_df.to_csv(os.path.join(processed_path, "test_clean.csv"),  index=False)
    print(f"✅  Saved to {processed_path}")


if __name__ == "__main__":
    main()
