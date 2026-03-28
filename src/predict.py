"""
predict.py
───────────
Standalone CLI tool to classify a single email using the trained pkl models.

Usage:
    python src/predict.py
    python src/predict.py --text "Server is down, urgent fix needed!"
"""

import argparse
import os
import pickle
import re
import sys

import nltk

# ── NLTK bootstrap ─────────────────────────────────────────────────────────────
for resource in ("punkt", "punkt_tab", "stopwords", "wordnet"):
    try:
        nltk.data.find(f"tokenizers/{resource}" if "punkt" in resource else resource)
    except LookupError:
        nltk.download(resource, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

CATEGORY_MAPPING = {
    # Legacy numeric classes from the HF dataset.
    0: "forum",
    1: "promotions",
    2: "social_media",
    3: "spam",
    4: "updates",
    5: "verify_code",
}

# ── Text cleaning ──────────────────────────────────────────────────────────────
_stop_words = set(stopwords.words("english"))
_lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+",        " ", text)
    text = re.sub(r"[^a-zA-Z\s]",    " ", text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in _stop_words and len(w) > 2]
    tokens = [_lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)


# ── Model loader ───────────────────────────────────────────────────────────────
def _load(filename: str):
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found: {path}\n"
            "Run the training pipeline first:\n"
            "  python src/preprocess.py\n"
            "  python src/train_model.py\n"
            "  python src/create_urgency_labels.py\n"
            "  python src/train_urgency_model.py"
        )
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Prediction ─────────────────────────────────────────────────────────────────
def predict_email(raw_text: str) -> tuple:
    """
    Returns (category, category_conf%, urgency, urgency_conf%)
    """
    cat_model = _load("email_classifier.pkl")
    cat_vec   = _load("vectorizer.pkl")
    urg_model = _load("urgency_model.pkl")
    urg_vec   = _load("urgency_vectorizer.pkl")

    cleaned = clean_text(raw_text)

    # Category
    cat_tfidf = cat_vec.transform([cleaned])
    cat_raw = cat_model.predict(cat_tfidf)[0]
    cat_conf = float(max(cat_model.predict_proba(cat_tfidf)[0])) * 100

    # Supports both newer string-label models and legacy numeric-label models.
    if isinstance(cat_raw, (int, float)):
        category = CATEGORY_MAPPING.get(int(cat_raw), "unknown")
    else:
        category = str(cat_raw)

    # Urgency
    urg_tfidf  = urg_vec.transform([cleaned])
    urgency    = str(urg_model.predict(urg_tfidf)[0])
    urg_conf   = float(max(urg_model.predict_proba(urg_tfidf)[0])) * 100

    return category, cat_conf, urgency, urg_conf


# ── CLI ────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Classify an email.")
    parser.add_argument("--text", type=str, default=None,
                        help="Email text (if omitted, prompts interactively)")
    args = parser.parse_args()

    text = args.text or input("Enter email text: ")
    if not text.strip():
        print("No text provided. Exiting.")
        sys.exit(1)

    cat, cat_conf, urg, urg_conf = predict_email(text)

    print("\n" + "=" * 40)
    print("  Prediction Result")
    print("=" * 40)
    print(f"  Category : {cat} ({cat_conf:.2f}%)")
    print(f"  Urgency  : {urg} ({urg_conf:.2f}%)")
    print("=" * 40)


if __name__ == "__main__":
    main()
