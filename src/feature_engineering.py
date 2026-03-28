"""
feature_engineering.py
───────────────────────
TF-IDF vectoriser utilities shared across training scripts.
"""

import pickle

from sklearn.feature_extraction.text import TfidfVectorizer


def create_vectorizer(max_features: int = 5000) -> TfidfVectorizer:
    return TfidfVectorizer(max_features=max_features, stop_words="english")


def fit_vectorizer(vectorizer: TfidfVectorizer, X_train):
    return vectorizer.fit_transform(X_train)


def transform_vectorizer(vectorizer: TfidfVectorizer, X):
    return vectorizer.transform(X)


def save_vectorizer(vectorizer: TfidfVectorizer, path: str = "models/vectorizer.pkl") -> None:
    with open(path, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"Vectorizer saved → {path}")


def load_vectorizer(path: str = "models/vectorizer.pkl") -> TfidfVectorizer:
    with open(path, "rb") as f:
        return pickle.load(f)
