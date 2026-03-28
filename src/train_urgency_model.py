"""
train_urgency_model.py
───────────────────────
Trains a TF-IDF + Logistic Regression urgency classifier.

Reads  : data/processed/train_with_urgency.csv  |  data/processed/test_with_urgency.csv
Saves  : models/urgency_model.pkl               |  models/urgency_vectorizer.pkl

Usage:
    python src/train_urgency_model.py
"""

import os
import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def main() -> None:
    print("Loading urgency datasets…")
    train_df = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "train_with_urgency.csv"))
    test_df  = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "test_with_urgency.csv"))

    X_train, y_train = train_df["clean_text"], train_df["urgency"]
    X_test,  y_test  = test_df["clean_text"],  test_df["urgency"]

    print("Vectorising (TF-IDF)…")
    vectorizer    = TfidfVectorizer(max_features=5000, stop_words="english")
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf  = vectorizer.transform(X_test)

    print("Training urgency model…")
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)
    print("\nTest Accuracy :", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    with open(os.path.join(MODELS_DIR, "urgency_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(MODELS_DIR, "urgency_vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    print("\n✅  Urgency model saved to models/")


if __name__ == "__main__":
    main()
