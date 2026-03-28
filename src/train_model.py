"""
train_model.py
───────────────
Trains a TF-IDF + Logistic Regression category classifier.

Reads  : data/processed/train_clean.csv  |  data/processed/test_clean.csv
Saves  : models/email_classifier.pkl     |  models/vectorizer.pkl

Usage:
    python src/train_model.py
"""

import os
import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import FeatureUnion

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(BASE_DIR, "data", "processed", "train_clean.csv")
TEST_PATH  = os.path.join(BASE_DIR, "data", "processed", "test_clean.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def main() -> None:
    # ── Load ──────────────────────────────────────────────────────────────────
    print("Loading processed data…")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df  = pd.read_csv(TEST_PATH)

    X_train, y_train = train_df["clean_text"], train_df["category"]
    X_test,  y_test  = test_df["clean_text"],  test_df["category"]

    # ── Vectorise ─────────────────────────────────────────────────────────────
    print("Vectorising text (word + char TF-IDF)…")
    vectorizer = FeatureUnion([
        (
            "word",
            TfidfVectorizer(
                analyzer="word",
                max_features=30000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.98,
                sublinear_tf=True,
            ),
        ),
        (
            "char",
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=2,
                max_features=20000,
                sublinear_tf=True,
            ),
        ),
    ])
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # ── Train ─────────────────────────────────────────────────────────────────
    print("Training Logistic Regression with CV hyperparameter search…")
    base_model = LogisticRegression(
        max_iter=4000,
        solver="saga",
        random_state=42,
    )

    param_grid = {
        "C": [1.0, 2.0, 3.5, 5.0],
        "class_weight": [None, "balanced"],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train_tfidf, y_train)
    model = search.best_estimator_

    print("Best parameters:", search.best_params_)
    print("Best CV macro-F1:", round(search.best_score_, 4))

    train_pred = model.predict(X_train_tfidf)
    print("Training Accuracy:", accuracy_score(y_train, train_pred))
    print("Training Macro F1:", round(f1_score(y_train, train_pred, average="macro"), 4))

    # ── Evaluate ──────────────────────────────────────────────────────────────
    y_pred = model.predict(X_test_tfidf)
    print("Test Accuracy    :", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    cv_scores = cross_val_score(model, X_train_tfidf, y_train, cv=cv, n_jobs=-1)
    print("\nCross-Validation Scores :", cv_scores)
    print("Mean CV Score           :", round(cv_scores.mean(), 4))

    # ── Save ──────────────────────────────────────────────────────────────────
    with open(os.path.join(MODELS_DIR, "email_classifier.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(MODELS_DIR, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    print("\n✅  Category model saved to models/")


if __name__ == "__main__":
    main()
