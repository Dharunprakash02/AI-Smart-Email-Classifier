"""
evaluate_model.py
──────────────────
Shared evaluation utilities (accuracy, classification report, confusion matrix).
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def evaluate_model(model, X_test, y_test):
    """Print metrics and return predictions."""
    y_pred = model.predict(X_test)
    print("Test Accuracy :", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))
    return y_pred


def plot_confusion_matrix(y_test, y_pred, title: str = "Confusion Matrix") -> None:
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.show()
