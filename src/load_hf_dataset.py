"""
load_hf_dataset.py
───────────────────
Helper to pull the email classification dataset from HuggingFace Hub.
"""

from datasets import load_dataset


def load_hf_data():
    return load_dataset("jason23322/high-accuracy-email-classifier")


if __name__ == "__main__":
    ds = load_hf_data()
    print("Dataset loaded successfully.")
    print("Train samples :", len(ds["train"]))
    print("Test  samples :", len(ds["test"]))
    print("\nSample record :", ds["train"][0])
