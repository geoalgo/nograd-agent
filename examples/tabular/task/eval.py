"""Fixed evaluation harness for the Adult (census income) classification task.

This file owns the data splits and the metric and must stay unchanged so that
scores of different versions of train.py are comparable. train.py only gets
access to the training split through load_train_data(), and prints the scores
returned by evaluate(). The validation split is used to compare candidates
during tuning, the test split only to report the final performance.
"""

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

SPLIT_SEED = 0
TEST_SIZE = 0.2  # fraction of all rows
VALID_SIZE = 0.2  # fraction of all rows
SPLITS = ("valid", "test")


def _load_splits() -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    # Downloaded once from OpenML, then cached in ~/scikit_learn_data.
    X, y = fetch_openml("adult", version=2, as_frame=True, return_X_y=True)
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SPLIT_SEED, stratify=y
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_rest,
        y_rest,
        test_size=VALID_SIZE / (1 - TEST_SIZE),
        random_state=SPLIT_SEED,
        stratify=y_rest,
    )
    return {
        "train": (X_train, y_train),
        "valid": (X_valid, y_valid),
        "test": (X_test, y_test),
    }


def load_train_data() -> tuple[pd.DataFrame, pd.Series]:
    """Return the training features and labels (">50K" or "<=50K")."""
    return _load_splits()["train"]


def evaluate(model) -> dict[str, float]:
    """Return the accuracy of a fitted model on the validation and test splits,
    its predict() is called on the raw feature DataFrame."""
    splits = _load_splits()
    return {
        f"{split}_accuracy": float((model.predict(X) == y).mean())
        for split, (X, y) in ((split, splits[split]) for split in SPLITS)
    }

