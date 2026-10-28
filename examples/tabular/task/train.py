"""Fit a classifier on the Adult training split and save it to model.joblib.

This is the file to modify to improve the validation accuracy reported by eval.py.
The saved model must accept the raw feature DataFrame returned by
load_train_data() in predict().
"""

import joblib
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eval import MODEL_PATH, load_train_data


def build_model():
    # Baseline: logistic regression on the numeric columns only, categorical
    # columns are dropped.
    preprocessing = make_column_transformer(
        (
            make_pipeline(SimpleImputer(), StandardScaler()),
            make_column_selector(dtype_include="number"),
        ),
    )
    return make_pipeline(preprocessing, LogisticRegression(max_iter=1000))


def main():
    X_train, y_train = load_train_data()
    model = build_model()
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)
    print(f"saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
