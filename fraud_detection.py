"""
Educational baseline for credit-card fraud detection.
Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
Place creditcard.csv in the project root after downloading it from Kaggle.
This script reports held-out metrics; it does not claim results until run locally.
"""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, average_precision_score,
    roc_auc_score
)

DATA_PATH = Path("creditcard.csv")
TARGET = "Class"

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "creditcard.csv not found. Download it from "
            "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud "
            "and place it beside fraud_detection.py."
        )

    df = pd.read_csv(DATA_PATH)
    if TARGET not in df.columns:
        raise ValueError("Expected target column 'Class' in the CSV.")
    df = df.drop_duplicates()
    df = df.dropna(subset=[TARGET])

    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)
    if set(y.unique()) - {0, 1}:
        raise ValueError("Target 'Class' must contain only 0 and 1.")

    # Stratification preserves the rare fraud proportion in each split.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    numeric_features = list(X.columns)
    preprocessor = ColumnTransformer(
        [("scale", StandardScaler(), numeric_features)],
        remainder="drop"
    )
    model = Pipeline([
        ("preprocess", preprocessor),
        ("classifier", LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        ))
    ])
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print(f"Rows used after duplicate removal: {len(df):,}")
    print(f"Fraud rate: {y.mean():.4%}")
    print("\nConfusion matrix [ [TN, FP], [FN, TP] ]:")
    print(confusion_matrix(y_test, pred))
    print("\nClassification report:")
    print(classification_report(y_test, pred, digits=4, zero_division=0))
    print(f"Average precision (PR-oriented summary): "
          f"{average_precision_score(y_test, probabilities):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.4f}")
    print("\nNote: tune the decision threshold on a validation set before deployment.")

if __name__ == "__main__":
    main()
