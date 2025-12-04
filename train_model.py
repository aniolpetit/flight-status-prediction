"""
Train ML model to predict arrival delays (> 15 minutes).
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib

from utils import get_model_dataset


def main():
    # -----------------------------
    # 1. Load and prepare dataset
    # -----------------------------
    print("Loading and preparing data...")
    df = get_model_dataset(max_rows_per_file=100_000, random_state=42)

    target_col = "IsArrDelayed"

    feature_cols_cat = [
        "Airline",
        "Origin",
        "Dest",
        "DepTimeOfDay",
        "DayOfWeekName",
        "MonthName",
    ]

    feature_cols_num = [
        "DepHour",
        "Distance",
    ]

    X = df[feature_cols_cat + feature_cols_num]
    y = df[target_col].astype(int)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"Train size: {X_train.shape[0]:,} rows")
    print(f"Test size:  {X_test.shape[0]:,} rows")
    print(f"Delay rate in train: {y_train.mean():.3f}")

    # -----------------------------
    # 2. Build preprocessing + model pipeline
    # -----------------------------
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=True
    )

    numeric_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, feature_cols_cat),
            ("num", numeric_transformer, feature_cols_num),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced"  # handle class imbalance
    )

    clf = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model),
    ])

    # -----------------------------
    # 3. Train
    # -----------------------------
    print("Training model...")
    clf.fit(X_train, y_train)

    # -----------------------------
    # 4. Evaluate
    # -----------------------------
    print("\nEvaluating model on test set...")

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=3))

    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"ROC-AUC: {roc_auc:.3f}")

    # -----------------------------
    # 5. Save trained pipeline
    # -----------------------------
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "arrival_delay_model.pkl")
    joblib.dump(clf, model_path)

    print(f"\nModel saved to: {model_path}")


if __name__ == "__main__":
    main()
