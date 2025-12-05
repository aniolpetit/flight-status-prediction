"""
Train ML model to predict arrival delays (> 15 minutes).

This script uses pre-generated balanced train/test datasets
(from generate_datasets.py) and trains a tree-based model:

- If XGBoost with GPU is available -> use GPU (gpu_hist)
- Else if XGBoost CPU is available -> use CPU (hist)
- Else -> fallback to RandomForest

It also:
- Handles preprocessing (imputation, scaling, encoding)
- Computes metrics
- Trains a shallow surrogate tree for explainability
- Computes SHAP values (if shap is installed)
- Saves model, preprocessors, metrics, surrogate tree, and SHAP data.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
)
import joblib

# Pre-generated datasets from generate_datasets.py
DATA_DIR = Path("data")
TRAIN_PATH = DATA_DIR / "train_balanced.csv"
TEST_PATH = DATA_DIR / "test_real.csv"


def main():
    # -----------------------------
    # 1. Load and prepare dataset
    # -----------------------------
    print("Loading pre-generated train/test datasets...")
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError(
            "Train/test files not found. Run `python generate_datasets.py` first."
        )

    df_train_full = pd.read_csv(TRAIN_PATH)
    df_test_full = pd.read_csv(TEST_PATH)

    target_col = "IsArrDelayed"

    # Feature selection: ONLY pre-departure features available at booking/scheduling time
    # (no DepDelay, TaxiOut, AirTime, etc., to avoid data leakage)
    feature_cols_candidates = [
        # Temporal features
        "DayOfWeek",
        "DayofMonth",
        "DepHour",
        "Month",
        "Quarter",
        "Year",
        # Route/distance
        "Distance",
        "DistanceGroup",
        # Scheduled departure time features
        "CRSDepTime",
        "DepTimeBlk",
        # Categorical features
        "Airline",
        "Origin",
        "Dest",
        "DepTimeOfDay",
        "DayOfWeekName",
        "MonthName",
    ]

    # Keep only columns that actually exist in the data
    feature_cols = [c for c in feature_cols_candidates if c in df_train_full.columns]

    print(f"\nSelected {len(feature_cols)} features:")
    for i, col in enumerate(feature_cols, 1):
        print(f"  {i}. {col}")

    # Split train/test from prepared files (already stratified by generate_datasets)
    X_train = df_train_full[feature_cols].copy()
    y_train = df_train_full[target_col].astype(int)
    X_test = df_test_full[feature_cols].copy()
    y_test = df_test_full[target_col].astype(int)

    print(f"\nTrain size: {X_train.shape[0]:,} rows")
    print(f"Test size:  {X_test.shape[0]:,} rows")
    print(f"Delay rate in train: {y_train.mean():.3f}")

    # -----------------------------
    # 2. Preprocessing: missing values & encoding
    # -----------------------------
    # Detect categorical and numerical columns from X_train
    categorical_cols = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    # Double-check any column not classified yet
    for col in X_train.columns:
        if col not in categorical_cols and col not in numerical_cols:
            if X_train[col].dtype == "object" or isinstance(
                X_train[col].dtype, pd.CategoricalDtype
            ):
                categorical_cols.append(col)
            else:
                numerical_cols.append(col)

    print(f"\nCategorical features ({len(categorical_cols)}): {categorical_cols}")
    print(f"Numerical features ({len(numerical_cols)}): {numerical_cols}")

    # Initialize preprocessors
    imputer_num = None
    scaler = None
    label_encoders = {}

    # Numerical: impute + scale
    if len(numerical_cols) > 0:
        imputer_num = SimpleImputer(strategy="median")
        X_train[numerical_cols] = imputer_num.fit_transform(X_train[numerical_cols])
        X_test[numerical_cols] = imputer_num.transform(X_test[numerical_cols])

        scaler = StandardScaler()
        X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
        X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    # Categorical: LabelEncoder per column
    if len(categorical_cols) > 0:
        for col in categorical_cols:
            le = LabelEncoder()

            # Convert categorical dtype to string
            if isinstance(X_train[col].dtype, pd.CategoricalDtype):
                X_train[col] = X_train[col].astype(str)
                X_test[col] = X_test[col].astype(str)

            # Fill NaN with placeholder
            X_train[col] = X_train[col].fillna("Unknown").astype(str)
            X_test[col] = X_test[col].fillna("Unknown").astype(str)

            # Fit encoder on train
            X_train[col] = le.fit_transform(X_train[col])

            # Handle unknown categories in test
            test_values = X_test[col].astype(str)
            unknown_mask = ~test_values.isin(le.classes_)
            if unknown_mask.any():
                most_common = le.classes_[0]
                test_values[unknown_mask] = most_common
            X_test[col] = le.transform(test_values)

            label_encoders[col] = le

    # Ensure everything is numeric
    for col in X_train.columns:
        X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
        X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

    # Fill any remaining NaN
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    # Convert to numpy arrays
    X_train = X_train.values.astype(float)
    X_test = X_test.values.astype(float)

    # -----------------------------
    # 3. Build model (auto GPU detection)
    # -----------------------------
    print("\nDetecting available training hardware...")

    gpu_available = False
    xgb_available = False
    xgb = None

    # Try to import XGBoost
    try:
        import xgboost as xgb

        xgb_available = True
        # Try to create a dummy GPU-based model
        try:
            _ = xgb.XGBClassifier(tree_method="gpu_hist", predictor="gpu_predictor")
            gpu_available = True
            print("✓ XGBoost detected and GPU is available (CUDA).")
        except Exception:
            print("⚠ XGBoost detected but GPU not available. Will use CPU.")
            gpu_available = False
    except ImportError:
        print("⚠ XGBoost not installed. Will fallback to RandomForest.")
        xgb_available = False
        gpu_available = False

    # Choose model based on availability
    if gpu_available and xgb_available:
        print("Using XGBoost with GPU acceleration...")
        scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
            tree_method="gpu_hist",
            predictor="gpu_predictor",
            scale_pos_weight=scale_pos_weight,
        )
    elif xgb_available:
        print("Using XGBoost on CPU...")
        scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
            tree_method="hist",
            predictor="cpu_predictor",
            scale_pos_weight=scale_pos_weight,
        )
    else:
        print("Using RandomForest (fallback model)...")
        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=42,
            class_weight="balanced_subsample",
        )

    # -----------------------------
    # 4. Train
    # -----------------------------
    print("\nTraining model...")
    model.fit(X_train, y_train)

    # -----------------------------
    # 5. Evaluate
    # -----------------------------
    print("\nEvaluating model on test set...")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.3f}")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=3))

    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"ROC-AUC: {roc_auc:.3f}")

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(cm)

    # -----------------------------
    # 6. Extra metrics & explainability helpers
    # -----------------------------
    from sklearn.metrics import roc_curve
    from sklearn.tree import DecisionTreeClassifier

    # ROC curve data
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)

    # Feature importances
    feature_importances = None
    if hasattr(model, "feature_importances_"):
        feature_importances = model.feature_importances_

    # Shallow surrogate tree for explanation
    print("\nTraining shallow decision tree for explainability...")
    surrogate_tree = DecisionTreeClassifier(
        max_depth=3,
        random_state=42,
        class_weight="balanced",
    )
    surrogate_tree.fit(X_train, y_train)

    # -----------------------------
    # 7. SHAP values (if shap installed)
    # -----------------------------
    print("\nCalculating SHAP values for explainability...")
    shap_available = False
    shap_data = None

    try:
        import shap

        # Use TreeExplainer (works for XGBoost and RandomForest)
        explainer = shap.TreeExplainer(model)

        # Sample up to 100 test samples for SHAP
        n_shap_samples = min(100, len(X_test))
        X_test_sample = X_test[:n_shap_samples]
        y_test_sample = y_test[:n_shap_samples]

        print(f"Computing SHAP values for {n_shap_samples} test samples...")
        shap_values = explainer.shap_values(X_test_sample, check_additivity=False)

        # For binary classification, shap_values may be [class0, class1]
        if isinstance(shap_values, list):
            shap_values_delayed = np.array(shap_values[1])
            if isinstance(explainer.expected_value, (list, np.ndarray)) and len(
                explainer.expected_value
            ) > 1:
                expected_value_delayed = explainer.expected_value[1]
            else:
                expected_value_delayed = explainer.expected_value
        else:
            shap_values_delayed = np.array(shap_values)
            expected_value_delayed = explainer.expected_value

        # Build SHAP data dict
        shap_data = {
            "explainer": explainer,
            "shap_values": shap_values_delayed,  # (n_samples, n_features)
            "shap_values_all_classes": shap_values,
            "X_test_sample": X_test_sample,
            "y_test_sample": y_test_sample,
            "expected_value": expected_value_delayed,
            "feature_names": feature_cols,
        }

        shap_available = True
        print("✓ SHAP values computed successfully.")
    except ImportError:
        print("⚠ SHAP library not available. Install with: pip install shap")
        shap_available = False
        shap_data = None
    except Exception as e:
        print(f"⚠ Error computing SHAP values: {e}")
        shap_available = False
        shap_data = None

    # -----------------------------
    # 8. Save model, preprocessors & metrics
    # -----------------------------
    os.makedirs("models", exist_ok=True)

    # Model
    model_path = os.path.join("models", "arrival_delay_model.pkl")
    joblib.dump(model, model_path)

    # Preprocessors
    preprocessor_path = os.path.join("models", "preprocessors.pkl")
    preprocessors = {
        "imputer_num": imputer_num,
        "label_encoders": label_encoders,
        "scaler": scaler,
        "categorical_cols": categorical_cols,
        "numerical_cols": numerical_cols,
        "feature_cols": feature_cols,
    }
    joblib.dump(preprocessors, preprocessor_path)

    # Metrics & curves
    metrics_path = os.path.join("models", "metrics.pkl")
    metrics = {
        "accuracy": acc,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "feature_importances": feature_importances,
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True
        ),
        "class_names": ["On-Time", "Delayed"],
        "n_train": len(X_train),
        "n_test": len(X_test),
        "delay_rate_train": y_train.mean(),
        "delay_rate_test": y_test.mean(),
    }
    joblib.dump(metrics, metrics_path)

    # Surrogate tree
    surrogate_path = os.path.join("models", "surrogate_tree.pkl")
    joblib.dump(surrogate_tree, surrogate_path)

    # SHAP data
    if shap_available and shap_data is not None:
        shap_path = os.path.join("models", "shap_data.pkl")
        joblib.dump(shap_data, shap_path)
        print(f"SHAP data saved to: {shap_path}")

    print(f"\nModel saved to: {model_path}")
    print(f"Preprocessors saved to: {preprocessor_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Surrogate tree saved to: {surrogate_path}")


if __name__ == "__main__":
    main()
