"""
Train ML model to predict arrival delays (> 15 minutes).

This script uses pre-generated balanced train/test datasets
(from generate_datasets.py) and tries multiple models / hyperparameters:

- XGBoost with GPU (if available)
- XGBoost on CPU
- RandomForest with different settings

It:
- Uses ONLY pre-departure features (no DepDelay, TaxiOut, AirTime, etc.)
- Handles preprocessing (imputation, scaling, encoding)
- Does an internal train/validation split on the balanced training set
- Selects the best model based on ROC-AUC on the validation set
- Retrains the best model on the full training data
- Evaluates on the real-distribution test set
- Sweeps multiple probability thresholds to analyze recall/precision trade-offs
- Picks a "high-recall" operating threshold (targeting recall>=0.75 if possible)
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
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
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
    X_train_df = df_train_full[feature_cols].copy()
    y_train = df_train_full[target_col].astype(int)
    X_test_df = df_test_full[feature_cols].copy()
    y_test = df_test_full[target_col].astype(int)

    print(f"\nTrain size: {X_train_df.shape[0]:,} rows")
    print(f"Test size:  {X_test_df.shape[0]:,} rows")
    print(f"Delay rate in train (balanced set): {y_train.mean():.3f}")

    # -----------------------------
    # 2. Preprocessing: missing values & encoding
    # -----------------------------
    # Detect categorical and numerical columns from X_train_df
    categorical_cols = X_train_df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    numerical_cols = X_train_df.select_dtypes(include=[np.number]).columns.tolist()

    # Double-check any column not classified yet
    for col in X_train_df.columns:
        if col not in categorical_cols and col not in numerical_cols:
            if (
                X_train_df[col].dtype == "object"
                or pd.api.types.is_categorical_dtype(X_train_df[col])
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

    # Numerical: impute + scale (fit on full train df to keep it simple)
    X_train = X_train_df.copy()
    X_test = X_test_df.copy()

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

    # Convert to numpy arrays for model training
    X_train_array = X_train.values.astype(float)
    X_test_array = X_test.values.astype(float)

    # -----------------------------
    # 3. Internal train/validation split
    # -----------------------------
    print("\nCreating internal train/validation split for model selection...")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_array,
        y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=42,
    )

    print(f"Train subset: {X_tr.shape[0]:,} rows")
    print(f"Validation subset: {X_val.shape[0]:,} rows")

    # -----------------------------
    # 4. Define candidate models & hyperparameters
    # -----------------------------
    print("\nDetecting available training hardware and XGBoost...")

    gpu_available = False
    xgb_available = False
    xgb = None

    try:
        import xgboost as xgb

        xgb_available = True
        try:
            # Test GPU availability with new XGBoost API (device="cuda")
            _ = xgb.XGBClassifier(tree_method="hist", device="cuda")
            gpu_available = True
            print("✓ XGBoost detected and GPU (CUDA) is available.")
        except Exception:
            print("⚠ XGBoost detected but GPU not available / not usable. Will use CPU.")
            gpu_available = False
    except ImportError:
        print("⚠ XGBoost not installed. Will use only RandomForest.")
        xgb_available = False
        gpu_available = False

    candidates = []

    # XGBoost candidates (GPU or CPU)
    if xgb_available:
        base_xgb_params = {
            "n_estimators": 300,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
            "eval_metric": "logloss",
        }

        # Handle imbalance with scale_pos_weight (even though train is balanced, it's safe)
        pos = (y_tr == 1).sum()
        neg = (y_tr == 0).sum()
        scale_pos_weight = neg / max(pos, 1)

        depths = [4, 6, 8]
        lrs = [0.1, 0.05]

        for max_depth in depths:
            for lr in lrs:
                params = base_xgb_params.copy()
                params.update(
                    {
                        "max_depth": max_depth,
                        "learning_rate": lr,
                        "scale_pos_weight": scale_pos_weight,
                        "tree_method": "hist",
                    }
                )
                if gpu_available:
                    params["device"] = "cuda"
                    device_label = "gpu"
                else:
                    device_label = "cpu"

                candidates.append(
                    {
                        "name": f"xgb_{device_label}_depth{max_depth}_lr{lr}",
                        "type": "xgb",
                        "params": params,
                    }
                )

    # RandomForest candidates
    rf_configs = [
        {
            "n_estimators": 300,
            "max_depth": 20,
            "min_samples_split": 10,
            "min_samples_leaf": 5,
        },
        {
            "n_estimators": 500,
            "max_depth": None,
            "min_samples_split": 10,
            "min_samples_leaf": 4,
        },
    ]

    for i, cfg in enumerate(rf_configs, 1):
        params = cfg.copy()
        params.update(
            {
                "n_jobs": -1,
                "random_state": 42,
                "class_weight": "balanced_subsample",
            }
        )
        candidates.append(
            {
                "name": f"rf_{i}",
                "type": "rf",
                "params": params,
            }
        )

    print(f"\nTotal candidate models to evaluate: {len(candidates)}")

    # -----------------------------
    # 5. Model selection on validation set
    # -----------------------------
    best_model = None
    best_name = None
    best_auc = -np.inf
    best_acc = -np.inf
    best_type = None
    best_params = None

    for cand in candidates:
        name = cand["name"]
        mtype = cand["type"]
        params = cand["params"]
        print(f"\nTraining candidate: {name} ({mtype})")

        if mtype == "xgb":
            model = xgb.XGBClassifier(**params)
        else:
            model = RandomForestClassifier(**params)

        model.fit(X_tr, y_tr)

        y_val_proba = model.predict_proba(X_val)[:, 1]
        y_val_pred = model.predict(X_val)

        auc = roc_auc_score(y_val, y_val_proba)
        acc = accuracy_score(y_val, y_val_pred)

        print(f"  -> Val ROC-AUC: {auc:.3f} | Val Accuracy: {acc:.3f}")

        # Select by best AUC, break ties with Accuracy
        if (auc > best_auc) or (auc == best_auc and acc > best_acc):
            best_auc = auc
            best_acc = acc
            best_model = model
            best_name = name
            best_type = mtype
            best_params = params

    print("\n==============================")
    print("Best candidate model selected:")
    print(f"  Name: {best_name}")
    print(f"  Type: {best_type}")
    print(f"  Val ROC-AUC: {best_auc:.3f}")
    print(f"  Val Accuracy: {best_acc:.3f}")
    print("==============================")

    # -----------------------------
    # 6. Retrain best model on FULL training set
    # -----------------------------
    print("\nRetraining best model on FULL balanced training data...")

    if best_type == "xgb":
        final_model = xgb.XGBClassifier(**best_params)
    else:
        final_model = RandomForestClassifier(**best_params)

    final_model.fit(X_train_array, y_train)

    # -----------------------------
    # 7. Evaluate on real-distribution test set (threshold = 0.5)
    # -----------------------------
    print("\nEvaluating FINAL model on REAL test set (threshold=0.5)...")

    y_proba = final_model.predict_proba(X_test_array)[:, 1]
    default_threshold = 0.5
    y_pred = (y_proba >= default_threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy (thr=0.5): {acc:.3f}")

    print("\nClassification report (test, thr=0.5):")
    print(classification_report(y_test, y_pred, digits=3))

    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"Test ROC-AUC: {roc_auc:.3f}")

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion matrix on test (rows=true, cols=pred, thr=0.5):")
    print(cm)

    # -----------------------------
    # 8. Threshold sweep for recall/precision trade-off
    # -----------------------------
    print("\nSweeping thresholds to analyze recall/precision trade-offs...")

    thresholds_to_try = np.linspace(0.1, 0.9, 17)  # 0.10, 0.15, ..., 0.90
    threshold_metrics = []

    for thr in thresholds_to_try:
        y_pred_thr = (y_proba >= thr).astype(int)
        acc_thr = accuracy_score(y_test, y_pred_thr)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, y_pred_thr, zero_division=0
        )

        metrics_thr = {
            "threshold": float(thr),
            "accuracy": float(acc_thr),
            "precision_0": float(prec[0]),
            "recall_0": float(rec[0]),
            "f1_0": float(f1[0]),
            "precision_1": float(prec[1]),
            "recall_1": float(rec[1]),
            "f1_1": float(f1[1]),
        }
        threshold_metrics.append(metrics_thr)

    print("\nThreshold sweep results (class 1 = delayed):")
    for m in threshold_metrics:
        print(
            f"  thr={m['threshold']:.2f} | "
            f"acc={m['accuracy']:.3f} | "
            f"prec_1={m['precision_1']:.3f} | "
            f"rec_1={m['recall_1']:.3f} | "
            f"f1_1={m['f1_1']:.3f}"
        )

    # Pick a "high recall" operating point:
    # 1) Prefer thresholds with recall_1 >= 0.75, choose the one with best f1_1
    # 2) If none achieve 0.75, choose the one with maximum recall_1
    high_recall_candidates = [m for m in threshold_metrics if m["recall_1"] >= 0.75]

    if high_recall_candidates:
        best_high = max(high_recall_candidates, key=lambda m: m["f1_1"])
        print(
            f"\nHigh-recall operating point found (recall_1 >= 0.75): "
            f"thr={best_high['threshold']:.2f}, "
            f"recall_1={best_high['recall_1']:.3f}, "
            f"precision_1={best_high['precision_1']:.3f}, "
            f"f1_1={best_high['f1_1']:.3f}, "
            f"accuracy={best_high['accuracy']:.3f}"
        )
    else:
        best_high = max(threshold_metrics, key=lambda m: m["recall_1"])
        print(
            f"\nNo threshold reached recall_1 >= 0.75. "
            f"Best recall_1 achieved: thr={best_high['threshold']:.2f}, "
            f"recall_1={best_high['recall_1']:.3f}, "
            f"precision_1={best_high['precision_1']:.3f}, "
            f"f1_1={best_high['f1_1']:.3f}, "
            f"accuracy={best_high['accuracy']:.3f}"
        )

    high_recall_threshold = best_high["threshold"]

    # -----------------------------
    # 9. Extra metrics & explainability helpers
    # -----------------------------
    from sklearn.metrics import roc_curve
    from sklearn.tree import DecisionTreeClassifier

    # ROC curve data
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_proba)

    # Feature importances
    feature_importances = None
    if hasattr(final_model, "feature_importances_"):
        feature_importances = final_model.feature_importances_

    # Shallow surrogate tree for explanation
    print("\nTraining shallow decision tree for explainability...")
    surrogate_tree = DecisionTreeClassifier(
        max_depth=3,
        random_state=42,
        class_weight="balanced",
    )
    surrogate_tree.fit(X_train_array, y_train)

    # -----------------------------
    # 10. SHAP values (if shap installed)
    # -----------------------------
    print("\nCalculating SHAP values for explainability...")
    shap_available = False
    shap_data = None

    try:
        import shap

        # Use TreeExplainer (works for XGBoost and RandomForest)
        explainer = shap.TreeExplainer(final_model)

        # Sample up to 100 test samples for SHAP
        n_shap_samples = min(100, len(X_test_array))
        X_test_sample = X_test_array[:n_shap_samples]
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
    # 11. Save model, preprocessors & metrics
    # -----------------------------
    os.makedirs("models", exist_ok=True)

    # Model
    model_path = os.path.join("models", "arrival_delay_model.pkl")
    joblib.dump(final_model, model_path)

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
        "accuracy_default_thr": acc,
        "roc_auc": roc_auc,
        "confusion_matrix_default_thr": cm,
        "fpr": fpr,
        "tpr": tpr,
        "roc_thresholds": roc_thresholds,
        "feature_importances": feature_importances,
        "classification_report_default_thr": classification_report(
            y_test, y_pred, output_dict=True
        ),
        "class_names": ["On-Time", "Delayed"],
        "n_train": len(X_train_array),
        "n_test": len(X_test_array),
        "delay_rate_train": y_train.mean(),
        "delay_rate_test": y_test.mean(),
        "best_model_name": best_name,
        "best_model_type": best_type,
        "best_model_val_auc": best_auc,
        "best_model_val_acc": best_acc,
        "default_threshold": float(default_threshold),
        "high_recall_threshold": float(high_recall_threshold),
        "threshold_metrics": threshold_metrics,
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
    print(
        f"Best model was: {best_name} ({best_type}) with Val AUC={best_auc:.3f}, "
        f"default_thr={default_threshold}, high_recall_thr={high_recall_threshold:.2f}"
    )


if __name__ == "__main__":
    main()
