"""
Train ML model to predict arrival delays (> 15 minutes).

This script replicates the high-performance approach from the reference notebook,
using post-departure features (DepDelay, TaxiOut, AirTime) that are highly predictive.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
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
    # We exclude post-departure features (DepDelay, TaxiOut, AirTime) to avoid data leakage
    # Instead, we'll use richer feature engineering and a stronger model to compensate
    feature_cols_candidates = [
        # Temporal features (rich calendar encoding)
        "DayOfWeek", "DayofMonth", "DepHour",
        "Month", "Quarter", "Year",
        
        # Route/distance
        "Distance", "DistanceGroup",
        
        # Scheduled departure time features
        "CRSDepTime", "DepTimeBlk",
        
        # Categorical features (will be encoded)
        "Airline", "Origin", "Dest",
        "DepTimeOfDay", "DayOfWeekName", "MonthName",
    ]

    # Filter to only columns that exist in the dataframe
    feature_cols = [c for c in feature_cols_candidates if c in df_train_full.columns]
    
    print(f"\nSelected {len(feature_cols)} features:")
    for i, col in enumerate(feature_cols, 1):
        print(f"  {i}. {col}")

    # Split train/test from prepared files (already stratified by generate_datasets)
    X_train = df_train_full[feature_cols].copy()
    y_train = df_train_full[target_col].astype(int)
    X_test = df_test_full[feature_cols].copy()
    y_test = df_test_full[target_col].astype(int)

    print(f"Train size: {X_train.shape[0]:,} rows")
    print(f"Test size:  {X_test.shape[0]:,} rows")
    print(f"Delay rate in train: {y_train.mean():.3f}")

    # -----------------------------
    # 2. Preprocessing: Handle missing values and encode categoricals
    # -----------------------------
    # Separate categorical and numerical features
    # Check for object dtype and also category dtype (pandas categorical)
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    # Also check for any columns that might be strings but not detected
    for col in X.columns:
        if col not in categorical_cols and col not in numerical_cols:
            # Check if it's actually categorical
            if X[col].dtype == 'object' or (hasattr(X[col].dtype, 'categories')):
                categorical_cols.append(col)
            else:
                numerical_cols.append(col)
    
    print(f"\nCategorical features ({len(categorical_cols)}): {categorical_cols}")
    print(f"Numerical features ({len(numerical_cols)}): {numerical_cols}")
    
    # Initialize preprocessors (will be None if not needed)
    imputer_num = None
    scaler = None
    label_encoders = {}
    
    # Handle missing values: impute with median for numerical
    # (matching the reference notebook approach)
    if len(numerical_cols) > 0:
        imputer_num = SimpleImputer(strategy='median')
        X_train[numerical_cols] = imputer_num.fit_transform(X_train[numerical_cols])
        X_test[numerical_cols] = imputer_num.transform(X_test[numerical_cols])
        
        # Standardize numerical features
        scaler = StandardScaler()
        X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
        X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    # Use LabelEncoder for categoricals (like the reference notebook)
    if len(categorical_cols) > 0:
        for col in categorical_cols:
            le = LabelEncoder()
            
            # Handle pandas Categorical dtype: convert to object first
            if isinstance(X_train[col].dtype, pd.CategoricalDtype):
                X_train[col] = X_train[col].astype(str)
                X_test[col] = X_test[col].astype(str)
            
            # Fill NaN values with a placeholder before encoding
            X_train[col] = X_train[col].fillna('Unknown').astype(str)
            X_test[col] = X_test[col].fillna('Unknown').astype(str)
            
            # Fit on train, transform both train and test
            X_train[col] = le.fit_transform(X_train[col])
            # Handle unknown categories in test set
            test_values = X_test[col].astype(str)
            unknown_mask = ~test_values.isin(le.classes_)
            if unknown_mask.any():
                # Replace unknown with most common class
                most_common = le.classes_[0]
                test_values[unknown_mask] = most_common
            X_test[col] = le.transform(test_values)
            label_encoders[col] = le
    
    # Convert all columns to numeric and then to numpy arrays for sklearn
    # This ensures no string values remain
    for col in X_train.columns:
        X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
    
    # Fill any remaining NaN values (from conversion errors) with 0
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    
    # Convert to numpy arrays for sklearn compatibility
    X_train = X_train.values.astype(float)
    X_test = X_test.values.astype(float)

    # -----------------------------
    # 3. Build model (matching reference notebook parameters)
    # -----------------------------
    # Use a stronger model to compensate for not using post-departure features
    # XGBoost typically performs better than RandomForest on tabular data
    try:
        from xgboost import XGBClassifier
        print("Using XGBoost for better performance...")
        model = XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss',
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),  # handle imbalance
        )
    except ImportError:
        print("XGBoost not available, using RandomForest with stronger settings...")
        # Fallback to RandomForest with more trees and better tuning
        model = RandomForestClassifier(
            n_estimators=500,      # more trees to compensate
            max_depth=20,          # deeper trees
            min_samples_split=10,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=42,
            class_weight="balanced_subsample",  # handle class imbalance
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

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.3f}")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=3))

    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"ROC-AUC: {roc_auc:.3f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(cm)

    # -----------------------------
    # 6. Calculate additional metrics for visualization
    # -----------------------------
    from sklearn.metrics import roc_curve
    from sklearn.tree import DecisionTreeClassifier
    
    # Calculate ROC curve data
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    
    # Get feature importances (if available)
    feature_importances = None
    if hasattr(model, 'feature_importances_'):
        feature_importances = model.feature_importances_
    
    # Train a shallow decision tree as a surrogate model for visualization
    print("\nTraining shallow decision tree for explainability...")
    surrogate_tree = DecisionTreeClassifier(
        max_depth=3,  # shallow for visualization
        random_state=42,
        class_weight="balanced"
    )
    surrogate_tree.fit(X_train, y_train)
    
    # Calculate SHAP values for explainability
    print("\nCalculating SHAP values for explainability...")
    try:
        import shap
        
        # Use TreeExplainer for tree-based models (much faster than KernelExplainer)
        explainer = shap.TreeExplainer(model)
        
        # Sample test data for SHAP calculation (to save memory and time)
        # Use up to 100 samples for detailed explanations
        n_shap_samples = min(100, len(X_test))
        X_test_sample = X_test[:n_shap_samples]
        y_test_sample = y_test[:n_shap_samples]
        
        print(f"Computing SHAP values for {n_shap_samples} test samples...")
        shap_values = explainer.shap_values(X_test_sample, check_additivity=False)
        
        # For binary classification, shap_values is a list [class0, class1]
        # We want the values for class 1 (delayed)
        if isinstance(shap_values, list):
            shap_values_delayed = np.array(shap_values[1])  # Ensure numpy array
            expected_value_delayed = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) and len(explainer.expected_value) > 1 else explainer.expected_value
        else:
            shap_values_delayed = np.array(shap_values)  # Ensure numpy array
            expected_value_delayed = explainer.expected_value
        
        # Ensure y_test_sample is a pandas Series or array
        if isinstance(y_test, pd.Series):
            y_test_sample = y_test.iloc[:n_shap_samples]
        else:
            y_test_sample = y_test[:n_shap_samples]
        
        # Ensure X_test_sample is numpy array
        if not isinstance(X_test_sample, np.ndarray):
            X_test_sample = np.array(X_test_sample)
        
        # Validate shapes
        if shap_values_delayed.shape[0] != X_test_sample.shape[0]:
            print(f"⚠ Warning: SHAP values shape {shap_values_delayed.shape} doesn't match X_test_sample shape {X_test_sample.shape}")
        
        if shap_values_delayed.shape[1] != len(feature_cols):
            print(f"⚠ Warning: SHAP values feature count {shap_values_delayed.shape[1]} doesn't match feature count {len(feature_cols)}")
        
        shap_data = {
            'explainer': explainer,
            'shap_values': shap_values_delayed,  # Shape: (n_samples, n_features)
            'shap_values_all_classes': shap_values,  # Keep both classes for flexibility
            'X_test_sample': X_test_sample,  # Shape: (n_samples, n_features)
            'y_test_sample': y_test_sample,
            'expected_value': expected_value_delayed,
            'feature_names': feature_cols,  # List of length n_features
        }
        
        print(f"✓ SHAP values computed successfully")
        shap_available = True
        
    except ImportError:
        print("⚠ SHAP library not available. Install with: pip install shap")
        shap_data = None
        shap_available = False
    except Exception as e:
        print(f"⚠ Error computing SHAP values: {e}")
        shap_data = None
        shap_available = False
    
    # -----------------------------
    # 7. Save trained model, preprocessors, and metrics
    # -----------------------------
    os.makedirs("models", exist_ok=True)
    
    # Save model
    model_path = os.path.join("models", "arrival_delay_model.pkl")
    joblib.dump(model, model_path)
    
    # Save preprocessors for inference
    preprocessor_path = os.path.join("models", "preprocessors.pkl")
    preprocessors = {
        'imputer_num': imputer_num,
        'label_encoders': label_encoders,
        'scaler': scaler,
        'categorical_cols': categorical_cols,
        'numerical_cols': numerical_cols,
        'feature_cols': feature_cols,
    }
    joblib.dump(preprocessors, preprocessor_path)
    
    # Save metrics and additional data for visualization
    metrics_path = os.path.join("models", "metrics.pkl")
    metrics = {
        'accuracy': acc,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'feature_importances': feature_importances,
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'class_names': ['On-Time', 'Delayed'],
        'n_train': len(X_train),
        'n_test': len(X_test),
        'delay_rate_train': y_train.mean(),
        'delay_rate_test': y_test.mean(),
    }
    joblib.dump(metrics, metrics_path)
    
    # Save surrogate tree
    surrogate_path = os.path.join("models", "surrogate_tree.pkl")
    joblib.dump(surrogate_tree, surrogate_path)
    
    # Save SHAP data if available
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
