"""
Model Explainability Page - SHAP-based explanations
for the RandomForest arrival delay model.
"""

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import shap
import matplotlib.pyplot as plt

from utils import get_model_dataset  # función que definimos antes en utils.py


st.title("🔍 Model Explainability")
st.markdown(
    "Understand **which factors drive the model's predictions** for arrival delays "
    "(> 15 minutes)."
)

st.markdown("---")


# -----------------------------
# Load model & data
# -----------------------------
@st.cache_resource
def load_trained_model():
    model_path = "models/arrival_delay_model.pkl"
    clf = joblib.load(model_path)
    return clf


@st.cache_data
def load_shap_dataset(max_rows_per_file: int = 40_000, random_state: int = 42):
    """
    Load a clean dataset for SHAP analysis.
    We reuse the same feature set as in training.
    """
    df = get_model_dataset(
        max_rows_per_file=max_rows_per_file,
        random_state=random_state,
    )

    feature_cols_cat = [
        "Airline",
        "Origin",
        "Dest",
        "DepTimeOfDay",
        "DayOfWeekName",
        "MonthName",
    ]
    feature_cols_num = ["DepHour", "Distance"]

    X = df[feature_cols_cat + feature_cols_num]
    y = df["IsArrDelayed"].astype(int)

    # To keep SHAP computations reasonable, sample a subset
    if len(X) > 3000:
        X_sample = X.sample(3000, random_state=random_state)
    else:
        X_sample = X.copy()

    return X_sample, y.loc[X_sample.index], feature_cols_cat, feature_cols_num


with st.spinner("Loading model and data for explainability..."):
    clf = load_trained_model()
    X_sample, y_sample, feature_cols_cat, feature_cols_num = load_shap_dataset()


# -----------------------------
# Create SHAP explainer
# -----------------------------
@st.cache_resource
def create_explainer(model, X_background):
    """
    We create a SHAP Explainer using the model's predict_proba function.
    This works with a sklearn Pipeline (preprocessing + model).
    """
    explainer = shap.Explainer(model.predict_proba, X_background, feature_names=X_background.columns)
    return explainer


with st.spinner("Computing SHAP values (first time may take a bit)..."):
    explainer = create_explainer(clf, X_sample)
    shap_values = explainer(X_sample)  # shape: [n_samples, n_classes, n_features]


st.markdown("## 🌍 Global feature importance")

st.markdown(
    """
    These plots show **which features matter most overall** for predicting whether a flight
    will arrive more than 15 minutes late (class 1 = delayed).
    """
)

# class 1 (delayed) shap values
shap_delayed = shap_values[:, 1, :]

# --- Bar plot (mean |SHAP|)
fig_bar, ax_bar = plt.subplots(figsize=(8, 5))
shap.plots.bar(shap_delayed, show=False)
st.pyplot(fig_bar)

st.markdown("---")

# -----------------------------
# Local explanation
# -----------------------------
st.markdown("## ✈️ Local explanation for a specific flight")

st.markdown(
    "Select one of the sampled flights and see **which features pushed the prediction "
    "towards delay vs on-time**."
)

# choose an index
idx = st.slider(
    "Select sample index",
    min_value=0,
    max_value=len(X_sample) - 1,
    value=0,
    step=1,
)

x_row = X_sample.iloc[idx : idx + 1]
y_row = y_sample.iloc[idx]
shap_row = shap_delayed[idx]

# Show raw features
st.markdown("### Flight features")
st.dataframe(x_row)

# Prediction + probas for this row
proba = float(clf.predict_proba(x_row)[0, 1])
pred_label = int(clf.predict(x_row)[0])

st.markdown(
    f"""
    **Model output for this flight:**
    - Predicted probability of delay > 15 min: **{proba * 100:.1f}%**
    - Predicted class: **{"Delayed" if pred_label == 1 else "Not delayed"}**
    - True label in data: **{"Delayed" if y_row == 1 else "Not delayed"}**
    """
)

st.markdown("### 🔍 SHAP waterfall plot (local contribution of each feature)")

fig_wf, ax_wf = plt.subplots(figsize=(8, 5))
shap.plots.waterfall(shap_row, max_display=10, show=False)
st.pyplot(fig_wf)

st.markdown(
    """
    **How to read this plot:**
    - The grey baseline on the left is the model's average output (log-odds) for the dataset.  
    - Red bars push the prediction **towards higher delay probability**,  
      blue bars push it **towards lower delay probability**.  
    - The length of each bar shows **how strong the contribution** of that feature is.
    """
)

st.markdown("---")

st.info(
    """
    In your report and presentation, you can highlight:
    - Which features are globally most important (airline, route, time of day, etc.).
    - For individual flights, which conditions (airport, departure time, weekday...) drove
      the model to predict a high or low delay risk.
    """
)
