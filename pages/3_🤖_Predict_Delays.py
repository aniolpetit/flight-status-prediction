"""
Predict Delays Page - Uses trained RandomForest model
to estimate probability of arrival delay > 15 minutes.
"""

import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st
import joblib

from utils import load_flight_data, prepare_flight_features

# OJO: set_page_config solo en app.py, no aquí en multipage apps

st.title("🤖 Flight Delay Prediction")
st.markdown(
    "Use the trained machine learning model to estimate the probability that a flight "
    "will arrive **more than 15 minutes late**."
)

st.markdown("---")


# -----------------------------
# Helpers: load model & metadata
# -----------------------------
@st.cache_resource
def load_trained_model():
    """Load the trained sklearn Pipeline from disk."""
    model_path = "models/arrival_delay_model.pkl"
    clf = joblib.load(model_path)
    return clf


@st.cache_data
def load_metadata(sample_rows_per_file: int = 50_000, random_state: int = 42):
    """
    Load a sample of data to:
      - populate dropdowns (airlines, airports)
      - estimate typical distance per route
    """
    df_raw = load_flight_data(
        max_rows_per_file=sample_rows_per_file,
        random_state=random_state,
    )
    df = prepare_flight_features(df_raw)
    df = df[(df["Cancelled"] == False) & (df["Diverted"] == False)]

    airlines = sorted(df["Airline"].dropna().unique().tolist())
    origins = sorted(df["Origin"].dropna().unique().tolist())
    dests = sorted(df["Dest"].dropna().unique().tolist())

    # median distance per (Origin, Dest)
    route_distance = (
        df.groupby(["Origin", "Dest"])["Distance"]
        .median()
        .reset_index()
        .rename(columns={"Distance": "MedianDistance"})
    )

    return df, airlines, origins, dests, route_distance


def get_dep_time_of_day(hour: int) -> str:
    """Replicate the same time-of-day bins used in utils.prepare_flight_features."""
    if 0 <= hour < 6:
        return "Night (0-6)"
    elif 6 <= hour < 12:
        return "Morning (6-12)"
    elif 12 <= hour < 18:
        return "Afternoon (12-18)"
    else:
        return "Evening (18-24)"


# -----------------------------
# Load resources
# -----------------------------
with st.spinner("Loading model and metadata..."):
    clf = load_trained_model()
    df_meta, airlines, origins, dests, route_distance = load_metadata()


# -----------------------------
# Layout
# -----------------------------
left_col, right_col = st.columns([1.3, 1])

with left_col:
    st.subheader("✏️ Flight details")

    # 1) Date
    flight_date = st.date_input(
        "Flight date",
        value=dt.date(2022, 4, 4),
        min_value=dt.date(2018, 1, 1),
        max_value=dt.date(2022, 12, 31),
    )
    day_of_week_name = flight_date.strftime("%A")
    month_name = flight_date.strftime("%B")

    # 2) Airline
    airline = st.selectbox("Airline", options=airlines)

    # 3) Origin & destination
    origin = st.selectbox("Origin airport (IATA)", options=origins, index=origins.index("DEN") if "DEN" in origins else 0)
    dest = st.selectbox("Destination airport (IATA)", options=dests, index=dests.index("LAX") if "LAX" in dests else 0)

    # 4) Departure time
    dep_time = st.time_input("Scheduled departure time", value=dt.time(12, 0))
    dep_hour = dep_time.hour
    dep_time_of_day = get_dep_time_of_day(dep_hour)

    # 5) Distance (auto from route, editable)
    #    Try to look up median distance for this route
    median_distance = (
        route_distance.loc[
            (route_distance["Origin"] == origin) & (route_distance["Dest"] == dest),
            "MedianDistance",
        ]
        .dropna()
        .astype(float)
    )

    if not median_distance.empty:
        default_distance = float(median_distance.iloc[0])
    else:
        # fallback: typical US domestic distance
        default_distance = 1000.0

    distance = st.number_input(
        "Route distance (miles)",
        min_value=50.0,
        max_value=5000.0,
        value=round(default_distance, 1),
        step=10.0,
        help="Approximate great-circle distance between origin and destination.",
    )

    st.markdown("")


with right_col:
    st.subheader("📌 Prediction")

    st.markdown(
        "The model predicts whether the flight will arrive **more than 15 minutes late** "
        "(binary classification: delayed vs not delayed)."
    )

    predict_button = st.button("🚀 Predict delay risk")

    result_placeholder = st.empty()
    explanation_placeholder = st.empty()


# -----------------------------
# Build feature row & predict
# -----------------------------
if predict_button:
    # The model was trained with these columns:
    feature_cols_cat = [
        "Airline",
        "Origin",
        "Dest",
        "DepTimeOfDay",
        "DayOfWeekName",
        "MonthName",
    ]
    feature_cols_num = ["DepHour", "Distance"]

    # Build a single-row DataFrame
    X_input = pd.DataFrame(
        {
            "Airline": [airline],
            "Origin": [origin],
            "Dest": [dest],
            "DepTimeOfDay": [dep_time_of_day],
            "DayOfWeekName": [day_of_week_name],
            "MonthName": [month_name],
            "DepHour": [dep_hour],
            "Distance": [distance],
        }
    )

    with st.spinner("Running model prediction..."):
        proba_delay = float(clf.predict_proba(X_input)[0, 1])
        pred_label = int(clf.predict(X_input)[0])

    # Interpret prediction
    risk_pct = proba_delay * 100

    if risk_pct < 15:
        risk_level = "Low"
        risk_color = "🟢"
    elif risk_pct < 35:
        risk_level = "Medium"
        risk_color = "🟡"
    else:
        risk_level = "High"
        risk_color = "🔴"

    result_placeholder.markdown(
        f"""
        ### {risk_color} Predicted delay risk: **{risk_pct:.1f}%**

        - **Prediction (class):** {"Delayed (>15 min)" if pred_label == 1 else "Not significantly delayed"}
        - **Risk level:** **{risk_level}**
        """
    )

    with explanation_placeholder:
        st.markdown("#### ℹ️ How to interpret this")
        st.markdown(
            """
            - The model has been trained on historical US domestic flights from 2018–2022.  
            - It predicts **whether arrival delay will exceed 15 minutes**.  
            - The probability reflects similar flights (same airline, airports, time of day, etc.).  
            - In the *Explainability* page you will see **which features** contribute most to this prediction.
            """
        )

else:
    st.info("👈 Fill in the flight details on the left and click **“Predict delay risk”**.")
