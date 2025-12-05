"""
Utility functions for data loading, processing, and aggregations
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import streamlit as st


SAMPLED_CSV_NAME = "sampled_flights_2018_2022.csv"


@st.cache_data
def load_flight_data(max_rows_per_file: int = 100000, random_state: int = 42):
    """
    Load flight data from the pre-computed sampled CSV.

    Notes
    -----
    - The heavy reservoir sampling is done once by `create_sampled_dataset.py`,
      which writes `data/sampled_flights_2018_2022.csv`.
    - This function simply reads that file and does basic preprocessing.
    - Parameters are kept for backward compatibility but are not used
      to re-sample from the raw yearly CSVs.
    """
    data_dir = Path("data")
    sampled_path = data_dir / SAMPLED_CSV_NAME

    if not sampled_path.exists():
        raise FileNotFoundError(
            f"{sampled_path} not found.\n"
            "Run `python create_sampled_dataset.py` once from the project root "
            "to generate the sampled CSV."
        )

    df = pd.read_csv(sampled_path)

    # Basic preprocessing
    if "FlightDate" in df.columns:
        df["FlightDate"] = pd.to_datetime(df["FlightDate"])

    return df


def categorize_delay(delay_minutes):
    """Categorize delays into meaningful bins"""
    if pd.isna(delay_minutes):
        return 'Unknown'
    elif delay_minutes < 0:
        return 'Early'
    elif delay_minutes == 0:
        return 'On Time'
    elif delay_minutes <= 15:
        return 'Small Delay'
    elif delay_minutes <= 60:
        return 'Moderate Delay'
    elif delay_minutes <= 180:
        return 'Large Delay'
    else:
        return 'Very Large Delay'


@st.cache_data
def prepare_flight_features(df):
    """
    Add derived features to flight data
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw flight data (can come from load_flight_data or raw CSVs)
        
    Returns:
    --------
    pd.DataFrame
        Data with additional features
    """
    df = df.copy()

    # --- 1) Asegurar tipos correctos en columnas clave ---

    # FlightDate -> datetime
    df["FlightDate"] = pd.to_datetime(df["FlightDate"], errors="coerce")
    df = df[df["FlightDate"].notna()].copy()

    # CRSDepTime -> numérico (HHMM) y sin NaN
    df["CRSDepTime"] = pd.to_numeric(df["CRSDepTime"], errors="coerce")
    df = df[df["CRSDepTime"].notna()].copy()

    # --- 2) Features temporales ---
    df["Year"] = df["FlightDate"].dt.year
    df["Month"] = df["FlightDate"].dt.month
    df["MonthName"] = df["FlightDate"].dt.month_name()
    df["DayOfWeek"] = df["FlightDate"].dt.dayofweek
    df["DayOfWeekName"] = df["FlightDate"].dt.day_name()
    df["WeekOfYear"] = df["FlightDate"].dt.isocalendar().week.astype(int)

    # --- 3) Hora de salida y tramo horario ---
    df["DepHour"] = (df["CRSDepTime"] // 100).astype(int)

    df["DepTimeOfDay"] = pd.cut(
        df["DepHour"],
        bins=[0, 6, 12, 18, 24],
        labels=["Night (0-6)", "Morning (6-12)", "Afternoon (12-18)", "Evening (18-24)"],
        include_lowest=True,
    )

    # --- 4) Categorías de delay ---
    df["ArrDelayCategory"] = df["ArrDelayMinutes"].apply(categorize_delay)
    df["DepDelayCategory"] = df["DepDelayMinutes"].apply(categorize_delay)

    # --- 5) Flags binarios de delay (> 15 min) ---
    df["IsArrDelayed"] = (df["ArrDelayMinutes"] > 15).astype(int)
    df["IsDepDelayed"] = (df["DepDelayMinutes"] > 15).astype(int)

    return df


@st.cache_data
def aggregate_daily_delays(df):
    """
    Aggregate delays at daily level
    
    Parameters:
    -----------
    df : pd.DataFrame
        Flight data with features
        
    Returns:
    --------
    pd.DataFrame
        Daily aggregated metrics
    """
    daily = df.groupby('FlightDate').agg(
        avg_arr_delay=('ArrDelayMinutes', 'mean'),
        avg_dep_delay=('DepDelayMinutes', 'mean'),
        median_arr_delay=('ArrDelayMinutes', 'median'),
        median_dep_delay=('DepDelayMinutes', 'median'),
        arr_delay_rate=('IsArrDelayed', 'mean'),
        dep_delay_rate=('IsDepDelayed', 'mean'),
        cancel_rate=('Cancelled', 'mean'),
        divert_rate=('Diverted', 'mean'),
        total_flights=('Cancelled', 'size')
    ).reset_index()
    
    # Add temporal features
    daily['Year'] = daily['FlightDate'].dt.year
    daily['Month'] = daily['FlightDate'].dt.month
    daily['MonthName'] = daily['FlightDate'].dt.month_name()
    daily['DayOfWeek'] = daily['FlightDate'].dt.dayofweek
    daily['DayOfWeekName'] = daily['FlightDate'].dt.day_name()
    daily['WeekOfYear'] = daily['FlightDate'].dt.isocalendar().week.astype(int)
    
    return daily


@st.cache_data
def aggregate_hour_weekday(df):
    """
    Aggregate delays by hour of day and day of week
    
    Parameters:
    -----------
    df : pd.DataFrame
        Flight data with features
        
    Returns:
    --------
    pd.DataFrame
        Hour × Weekday aggregated metrics
    """
    agg = df.groupby(['DayOfWeekName', 'DepHour']).agg(
        avg_arr_delay=('ArrDelayMinutes', 'mean'),
        avg_dep_delay=('DepDelayMinutes', 'mean'),
        arr_delay_rate=('IsArrDelayed', 'mean'),
        dep_delay_rate=('IsDepDelayed', 'mean'),
        cancel_rate=('Cancelled', 'mean'),
        total_flights=('Cancelled', 'size')
    ).reset_index()
    
    # Ensure proper weekday ordering
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    agg['DayOfWeekName'] = pd.Categorical(agg['DayOfWeekName'], categories=weekday_order, ordered=True)
    agg = agg.sort_values(['DayOfWeekName', 'DepHour'])
    
    return agg


@st.cache_data
def aggregate_by_airline(df, min_flights=1000):
    """
    Aggregate delays by airline
    
    Parameters:
    -----------
    df : pd.DataFrame
        Flight data with features
    min_flights : int
        Minimum flights for an airline to be included
        
    Returns:
    --------
    pd.DataFrame
        Airline aggregated metrics
    """
    agg = df.groupby('Airline').agg(
        avg_arr_delay=('ArrDelayMinutes', 'mean'),
        median_arr_delay=('ArrDelayMinutes', 'median'),
        avg_dep_delay=('DepDelayMinutes', 'mean'),
        median_dep_delay=('DepDelayMinutes', 'median'),
        arr_delay_rate=('IsArrDelayed', 'mean'),
        dep_delay_rate=('IsDepDelayed', 'mean'),
        cancel_rate=('Cancelled', 'mean'),
        divert_rate=('Diverted', 'mean'),
        total_flights=('Cancelled', 'size')
    ).reset_index()
    
    # Filter by minimum flights
    agg = agg[agg['total_flights'] >= min_flights]
    
    return agg.sort_values('avg_arr_delay', ascending=False)


@st.cache_data
def aggregate_by_route(df, min_flights=100):
    """
    Aggregate delays by route (origin-destination pair)
    
    Parameters:
    -----------
    df : pd.DataFrame
        Flight data with features
    min_flights : int
        Minimum flights for a route to be included
        
    Returns:
    --------
    pd.DataFrame
        Route aggregated metrics
    """
    agg = df.groupby(['Origin', 'Dest']).agg(
        avg_arr_delay=('ArrDelayMinutes', 'mean'),
        median_arr_delay=('ArrDelayMinutes', 'median'),
        arr_delay_rate=('IsArrDelayed', 'mean'),
        cancel_rate=('Cancelled', 'mean'),
        total_flights=('Cancelled', 'size')
    ).reset_index()
    
    # Filter by minimum flights
    agg = agg[agg['total_flights'] >= min_flights]
    
    # Create route label
    agg['Route'] = agg['Origin'] + ' → ' + agg['Dest']
    
    return agg.sort_values('avg_arr_delay', ascending=False)


def get_delay_category_distribution(df, group_by_col):
    """
    Get delay category distribution for a given grouping
    
    Parameters:
    -----------
    df : pd.DataFrame
        Flight data with ArrDelayCategory
    group_by_col : str
        Column to group by
        
    Returns:
    --------
    pd.DataFrame
        Pivot table with delay category percentages
    """
    category_order = ['Early', 'On Time', 'Small Delay', 'Moderate Delay', 
                      'Large Delay', 'Very Large Delay', 'Unknown']
    
    cat_counts = (
        df.groupby([group_by_col, 'ArrDelayCategory'])
        .size()
        .reset_index(name='count')
    )
    
    # Calculate percentages
    totals = cat_counts.groupby(group_by_col)['count'].transform('sum')
    cat_counts['percentage'] = cat_counts['count'] / totals * 100
    
    # Pivot
    pivot = cat_counts.pivot(index=group_by_col, columns='ArrDelayCategory', values='percentage')
    pivot = pivot[[c for c in category_order if c in pivot.columns]]
    
    return pivot.fillna(0)


def format_metric(value, metric_type='percentage', decimals=1):
    """Format metrics for display"""
    if pd.isna(value):
        return "N/A"
    
    if metric_type == 'percentage':
        return f"{value:.{decimals}f}%"
    elif metric_type == 'minutes':
        return f"{value:.{decimals}f} min"
    elif metric_type == 'count':
        return f"{int(value):,}"
    else:
        return f"{value:.{decimals}f}"


@st.cache_data
def get_model_dataset():
    """
    Load data, engineer features and return a clean dataset for ML modelling.

    Design choices
    --------------
    - Target: binary classification of **significant arrival delay** (>15 minutes),
      using the `IsArrDelayed` flag created in `prepare_flight_features`.
    - Scope: we only keep **non-cancelled and non-diverted** flights, because those
      represent a different prediction problem (handled separately if needed).
    - Features: we keep a curated set of features that are:
        * Available at or before scheduled departure time
        * Non-leaky (no columns derived from arrival delay, cancellation reasons,
          or post-arrival information)
        * Useful for modelling based on our EDA (airline, route, time of day,
          day of week, month/season, distance).

    The returned dataframe is ready to be split into X/y by downstream code
    (see `train_model.py`), with all required columns present and nulls removed.
    """
    # 1) Load sampled data from precomputed CSV
    df_raw = load_flight_data()

    # 2) Add engineered features (temporal features, delay flags, etc.)
    df = prepare_flight_features(df_raw)

    # 3) Keep only flights with valid arrival delay info and non-cancelled/diverted
    #    (for this model we focus purely on "delay vs on-time")
    if "Cancelled" in df.columns:
        df = df[df["Cancelled"] == 0]
    if "Diverted" in df.columns:
        df = df[df["Diverted"] == 0]

    # Also ensure the target is defined
    if "IsArrDelayed" not in df.columns:
        # Fallback: compute from ArrDelayMinutes if needed
        if "ArrDelayMinutes" not in df.columns:
            raise ValueError(
                "Neither 'IsArrDelayed' nor 'ArrDelayMinutes' is available in the dataframe."
            )
        df["IsArrDelayed"] = (df["ArrDelayMinutes"] > 15).astype(int)

    # 4) Curate feature set: ONLY pre-departure features available at booking/scheduling time
    #    We exclude post-departure features (DepDelay, TaxiOut, AirTime) to avoid data leakage
    #    and ensure the model can be used for real-world prediction scenarios.
    preferred_feature_cols = [
        # Airline / route identifiers
        "Airline",                 # main carrier identifier used across the project
        "Origin", "Dest",          # IATA airport codes (human-readable, good for SHAP)
        "OriginAirportID", "DestAirportID",  # stable numeric IDs if present
        "Distance", "DistanceGroup",

        # Temporal / schedule information (all available before departure)
        "FlightDate",
        "Year", "Month", "MonthName",
        "DayOfWeek", "DayOfWeekName",
        "DayofMonth", "Quarter",
        "DepHour", "DepTimeOfDay",
        "CRSDepTime", "DepTimeBlk",

        # Volume / context (can be useful for some models or analysis)
        "Flights",
    ]

    # Ensure target column is always preserved
    target_col = "IsArrDelayed"

    keep_cols = [col for col in preferred_feature_cols if col in df.columns]
    keep_cols.append(target_col)

    df = df[keep_cols].copy()

    # 5) Drop rows with any missing values in critical columns.
    #    We require core identifiers and target, but allow some missing values in
    #    other features which will be imputed during training.
    required_for_model = [
        # Core identifiers (must be present)
        "Airline", "Origin", "Dest",
        target_col,
    ]
    required_existing = [c for c in required_for_model if c in df.columns]
    df = df.dropna(subset=required_existing)

    return df