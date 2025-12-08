"""
Utility functions for data loading, processing, and aggregations
"""

import pandas as pd
import numpy as np
from pathlib import Path


SAMPLED_CSV_NAME = "sampled_flights_2018_2022.csv"


# --------------------------------------------------
# Load sampled data
# --------------------------------------------------
def load_flight_data(max_rows_per_file: int = 100000, random_state: int = 42):
    """
    Load flight data from the pre-computed sampled CSV.
    """
    data_dir = Path("data")
    sampled_path = data_dir / SAMPLED_CSV_NAME

    if not sampled_path.exists():
        raise FileNotFoundError(
            f"{sampled_path} not found.\n"
            "Run `python create_sampled_dataset.py` once from the project root "
            "to generate the sampled CSV."
        )

    df = pd.read_csv(sampled_path, low_memory=False)

    # Convert FlightDate if present
    if "FlightDate" in df.columns:
        df["FlightDate"] = pd.to_datetime(df["FlightDate"], errors="coerce")

    return df


# --------------------------------------------------
# Delay categorization helper
# --------------------------------------------------
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


# --------------------------------------------------
# Feature engineering
# --------------------------------------------------
def prepare_flight_features(df):
    """
    Add derived features to flight data.
    Safe for both sampled dataset AND raw 2018–2022 CSV chunks.
    """
    df = df.copy()

    # Convert date safely (CRITICAL FIX)
    if "FlightDate" in df.columns:
        df["FlightDate"] = pd.to_datetime(df["FlightDate"], errors="coerce")

    # Remove rows with invalid dates
    df = df[df["FlightDate"].notna()].copy()

    # ---- Temporal features ----
    df['Year'] = df['FlightDate'].dt.year
    df['Month'] = df['FlightDate'].dt.month
    df['MonthName'] = df['FlightDate'].dt.month_name()
    df['DayOfWeek'] = df['FlightDate'].dt.dayofweek
    df['DayOfWeekName'] = df['FlightDate'].dt.day_name()
    df['WeekOfYear'] = df['FlightDate'].dt.isocalendar().week.astype(int)

    # ---- Hour of departure ----
    if "CRSDepTime" in df.columns:
        df['DepHour'] = (df['CRSDepTime'] // 100).astype(int)
    else:
        df['DepHour'] = np.nan

    # Time-of-day categories
    df['DepTimeOfDay'] = pd.cut(
        df['DepHour'],
        bins=[0, 6, 12, 18, 24],
        labels=['Night (0-6)', 'Morning (6-12)', 'Afternoon (12-18)', 'Evening (18-24)'],
        include_lowest=True
    )

    # ---- Delay categories ----
    if "ArrDelayMinutes" in df.columns:
        df['ArrDelayCategory'] = df['ArrDelayMinutes'].apply(categorize_delay)
        df['IsArrDelayed'] = (df['ArrDelayMinutes'] > 15).astype(int)
    else:
        df['ArrDelayCategory'] = 'Unknown'
        df['IsArrDelayed'] = np.nan

    if "DepDelayMinutes" in df.columns:
        df['DepDelayCategory'] = df['DepDelayMinutes'].apply(categorize_delay)
        df['IsDepDelayed'] = (df['DepDelayMinutes'] > 15).astype(int)
    else:
        df['DepDelayCategory'] = 'Unknown'
        df['IsDepDelayed'] = np.nan

    return df


# --------------------------------------------------
# Daily aggregation
# --------------------------------------------------
def aggregate_daily_delays(df):
    """Aggregate delays at daily level"""
    daily = df.groupby('FlightDate').agg(
        avg_arr_delay=('ArrDelayMinutes', 'mean'),
        avg_dep_delay=('DepDelayMinutes', 'mean'),
        median_arr_delay=('ArrDelayMinutes', 'median'),
        median_dep_delay=('DepDelayMinutes', 'median'),
        arr_delay_rate=('IsArrDelayed', 'mean'),
        dep_delay_rate=('IsDepDelayed', 'mean'),
        cancel_rate=('Cancelled', 'mean') if "Cancelled" in df.columns else ('IsArrDelayed', 'mean'),
        divert_rate=('Diverted', 'mean') if "Diverted" in df.columns else ('IsArrDelayed', 'mean'),
        total_flights=('IsArrDelayed', 'size')
    ).reset_index()

    # Temporal features
    daily['Year'] = daily['FlightDate'].dt.year
    daily['Month'] = daily['FlightDate'].dt.month
    daily['MonthName'] = daily['FlightDate'].dt.month_name()
    daily['DayOfWeek'] = daily['FlightDate'].dt.dayofweek
    daily['DayOfWeekName'] = daily['FlightDate'].dt.day_name()
    daily['WeekOfYear'] = daily['FlightDate'].dt.isocalendar().week.astype(int)

    return daily


# --------------------------------------------------
# Hour × Weekday aggregation
# --------------------------------------------------
def aggregate_hour_weekday(df):
    """Aggregate delays by hour of day and weekday"""

    agg = df.groupby(['DayOfWeekName', 'DepHour']).agg(
        avg_arr_delay=('ArrDelayMinutes', 'mean'),
        avg_dep_delay=('DepDelayMinutes', 'mean'),
        arr_delay_rate=('IsArrDelayed', 'mean'),
        dep_delay_rate=('IsDepDelayed', 'mean'),
        total_flights=('IsArrDelayed', 'size')
    ).reset_index()

    # Proper weekday sorting
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    agg['DayOfWeekName'] = pd.Categorical(agg['DayOfWeekName'], categories=weekday_order, ordered=True)
    agg = agg.sort_values(['DayOfWeekName', 'DepHour'])

    return agg


# --------------------------------------------------
# Airline aggregation
# --------------------------------------------------
def aggregate_by_airline(df, min_flights=1000):
    """Aggregate delays by airline"""
    agg = df.groupby('Airline').agg(
        avg_arr_delay=('ArrDelayMinutes', 'mean'),
        median_arr_delay=('ArrDelayMinutes', 'median'),
        avg_dep_delay=('DepDelayMinutes', 'mean'),
        median_dep_delay=('DepDelayMinutes', 'median'),
        arr_delay_rate=('IsArrDelayed', 'mean'),
        dep_delay_rate=('IsDepDelayed', 'mean'),
        cancel_rate=('Cancelled', 'mean'),
        total_flights=('IsArrDelayed', 'size')
    ).reset_index()

    agg = agg[agg['total_flights'] >= min_flights]
    return agg.sort_values('avg_arr_delay', ascending=False)


# --------------------------------------------------
# Route aggregation
# --------------------------------------------------
def aggregate_by_route(df, min_flights=100):
    """Aggregate delays by route"""
    agg = df.groupby(['Origin', 'Dest']).agg(
        avg_arr_delay=('ArrDelayMinutes', 'mean'),
        median_arr_delay=('ArrDelayMinutes', 'median'),
        arr_delay_rate=('IsArrDelayed', 'mean'),
        total_flights=('IsArrDelayed', 'size')
    ).reset_index()

    # Add cancellation rate when available; fallback to 0 to avoid plot errors
    if "Cancelled" in df.columns:
        cancel_rate = df.groupby(['Origin', 'Dest'])['Cancelled'].mean().reset_index(name='cancel_rate')
        agg = agg.merge(cancel_rate, on=['Origin', 'Dest'], how='left')
    else:
        agg['cancel_rate'] = 0.0

    agg = agg[agg['total_flights'] >= min_flights]
    agg['Route'] = agg['Origin'] + ' → ' + agg['Dest']
    return agg.sort_values('avg_arr_delay', ascending=False)


# --------------------------------------------------
# Delay distribution helper
# --------------------------------------------------
def get_delay_category_distribution(df, group_by_col):
    """Compute percentage distribution of delay categories per group"""
    category_order = [
        'Early', 'On Time', 'Small Delay', 'Moderate Delay',
        'Large Delay', 'Very Large Delay', 'Unknown'
    ]

    cat_counts = (
        df.groupby([group_by_col, 'ArrDelayCategory'])
        .size()
        .reset_index(name='count')
    )

    totals = cat_counts.groupby(group_by_col)['count'].transform('sum')
    cat_counts['percentage'] = cat_counts['count'] / totals * 100

    pivot = cat_counts.pivot(index=group_by_col, columns='ArrDelayCategory', values='percentage')
    pivot = pivot[[c for c in category_order if c in pivot.columns]]

    return pivot.fillna(0)


# --------------------------------------------------
# Formatting helper
# --------------------------------------------------
def format_metric(value, metric_type='percentage', decimals=1):
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


# --------------------------------------------------
# Build model dataset
# --------------------------------------------------
def get_model_dataset():
    """
    Build the clean dataset for ML training.
    """
    df_raw = load_flight_data()
    df = prepare_flight_features(df_raw)

    # Remove cancelled/diverted
    if "Cancelled" in df.columns:
        df = df[df["Cancelled"] == 0]
    if "Diverted" in df.columns:
        df = df[df["Diverted"] == 0]

    # Ensure target exists
    if "IsArrDelayed" not in df.columns:
        df["IsArrDelayed"] = (df["ArrDelayMinutes"] > 15).astype(int)

    # Pre-departure features only
    preferred_cols = [
        "Airline", "Origin", "Dest",
        "OriginAirportID", "DestAirportID",
        "Distance", "DistanceGroup",
        "FlightDate",
        "Year", "Month", "MonthName",
        "DayOfWeek", "DayOfWeekName",
        "DayofMonth", "Quarter",
        "DepHour", "DepTimeOfDay",
        "CRSDepTime", "DepTimeBlk",
        "Flights"
    ]

    target = "IsArrDelayed"
    keep_cols = [c for c in preferred_cols if c in df.columns] + [target]
    df = df[keep_cols].copy()

    # Drop rows missing key identifiers
    df = df.dropna(subset=["Airline", "Origin", "Dest", target])

    return df
