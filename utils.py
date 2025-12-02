"""
Utility functions for data loading, processing, and aggregations
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import random
from io import StringIO
import streamlit as st


@st.cache_data
def load_flight_data(max_rows_per_file=100000, random_state=42):
    """
    Load flight data from CSV files using reservoir sampling
    
    Parameters:
    -----------
    max_rows_per_file : int
        Maximum rows to sample per CSV file
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    pd.DataFrame
        Combined flight data
    """
    
    def reservoir_sample_csv(file_path, k, random_state=42):
        """Memory-efficient random sampling using reservoir sampling"""
        random.seed(random_state)
        sample_lines = []
        with open(file_path, 'r', encoding='utf-8') as f:
            header = f.readline()
            for i, line in enumerate(f):
                if i < k:
                    sample_lines.append(line)
                else:
                    j = random.randint(0, i)
                    if j < k:
                        sample_lines[j] = line
        csv_buffer = StringIO(header + ''.join(sample_lines))
        return pd.read_csv(csv_buffer)
    
    data_dir = 'data'
    dfs = []
    
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            file_path = os.path.join(data_dir, filename)
            df_temp = reservoir_sample_csv(file_path, max_rows_per_file, random_state)
            dfs.append(df_temp)
    
    df = pd.concat(dfs, ignore_index=True)
    
    # Basic preprocessing
    df['FlightDate'] = pd.to_datetime(df['FlightDate'])
    
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
        Raw flight data
        
    Returns:
    --------
    pd.DataFrame
        Data with additional features
    """
    df = df.copy()
    
    # Temporal features
    df['Year'] = df['FlightDate'].dt.year
    df['Month'] = df['FlightDate'].dt.month
    df['MonthName'] = df['FlightDate'].dt.month_name()
    df['DayOfWeek'] = df['FlightDate'].dt.dayofweek
    df['DayOfWeekName'] = df['FlightDate'].dt.day_name()
    df['WeekOfYear'] = df['FlightDate'].dt.isocalendar().week.astype(int)
    
    # Hour of departure
    df['DepHour'] = (df['CRSDepTime'] // 100).astype(int)
    
    # Time of day categories
    df['DepTimeOfDay'] = pd.cut(
        df['DepHour'],
        bins=[0, 6, 12, 18, 24],
        labels=['Night (0-6)', 'Morning (6-12)', 'Afternoon (12-18)', 'Evening (18-24)'],
        include_lowest=True
    )
    
    # Delay categories
    df['ArrDelayCategory'] = df['ArrDelayMinutes'].apply(categorize_delay)
    df['DepDelayCategory'] = df['DepDelayMinutes'].apply(categorize_delay)
    
    # Delay flags
    df['IsArrDelayed'] = (df['ArrDelayMinutes'] > 15).astype(int)
    df['IsDepDelayed'] = (df['DepDelayMinutes'] > 15).astype(int)
    
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

