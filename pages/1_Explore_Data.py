"""
Explore Data Page - Interactive Filtering and Visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
sys.path.append('..')
from utils import (
    load_flight_data, 
    prepare_flight_features,
    aggregate_daily_delays,
    aggregate_hour_weekday,
    aggregate_by_airline,
    aggregate_by_route,
    format_metric
)

# Page config
st.set_page_config(page_title="Explore Data", page_icon="📊", layout="wide")

st.title("📊 Explore Flight Data")
st.markdown("**Interactive exploration** with customizable filters and visualizations")

# Load data
with st.spinner("Loading flight data..."):
    df = load_flight_data(max_rows_per_file=100000)
    df = prepare_flight_features(df)

st.success(f"✅ Loaded {len(df):,} flight records from 2018-2022")

# Sidebar filters
st.sidebar.header("🔍 Filters")

# Date range filter
date_min = df['FlightDate'].min().date()
date_max = df['FlightDate'].max().date()
date_range = st.sidebar.date_input(
    "Date Range",
    value=(date_min, date_max),
    min_value=date_min,
    max_value=date_max
)

# Convert to datetime for comparison
if len(date_range) == 2:
    date_start, date_end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    df_filtered = df[(df['FlightDate'] >= date_start) & (df['FlightDate'] <= date_end)]
else:
    df_filtered = df.copy()

# Airline filter
all_airlines = sorted(df['Airline'].unique())
selected_airlines = st.sidebar.multiselect(
    "Airlines",
    options=all_airlines,
    default=None,
    help="Leave empty to select all airlines"
)

if selected_airlines:
    df_filtered = df_filtered[df_filtered['Airline'].isin(selected_airlines)]

# Day of week filter
all_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
selected_days = st.sidebar.multiselect(
    "Day of Week",
    options=all_days,
    default=None,
    help="Leave empty to select all days"
)

if selected_days:
    df_filtered = df_filtered[df_filtered['DayOfWeekName'].isin(selected_days)]

# Time of day filter
selected_time_of_day = st.sidebar.multiselect(
    "Time of Day",
    options=['Night (0-6)', 'Morning (6-12)', 'Afternoon (12-18)', 'Evening (18-24)'],
    default=None,
    help="Leave empty to select all times"
)

if selected_time_of_day:
    df_filtered = df_filtered[df_filtered['DepTimeOfDay'].isin(selected_time_of_day)]

# Distance filter
if 'Distance' in df_filtered.columns:
    distance_min = int(df_filtered['Distance'].min())
    distance_max = int(df_filtered['Distance'].max())
    distance_range = st.sidebar.slider(
        "Flight Distance (miles)",
        min_value=distance_min,
        max_value=distance_max,
        value=(distance_min, distance_max)
    )
    df_filtered = df_filtered[
        (df_filtered['Distance'] >= distance_range[0]) & 
        (df_filtered['Distance'] <= distance_range[1])
    ]

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Filtered Data**: {len(df_filtered):,} flights ({len(df_filtered)/len(df)*100:.1f}%)")

# Main content
st.markdown("---")

# Key metrics
st.subheader("📈 Overview Metrics")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    avg_arr_delay = df_filtered['ArrDelayMinutes'].mean()
    st.metric(
        "Avg Arrival Delay",
        f"{avg_arr_delay:.1f} min",
        delta=None
    )

with col2:
    delay_rate = (df_filtered['IsArrDelayed'].mean() * 100)
    st.metric(
        "Delay Rate (>15 min)",
        f"{delay_rate:.1f}%"
    )

with col3:
    cancel_rate = (df_filtered['Cancelled'].mean() * 100)
    st.metric(
        "Cancellation Rate",
        f"{cancel_rate:.2f}%"
    )

with col4:
    divert_rate = (df_filtered['Diverted'].mean() * 100)
    st.metric(
        "Diversion Rate",
        f"{divert_rate:.2f}%"
    )

with col5:
    on_time_rate = ((df_filtered['ArrDelayMinutes'] <= 15).sum() / len(df_filtered) * 100)
    st.metric(
        "On-Time Rate (≤15 min)",
        f"{on_time_rate:.1f}%"
    )

st.markdown("---")

# Visualization selector
st.subheader("📊 Interactive Visualizations")

viz_type = st.selectbox(
    "Select Visualization Type",
    [
        "Delay Distribution",
        "Time Series Analysis",
        "Hour × Weekday Heatmap",
        "Airline Performance",
        "Route Analysis",
        "Delay Categories",
        "Correlations"
    ]
)

st.markdown("---")

# Visualization 1: Delay Distribution
if viz_type == "Delay Distribution":
    st.markdown("### Distribution of Arrival Delays")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histogram
        fig = px.histogram(
            df_filtered[df_filtered['ArrDelayMinutes'].notna()],
            x='ArrDelayMinutes',
            nbins=100,
            title="Arrival Delay Distribution",
            labels={'ArrDelayMinutes': 'Arrival Delay (minutes)'},
            color_discrete_sequence=['#1f77b4']
        )
        fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="On Time")
        fig.add_vline(x=15, line_dash="dash", line_color="orange", annotation_text="Delay Threshold")
        fig.update_xaxes(range=[-50, 300])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Box plot
        fig = px.box(
            df_filtered[df_filtered['ArrDelayMinutes'].notna()],
            y='ArrDelayMinutes',
            title="Arrival Delay Box Plot",
            labels={'ArrDelayMinutes': 'Arrival Delay (minutes)'}
        )
        fig.update_yaxes(range=[-50, 200])
        st.plotly_chart(fig, use_container_width=True)
    
    # Percentiles
    st.markdown("#### Delay Percentiles")
    percentiles = df_filtered['ArrDelayMinutes'].quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    perc_df = pd.DataFrame({
        'Percentile': ['10%', '25%', '50%', '75%', '90%', '95%', '99%'],
        'Delay (minutes)': percentiles.values
    })
    st.dataframe(perc_df, use_container_width=True)

# Visualization 2: Time Series
elif viz_type == "Time Series Analysis":
    st.markdown("### Delay Trends Over Time")
    
    daily = aggregate_daily_delays(df_filtered)
    
    metric_choice = st.radio(
        "Select Metric",
        ["Average Delay", "Delay Rate", "Cancellation Rate"],
        horizontal=True
    )
    
    if metric_choice == "Average Delay":
        y_col = 'avg_arr_delay'
        y_label = 'Average Arrival Delay (minutes)'
    elif metric_choice == "Delay Rate":
        y_col = 'arr_delay_rate'
        y_label = 'Arrival Delay Rate (>15 min)'
    else:
        y_col = 'cancel_rate'
        y_label = 'Cancellation Rate'
    
    fig = px.line(
        daily,
        x='FlightDate',
        y=y_col,
        title=f"Daily {metric_choice}",
        labels={'FlightDate': 'Date', y_col: y_label}
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5)
    st.plotly_chart(fig, use_container_width=True)
    
    # Monthly aggregation
    st.markdown("#### Monthly Averages")
    monthly = daily.groupby('MonthName')[y_col].mean().reset_index()
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    monthly['MonthName'] = pd.Categorical(monthly['MonthName'], categories=month_order, ordered=True)
    monthly = monthly.sort_values('MonthName')
    
    fig = px.bar(
        monthly,
        x='MonthName',
        y=y_col,
        title=f"Average {metric_choice} by Month",
        labels={'MonthName': 'Month', y_col: y_label}
    )
    st.plotly_chart(fig, use_container_width=True)

# Visualization 3: Hour × Weekday Heatmap
elif viz_type == "Hour × Weekday Heatmap":
    st.markdown("### Delay Patterns by Hour and Weekday")
    
    hw_agg = aggregate_hour_weekday(df_filtered)
    
    metric_choice = st.radio(
        "Select Metric",
        ["Average Delay", "Delay Rate"],
        horizontal=True
    )
    
    if metric_choice == "Average Delay":
        value_col = 'avg_arr_delay'
        title = 'Average Arrival Delay (minutes)'
        colorscale = 'RdYlGn_r'
    else:
        value_col = 'arr_delay_rate'
        title = 'Arrival Delay Rate (>15 min)'
        colorscale = 'Reds'
    
    pivot = hw_agg.pivot(index='DayOfWeekName', columns='DepHour', values=value_col)
    
    fig = px.imshow(
        pivot,
        labels=dict(x="Departure Hour", y="Day of Week", color=title),
        x=pivot.columns,
        y=pivot.index,
        color_continuous_scale=colorscale,
        title=title + " by Hour and Weekday"
    )
    st.plotly_chart(fig, use_container_width=True)

# Visualization 4: Airline Performance
elif viz_type == "Airline Performance":
    st.markdown("### Airline Performance Comparison")
    
    airline_agg = aggregate_by_airline(df_filtered, min_flights=100)
    
    top_n = st.slider("Number of Airlines to Display", 5, 20, 15)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Average delay
        top_airlines = airline_agg.nlargest(top_n, 'avg_arr_delay')
        fig = px.bar(
            top_airlines,
            x='avg_arr_delay',
            y='Airline',
            orientation='h',
            title=f"Top {top_n} Airlines by Average Arrival Delay",
            labels={'avg_arr_delay': 'Average Arrival Delay (minutes)'},
            color='avg_arr_delay',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Cancellation rate
        top_cancel = airline_agg.nlargest(top_n, 'cancel_rate')
        fig = px.bar(
            top_cancel,
            x='cancel_rate',
            y='Airline',
            orientation='h',
            title=f"Top {top_n} Airlines by Cancellation Rate",
            labels={'cancel_rate': 'Cancellation Rate'},
            color='cancel_rate',
            color_continuous_scale='Oranges'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Scatter: flight volume vs delay
    st.markdown("#### Flight Volume vs Average Delay")
    fig = px.scatter(
        airline_agg,
        x='total_flights',
        y='avg_arr_delay',
        hover_data=['Airline'],
        title="Flight Volume vs Average Delay",
        labels={'total_flights': 'Number of Flights', 'avg_arr_delay': 'Average Arrival Delay (minutes)'},
        size='total_flights',
        color='avg_arr_delay',
        color_continuous_scale='RdYlGn_r'
    )
    st.plotly_chart(fig, use_container_width=True)

# Visualization 5: Route Analysis
elif viz_type == "Route Analysis":
    st.markdown("### Route Performance Analysis")
    
    route_agg = aggregate_by_route(df_filtered, min_flights=50)
    if 'cancel_rate' not in route_agg.columns:
        route_agg['cancel_rate'] = 0.0
    
    col1, col2 = st.columns(2)
    
    with col1:
        top_n = st.slider("Number of Routes to Display", 10, 50, 20)
    
    with col2:
        sort_by = st.radio(
            "Sort By",
            ["Average Delay", "Delay Rate", "Cancellation Rate"],
            horizontal=True
        )
    
    if sort_by == "Average Delay":
        sort_col = 'avg_arr_delay'
    elif sort_by == "Delay Rate":
        sort_col = 'arr_delay_rate'
    else:
        sort_col = 'cancel_rate'
    
    top_routes = route_agg.nlargest(top_n, sort_col)
    
    fig = px.bar(
        top_routes,
        x=sort_col,
        y='Route',
        orientation='h',
        title=f"Top {top_n} Routes by {sort_by}",
        color=sort_col,
        color_continuous_scale='Reds'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Volume vs Delay Rate analysis
    st.markdown("#### Flight Volume vs Delay Rate")
    st.markdown("Understanding the relationship between route popularity and delay performance")
    
    fig = px.scatter(
        route_agg,
        x='total_flights',
        y='arr_delay_rate',
        size='avg_arr_delay',
        hover_data=['Route'],
        title="Route Volume vs Delay Rate (bubble size = average delay)",
        labels={
            'total_flights': 'Number of Flights (Volume)',
            'arr_delay_rate': 'Delay Rate (>15 min)',
            'avg_arr_delay': 'Average Delay (min)'
        },
        color='arr_delay_rate',
        color_continuous_scale='Reds'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Highlight problematic routes (high volume + high delay rate)
    problematic = route_agg[
        (route_agg['total_flights'] > route_agg['total_flights'].quantile(0.75)) &
        (route_agg['arr_delay_rate'] > route_agg['arr_delay_rate'].quantile(0.75))
    ].nlargest(10, 'total_flights')
    
    if not problematic.empty:
        st.markdown("**⚠️ High-Volume Routes with High Delay Rates:**")
        st.dataframe(
            problematic[['Route', 'total_flights', 'arr_delay_rate', 'avg_arr_delay']].round(2),
            use_container_width=True,
            hide_index=True
        )

# Visualization 6: Delay Categories
elif viz_type == "Delay Categories":
    st.markdown("### Delay Category Distribution")
    
    # Add legend/definitions
    with st.expander("📖 Delay Category Definitions", expanded=False):
        st.markdown("""
        **Delay Categories:**
        - **Early**: Arrival delay < 0 minutes (arrived early)
        - **On Time**: Arrival delay = 0 minutes (exactly on time)
        - **Small Delay**: Arrival delay 1-15 minutes
        - **Moderate Delay**: Arrival delay 16-60 minutes
        - **Large Delay**: Arrival delay 61-180 minutes
        - **Very Large Delay**: Arrival delay > 180 minutes
        
        **Note**: Flights with delay ≤ 15 minutes (Early + On Time + Small Delay) are considered "on-time" for operational purposes.
        """)
    
    category_counts = df_filtered['ArrDelayCategory'].value_counts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart
        fig = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            title="Delay Category Distribution",
            hole=0.3
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Bar chart
        fig = px.bar(
            x=category_counts.index,
            y=category_counts.values,
            title="Delay Category Counts",
            labels={'x': 'Category', 'y': 'Number of Flights'},
            color=category_counts.values,
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # By time of day
    st.markdown("#### Delay Categories by Time of Day")
    
    tod_cat = df_filtered.groupby(['DepTimeOfDay', 'ArrDelayCategory']).size().reset_index(name='count')
    tod_totals = tod_cat.groupby('DepTimeOfDay')['count'].transform('sum')
    tod_cat['percentage'] = tod_cat['count'] / tod_totals * 100
    
    fig = px.bar(
        tod_cat,
        x='DepTimeOfDay',
        y='percentage',
        color='ArrDelayCategory',
        title="Delay Category Distribution by Time of Day",
        labels={'percentage': 'Percentage of Flights (%)', 'DepTimeOfDay': 'Time of Day'}
    )
    st.plotly_chart(fig, use_container_width=True)

# Visualization 7: Correlations
elif viz_type == "Correlations":
    st.markdown("### Correlation Analysis")
    
    # Select numeric columns
    numeric_cols = ['ArrDelayMinutes', 'DepDelayMinutes', 'TaxiOut', 'TaxiIn', 
                    'DepHour', 'DayOfWeek', 'Month']
    
    if 'Distance' in df_filtered.columns:
        numeric_cols.append('Distance')
    
    corr_data = df_filtered[numeric_cols].dropna()
    corr_matrix = corr_data.corr()
    
    # Create heatmap with text annotations showing correlation values
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu_r',
        zmin=-1,
        zmax=1,
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title="Correlation Matrix (values shown in cells)",
        height=600,
        xaxis_title="",
        yaxis_title=""
    )
    
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("💡 **Tip**: Use the sidebar filters to focus on specific airlines, time periods, or flight characteristics!")

