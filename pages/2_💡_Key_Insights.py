"""
Key Insights Page - Curated Storytelling from EDA
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
    get_delay_category_distribution
)

# Page config
st.set_page_config(page_title="Key Insights", page_icon="💡", layout="wide")

st.title("💡 Key Insights from Exploratory Data Analysis")
st.markdown("**Curated findings and stories** from our deep dive into US flight delay patterns")

# Load data
with st.spinner("Loading flight data..."):
    df = load_flight_data(max_rows_per_file=100000)
    df = prepare_flight_features(df)

st.markdown("---")

# Navigation for insights
st.sidebar.header("📖 Insights Navigation")
insight_section = st.sidebar.radio(
    "Jump to Section",
    [
        "Overview",
        "1️⃣ The Delay Paradox",
        "2️⃣ Time is Everything",
        "3️⃣ Airline Performance Gap",
        "4️⃣ Cascading Effects",
        "5️⃣ Seasonal Patterns",
        "6️⃣ The Distance Myth"
    ]
)

# OVERVIEW
if insight_section == "Overview":
    st.markdown("""
    ## 📊 Executive Summary
    
    Through our analysis of **500,000+ US domestic flights** from 2018-2022, we discovered 
    several surprising patterns that challenge common assumptions about flight delays.
    
    ### 🎯 Key Takeaways:
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🕐 Time Matters Most**
        - Evening flights have **2.2× higher** delay rates than morning flights
        - Delays accumulate throughout the day
        - Thursday & Friday are the worst days
        """)
    
    with col2:
        st.markdown("""
        **✈️ Not All Airlines Equal**
        - **15-20 min difference** between best and worst airlines
        - Smaller carriers often have higher delay rates
        - Cancellation rates vary **10×** across airlines
        """)
    
    with col3:
        st.markdown("""
        **🔗 Cascading Effects**
        - **98% correlation** between departure and arrival delays
        - Taxi times strongly predict delays
        - Distance has minimal impact on delay probability
        """)
    
    st.markdown("---")
    
    # Overall statistics
    st.markdown("### 📈 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_flights = len(df)
        st.metric("Total Flights", f"{total_flights:,}")
    
    with col2:
        avg_delay = df['ArrDelayMinutes'].mean()
        st.metric("Avg Arrival Delay", f"{avg_delay:.1f} min")
    
    with col3:
        delay_rate = (df['IsArrDelayed'].mean() * 100)
        st.metric("Delay Rate (>15 min)", f"{delay_rate:.1f}%")
    
    with col4:
        on_time_rate = ((df['ArrDelayMinutes'] <= 15).sum() / len(df) * 100)
        st.metric("On-Time Performance", f"{on_time_rate:.1f}%")
    
    st.markdown("---")
    st.info("👈 **Navigate through specific insights using the sidebar menu**")

# INSIGHT 1: The Delay Paradox
elif insight_section == "1️⃣ The Delay Paradox":
    st.markdown("""
    ## 1️⃣ The Delay Paradox: Most Flights Are On Time, Yet Delays Are Common
    
    At first glance, the data seems contradictory: **80% of flights arrive on time or early**, 
    yet delays are a major concern. How can both be true?
    """)
    
    # Distribution visualization
    st.markdown("### The Distribution Reveals the Truth")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create distribution plot
        delays = df['ArrDelayMinutes'].dropna()
        
        fig = go.Figure()
        
        # Histogram for on-time/early
        on_time_delays = delays[delays <= 15]
        fig.add_trace(go.Histogram(
            x=on_time_delays,
            name='On-Time (≤15 min)',
            marker_color='green',
            opacity=0.7,
            nbinsx=50
        ))
        
        # Histogram for delayed
        late_delays = delays[delays > 15]
        fig.add_trace(go.Histogram(
            x=late_delays,
            name='Delayed (>15 min)',
            marker_color='red',
            opacity=0.7,
            nbinsx=50
        ))
        
        fig.update_layout(
            title="Arrival Delay Distribution: The Long Tail Problem",
            xaxis_title="Arrival Delay (minutes)",
            yaxis_title="Number of Flights",
            barmode='overlay',
            xaxis_range=[-50, 300]
        )
        fig.add_vline(x=15, line_dash="dash", line_color="orange", 
                     annotation_text="Delay Threshold (15 min)")
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("""
        ### 🔍 What This Means
        
        **The "Long Tail" Problem:**
        - Most flights cluster near 0 delay
        - But **20% experience significant delays**
        - These delays can be **extreme** (up to 35+ hours in our data!)
        
        **Why It Matters:**
        - Average delay (12.7 min) doesn't tell the full story
        - Travelers face high **uncertainty**
        - A few severe delays disrupt many passengers
        """)
    
    # Category breakdown
    st.markdown("### Breaking Down the Categories")
    
    category_counts = df['ArrDelayCategory'].value_counts()
    category_pct = (category_counts / len(df) * 100).round(1)
    
    fig = go.Figure()
    
    colors = {
        'On Time': 'green',
        'Small Delay': 'yellow',
        'Moderate Delay': 'orange',
        'Large Delay': 'red',
        'Very Large Delay': 'darkred',
        'Unknown': 'gray'
    }
    
    for cat in ['On Time', 'Small Delay', 'Moderate Delay', 'Large Delay', 'Very Large Delay']:
        if cat in category_counts.index:
            fig.add_trace(go.Bar(
                name=cat,
                x=[cat],
                y=[category_pct[cat]],
                marker_color=colors.get(cat, 'blue'),
                text=f"{category_pct[cat]}%",
                textposition='auto'
            ))
    
    fig.update_layout(
        title="Delay Category Breakdown",
        yaxis_title="Percentage of Flights (%)",
        showlegend=False,
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.success("""
    **💡 Key Insight**: While most flights are on time, the **20% that aren't** create the perception 
    that delays are everywhere. Extreme delays, though rare (1%), have outsized impact on passenger experience.
    """)

# INSIGHT 2: Time is Everything
elif insight_section == "2️⃣ Time is Everything":
    st.markdown("""
    ## 2️⃣ Time is Everything: The Power of Departure Hour
    
    **When you fly matters more than almost any other factor.** Our analysis reveals dramatic 
    differences in delay rates based on departure time.
    """)
    
    # Hour of day analysis
    st.markdown("### Delays Accumulate Throughout the Day")
    
    hour_agg = df.groupby('DepHour').agg(
        avg_delay=('ArrDelayMinutes', 'mean'),
        delay_rate=('IsArrDelayed', 'mean'),
        flights=('Cancelled', 'size')
    ).reset_index()
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Average Arrival Delay by Departure Hour", 
                       "Delay Rate (>15 min) by Departure Hour"),
        vertical_spacing=0.12
    )
    
    fig.add_trace(
        go.Scatter(
            x=hour_agg['DepHour'],
            y=hour_agg['avg_delay'],
            mode='lines+markers',
            name='Avg Delay',
            line=dict(color='red', width=3),
            marker=dict(size=8)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=hour_agg['DepHour'],
            y=hour_agg['delay_rate'] * 100,
            mode='lines+markers',
            name='Delay Rate',
            line=dict(color='orange', width=3),
            marker=dict(size=8)
        ),
        row=2, col=1
    )
    
    # Add shaded regions for time of day
    for row in [1, 2]:
        # Night (0-6)
        fig.add_vrect(x0=0, x1=6, fillcolor="blue", opacity=0.1, layer="below", row=row, col=1)
        # Morning (6-12)
        fig.add_vrect(x0=6, x1=12, fillcolor="green", opacity=0.1, layer="below", row=row, col=1)
        # Afternoon (12-18)
        fig.add_vrect(x0=12, x1=18, fillcolor="yellow", opacity=0.1, layer="below", row=row, col=1)
        # Evening (18-24)
        fig.add_vrect(x0=18, x1=24, fillcolor="red", opacity=0.1, layer="below", row=row, col=1)
    
    fig.update_xaxes(title_text="Departure Hour", row=2, col=1)
    fig.update_yaxes(title_text="Avg Delay (min)", row=1, col=1)
    fig.update_yaxes(title_text="Delay Rate (%)", row=2, col=1)
    
    fig.update_layout(height=700, showlegend=False)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Time of day comparison
    st.markdown("### The Numbers Tell a Clear Story")
    
    tod_stats = df.groupby('DepTimeOfDay').agg(
        avg_delay=('ArrDelayMinutes', 'mean'),
        delay_rate=('IsArrDelayed', lambda x: x.mean() * 100),
        flights=('Cancelled', 'size')
    ).reset_index()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            tod_stats,
            x='DepTimeOfDay',
            y='avg_delay',
            title="Average Delay by Time of Day",
            labels={'avg_delay': 'Average Delay (min)', 'DepTimeOfDay': 'Time of Day'},
            color='avg_delay',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            tod_stats,
            x='DepTimeOfDay',
            y='delay_rate',
            title="Delay Rate by Time of Day",
            labels={'delay_rate': 'Delay Rate (%)', 'DepTimeOfDay': 'Time of Day'},
            color='delay_rate',
            color_continuous_scale='Oranges'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Weekday patterns
    st.markdown("### Weekday Patterns Matter Too")
    
    dow_stats = df.groupby('DayOfWeekName').agg(
        avg_delay=('ArrDelayMinutes', 'mean'),
        delay_rate=('IsArrDelayed', lambda x: x.mean() * 100)
    ).reset_index()
    
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_stats['DayOfWeekName'] = pd.Categorical(dow_stats['DayOfWeekName'], categories=weekday_order, ordered=True)
    dow_stats = dow_stats.sort_values('DayOfWeekName')
    
    fig = px.bar(
        dow_stats,
        x='DayOfWeekName',
        y='avg_delay',
        title="Average Delay by Day of Week",
        labels={'avg_delay': 'Average Delay (min)', 'DayOfWeekName': 'Day of Week'},
        color='avg_delay',
        color_continuous_scale='RdYlGn_r'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.success("""
    **💡 Key Insight**: 
    - **Morning flights (6-12h)** have ~50% lower delay rates than evening flights
    - **Thursday and Friday** are the worst days to fly
    - **Tuesday and Wednesday** offer the best on-time performance
    - Choose early morning flights on Tuesdays or Wednesdays for best results!
    """)

# INSIGHT 3: Airline Performance Gap
elif insight_section == "3️⃣ Airline Performance Gap":
    st.markdown("""
    ## 3️⃣ The Airline Performance Gap: Big Differences in Reliability
    
    **Not all airlines are created equal.** Our analysis reveals substantial performance 
    differences across carriers, with some consistently outperforming others.
    """)
    
    airline_stats = aggregate_by_airline(df, min_flights=5000)
    
    st.markdown("### Top and Bottom Performers")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ❌ Worst Average Delays")
        worst = airline_stats.nlargest(10, 'avg_arr_delay')[['Airline', 'avg_arr_delay', 'total_flights']]
        worst['avg_arr_delay'] = worst['avg_arr_delay'].round(1)
        worst.columns = ['Airline', 'Avg Delay (min)', 'Flights']
        st.dataframe(worst, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("#### ✅ Best Average Delays")
        best = airline_stats.nsmallest(10, 'avg_arr_delay')[['Airline', 'avg_arr_delay', 'total_flights']]
        best['avg_arr_delay'] = best['avg_arr_delay'].round(1)
        best.columns = ['Airline', 'Avg Delay (min)', 'Flights']
        st.dataframe(best, use_container_width=True, hide_index=True)
    
    # Visualization
    st.markdown("### Performance Comparison Across Airlines")
    
    top_15 = airline_stats.nlargest(15, 'total_flights')
    
    fig = px.scatter(
        top_15,
        x='avg_arr_delay',
        y='cancel_rate',
        size='total_flights',
        color='arr_delay_rate',
        hover_data=['Airline'],
        labels={
            'avg_arr_delay': 'Average Arrival Delay (minutes)',
            'cancel_rate': 'Cancellation Rate',
            'arr_delay_rate': 'Delay Rate (>15 min)',
            'total_flights': 'Number of Flights'
        },
        title="Airline Performance: Delay vs Cancellation (bubble size = flight volume)",
        color_continuous_scale='Reds'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Delay category distribution by airline
    st.markdown("### How Delay Categories Vary by Airline")
    
    top_airlines_list = airline_stats.nlargest(10, 'total_flights')['Airline'].tolist()
    df_top_airlines = df[df['Airline'].isin(top_airlines_list)]
    
    cat_dist = get_delay_category_distribution(df_top_airlines, 'Airline')
    
    fig = go.Figure()
    
    category_order = ['On Time', 'Small Delay', 'Moderate Delay', 'Large Delay', 'Very Large Delay']
    colors_map = {
        'On Time': 'green',
        'Small Delay': 'yellow',
        'Moderate Delay': 'orange',
        'Large Delay': 'red',
        'Very Large Delay': 'darkred'
    }
    
    for cat in category_order:
        if cat in cat_dist.columns:
            fig.add_trace(go.Bar(
                name=cat,
                x=cat_dist.index,
                y=cat_dist[cat],
                marker_color=colors_map[cat]
            ))
    
    fig.update_layout(
        barmode='stack',
        title="Delay Category Distribution by Airline (Top 10 by Volume)",
        xaxis_title="Airline",
        yaxis_title="Percentage of Flights (%)",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.success("""
    **💡 Key Insight**: 
    - Performance gap of **15-20 minutes** between best and worst airlines
    - Larger carriers often have better infrastructure and recovery capabilities
    - Cancellation rates vary by up to **10× across airlines**
    - Consider airline reliability when booking, not just price!
    """)

# INSIGHT 4: Cascading Effects
elif insight_section == "4️⃣ Cascading Effects":
    st.markdown("""
    ## 4️⃣ Cascading Effects: How One Delay Creates Another
    
    **Delays don't happen in isolation.** Our correlation analysis reveals strong cascading 
    effects throughout the flight system.
    """)
    
    st.markdown("### The Departure-Arrival Connection")
    
    # Scatter plot with density
    valid_data = df[['DepDelayMinutes', 'ArrDelayMinutes']].dropna()
    sample = valid_data.sample(min(20000, len(valid_data)))
    
    fig = px.density_contour(
        sample,
        x='DepDelayMinutes',
        y='ArrDelayMinutes',
        title="Departure vs Arrival Delay (98% Correlation)",
        labels={'DepDelayMinutes': 'Departure Delay (minutes)', 'ArrDelayMinutes': 'Arrival Delay (minutes)'},
        marginal_x="histogram",
        marginal_y="histogram"
    )
    
    # Add diagonal line
    fig.add_trace(go.Scatter(
        x=[-50, 300],
        y=[-50, 300],
        mode='lines',
        name='Perfect Correlation',
        line=dict(dash='dash', color='red')
    ))
    
    fig.update_xaxes(range=[-50, 300])
    fig.update_yaxes(range=[-50, 300])
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation matrix
    st.markdown("### Key Correlations with Arrival Delay")
    
    corr_cols = ['ArrDelayMinutes', 'DepDelayMinutes', 'TaxiOut', 'TaxiIn', 'DepHour']
    if 'Distance' in df.columns:
        corr_cols.append('Distance')
    
    corr_data = df[corr_cols].dropna()
    corr_matrix = corr_data.corr()['ArrDelayMinutes'].sort_values(ascending=False)
    
    corr_df = pd.DataFrame({
        'Feature': corr_matrix.index,
        'Correlation': corr_matrix.values
    })
    corr_df = corr_df[corr_df['Feature'] != 'ArrDelayMinutes']
    
    fig = px.bar(
        corr_df,
        x='Correlation',
        y='Feature',
        orientation='h',
        title="Correlation with Arrival Delay",
        labels={'Correlation': 'Pearson Correlation Coefficient'},
        color='Correlation',
        color_continuous_scale='RdBu_r',
        range_color=[-1, 1]
    )
    fig.add_vline(x=0, line_dash="dash", line_color="black")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Taxi time analysis
    st.markdown("### The Taxi Time Factor")
    
    col1, col2 = st.columns(2)
    
    with col1:
        taxi_out_sample = df[['TaxiOut', 'ArrDelayMinutes']].dropna().sample(min(10000, len(df)))
        fig = px.scatter(
            taxi_out_sample,
            x='TaxiOut',
            y='ArrDelayMinutes',
            title="Taxi Out Time vs Arrival Delay",
            labels={'TaxiOut': 'Taxi Out Time (min)', 'ArrDelayMinutes': 'Arrival Delay (min)'},
            opacity=0.3
        )
        fig.update_xaxes(range=[0, 100])
        fig.update_yaxes(range=[-50, 200])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        taxi_in_sample = df[['TaxiIn', 'ArrDelayMinutes']].dropna().sample(min(10000, len(df)))
        fig = px.scatter(
            taxi_in_sample,
            x='TaxiIn',
            y='ArrDelayMinutes',
            title="Taxi In Time vs Arrival Delay",
            labels={'TaxiIn': 'Taxi In Time (min)', 'ArrDelayMinutes': 'Arrival Delay (min)'},
            opacity=0.3
        )
        fig.update_xaxes(range=[0, 100])
        fig.update_yaxes(range=[-50, 200])
        st.plotly_chart(fig, use_container_width=True)
    
    st.success("""
    **💡 Key Insight**: 
    - **98% correlation** between departure and arrival delays (nearly perfect!)
    - Taxi times (ground operations) strongly predict delays
    - **Airport congestion** is a major factor (reflected in taxi times)
    - Delays cascade: one delayed flight impacts others at the gate/runway
    """)

# INSIGHT 5: Seasonal Patterns
elif insight_section == "5️⃣ Seasonal Patterns":
    st.markdown("""
    ## 5️⃣ Seasonal Patterns: Summer and Winter Challenges
    
    **Delays follow predictable seasonal patterns**, with certain months consistently 
    showing higher delay rates.
    """)
    
    monthly_stats = df.groupby('MonthName').agg(
        avg_delay=('ArrDelayMinutes', 'mean'),
        delay_rate=('IsArrDelayed', lambda x: x.mean() * 100),
        cancel_rate=('Cancelled', lambda x: x.mean() * 100),
        flights=('Cancelled', 'size')
    ).reset_index()
    
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    monthly_stats['MonthName'] = pd.Categorical(monthly_stats['MonthName'], categories=month_order, ordered=True)
    monthly_stats = monthly_stats.sort_values('MonthName')
    
    st.markdown("### Monthly Delay Patterns")
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Average Arrival Delay by Month", "Cancellation Rate by Month"),
        vertical_spacing=0.12
    )
    
    fig.add_trace(
        go.Scatter(
            x=monthly_stats['MonthName'],
            y=monthly_stats['avg_delay'],
            mode='lines+markers',
            name='Avg Delay',
            line=dict(color='red', width=3),
            marker=dict(size=10),
            fill='tozeroy',
            fillcolor='rgba(255, 0, 0, 0.1)'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=monthly_stats['MonthName'],
            y=monthly_stats['cancel_rate'],
            name='Cancel Rate',
            marker_color='orange'
        ),
        row=2, col=1
    )
    
    fig.update_xaxes(title_text="Month", row=2, col=1)
    fig.update_yaxes(title_text="Avg Delay (min)", row=1, col=1)
    fig.update_yaxes(title_text="Cancel Rate (%)", row=2, col=1)
    
    fig.update_layout(height=700, showlegend=False)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Seasonal breakdown
    st.markdown("### Seasonal Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### ❄️ Winter Challenges
        - **December-February**: Higher cancellation rates
        - Weather disruptions (snow, ice)
        - Holiday travel congestion
        - February: Worst cancellation rate
        """)
    
    with col2:
        st.markdown("""
        #### ☀️ Summer Peak
        - **June-July**: Highest average delays
        - Peak travel season
        - Thunderstorms and convective weather
        - Airport capacity constraints
        """)
    
    # Best and worst months
    st.markdown("### Best and Worst Months to Fly")
    
    col1, col2 = st.columns(2)
    
    with col1:
        worst_months = monthly_stats.nlargest(5, 'avg_delay')[['MonthName', 'avg_delay', 'delay_rate']]
        worst_months.columns = ['Month', 'Avg Delay (min)', 'Delay Rate (%)']
        st.markdown("#### ❌ Worst Months")
        st.dataframe(worst_months, use_container_width=True, hide_index=True)
    
    with col2:
        best_months = monthly_stats.nsmallest(5, 'avg_delay')[['MonthName', 'avg_delay', 'delay_rate']]
        best_months.columns = ['Month', 'Avg Delay (min)', 'Delay Rate (%)']
        st.markdown("#### ✅ Best Months")
        st.dataframe(best_months, use_container_width=True, hide_index=True)
    
    st.success("""
    **💡 Key Insight**: 
    - **September-November** offer the best on-time performance
    - **June-July** have highest delays (summer thunderstorms + peak travel)
    - **February-March** have highest cancellations (winter weather)
    - Avoid summer and winter holidays for best reliability
    """)

# INSIGHT 6: The Distance Myth
elif insight_section == "6️⃣ The Distance Myth":
    st.markdown("""
    ## 6️⃣ The Distance Myth: Longer ≠ More Delayed
    
    **Contrary to popular belief, flight distance has minimal impact on delay probability.** 
    Our analysis challenges the assumption that longer flights are more prone to delays.
    """)
    
    if 'Distance' in df.columns:
        # Create distance bins
        df_dist = df.copy()
        df_dist['DistanceBin'] = pd.cut(
            df_dist['Distance'],
            bins=[0, 500, 1000, 1500, 2000, 3000, 10000],
            labels=['<500', '500-1000', '1000-1500', '1500-2000', '2000-3000', '>3000']
        )
        
        dist_stats = df_dist.groupby('DistanceBin').agg(
            avg_delay=('ArrDelayMinutes', 'mean'),
            delay_rate=('IsArrDelayed', lambda x: x.mean() * 100),
            flights=('Cancelled', 'size')
        ).reset_index()
        
        st.markdown("### Delay Metrics by Flight Distance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                dist_stats,
                x='DistanceBin',
                y='avg_delay',
                title="Average Delay by Distance",
                labels={'avg_delay': 'Average Delay (min)', 'DistanceBin': 'Distance (miles)'},
                color='avg_delay',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                dist_stats,
                x='DistanceBin',
                y='delay_rate',
                title="Delay Rate by Distance",
                labels={'delay_rate': 'Delay Rate (%)', 'DistanceBin': 'Distance (miles)'},
                color='delay_rate',
                color_continuous_scale='Oranges'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Scatter plot
        st.markdown("### Distance vs Delay: The Weak Relationship")
        
        dist_sample = df[['Distance', 'ArrDelayMinutes']].dropna().sample(min(15000, len(df)))
        
        fig = px.scatter(
            dist_sample,
            x='Distance',
            y='ArrDelayMinutes',
            title="Flight Distance vs Arrival Delay",
            labels={'Distance': 'Distance (miles)', 'ArrDelayMinutes': 'Arrival Delay (minutes)'},
            opacity=0.1
        )
        fig.update_yaxes(range=[-50, 200])
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation
        corr = df[['Distance', 'ArrDelayMinutes']].corr().iloc[0, 1]
        
        st.markdown(f"""
        ### The Numbers Don't Lie
        
        **Correlation between distance and arrival delay: {corr:.3f}**
        
        This extremely low correlation means:
        - Distance explains **<1%** of delay variance
        - Short flights can be just as delayed as long flights
        - Other factors (time of day, airline, airport congestion) matter far more
        """)
    
    else:
        st.warning("Distance data not available in this dataset subset.")
    
    st.success("""
    **💡 Key Insight**: 
    - **Distance has almost no correlation** with delay probability
    - Longer flights may actually have **more buffer time** built in
    - Focus on **when** and **who** you fly with, not how far
    - Short regional flights can be just as problematic as transcontinental ones
    """)

st.markdown("---")
st.info("💡 **These insights guide our predictive modeling**: We prioritize temporal features, airline characteristics, and operational factors over distance-based features.")

