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
        "4️⃣ Seasonal Patterns",
        "5️⃣ The Distance Myth",
        "6️⃣ Year-over-Year Trends"
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
        **📅 Temporal Patterns**
        - **Year-over-year** trends show significant variations
        - **Seasonal patterns** reveal summer and winter challenges
        - **Time of day** is the strongest predictor
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
    st.caption("Categories: Early (<0), On Time (0), Small (1-15), Moderate (16-60), Large (61-180), Very Large (>180) minutes")
    
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
    # Ensure cancellation rate exists (some datasets may lack Cancelled column)
    if 'cancel_rate' not in airline_stats.columns:
        if 'Cancelled' in df.columns:
            cancel_rate = df.groupby('Airline')['Cancelled'].mean().reset_index(name='cancel_rate')
            airline_stats = airline_stats.merge(cancel_rate, on='Airline', how='left')
        else:
            airline_stats['cancel_rate'] = 0.0
    
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
    
    # Delay category distribution by airline - Top 5 worst and Bottom 5 best
    st.markdown("### How Delay Categories Vary by Airline: Best vs Worst Performers")
    
    # Get top 5 worst and bottom 5 best by delay rate
    worst_5 = airline_stats.nlargest(5, 'arr_delay_rate')['Airline'].tolist()
    best_5 = airline_stats.nsmallest(5, 'arr_delay_rate')['Airline'].tolist()
    
    comparison_airlines = worst_5 + best_5
    df_comparison = df[df['Airline'].isin(comparison_airlines)]
    
    cat_dist = get_delay_category_distribution(df_comparison, 'Airline')
    
    # Reorder: worst first, then best
    airline_order = worst_5 + best_5
    cat_dist = cat_dist.reindex([a for a in airline_order if a in cat_dist.index])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ❌ Top 5 Worst Performers (Highest Delay Rate)")
        worst_df = df[df['Airline'].isin(worst_5)]
        worst_cat_dist = get_delay_category_distribution(worst_df, 'Airline')
        worst_cat_dist = worst_cat_dist.reindex([a for a in worst_5 if a in worst_cat_dist.index])
        
        fig_worst = go.Figure()
        category_order = ['On Time', 'Small Delay', 'Moderate Delay', 'Large Delay', 'Very Large Delay']
        colors_map = {
            'On Time': 'green',
            'Small Delay': 'yellow',
            'Moderate Delay': 'orange',
            'Large Delay': 'red',
            'Very Large Delay': 'darkred'
        }
        
        for cat in category_order:
            if cat in worst_cat_dist.columns:
                fig_worst.add_trace(go.Bar(
                    name=cat,
                    x=worst_cat_dist.index,
                    y=worst_cat_dist[cat],
                    marker_color=colors_map[cat]
                ))
        
        fig_worst.update_layout(
            barmode='stack',
            title="Worst 5 Airlines",
            xaxis_title="Airline",
            yaxis_title="Percentage of Flights (%)",
            height=400,
            showlegend=True
        )
        st.plotly_chart(fig_worst, use_container_width=True)
    
    with col2:
        st.markdown("#### ✅ Top 5 Best Performers (Lowest Delay Rate)")
        best_df = df[df['Airline'].isin(best_5)]
        best_cat_dist = get_delay_category_distribution(best_df, 'Airline')
        best_cat_dist = best_cat_dist.reindex([a for a in best_5 if a in best_cat_dist.index])
        
        fig_best = go.Figure()
        
        for cat in category_order:
            if cat in best_cat_dist.columns:
                fig_best.add_trace(go.Bar(
                    name=cat,
                    x=best_cat_dist.index,
                    y=best_cat_dist[cat],
                    marker_color=colors_map[cat]
                ))
        
        fig_best.update_layout(
            barmode='stack',
            title="Best 5 Airlines",
            xaxis_title="Airline",
            yaxis_title="Percentage of Flights (%)",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig_best, use_container_width=True)
    
    # Comparison table
    st.markdown("#### 📊 Performance Comparison")
    comparison_stats = airline_stats[airline_stats['Airline'].isin(comparison_airlines)][
        ['Airline', 'arr_delay_rate', 'avg_arr_delay', 'total_flights']
    ].round(2)
    comparison_stats = comparison_stats.sort_values('arr_delay_rate', ascending=False)
    comparison_stats.columns = ['Airline', 'Delay Rate (%)', 'Avg Delay (min)', 'Total Flights']
    st.dataframe(comparison_stats, use_container_width=True, hide_index=True)
    
    st.success("""
    **💡 Key Insight**: 
    - Performance gap of **15-20 minutes** between best and worst airlines
    - Larger carriers often have better infrastructure and recovery capabilities
    - Cancellation rates vary by up to **10× across airlines**
    - Consider airline reliability when booking, not just price!
    """)

# INSIGHT 4: Seasonal Patterns
elif insight_section == "4️⃣ Seasonal Patterns":
    st.markdown("""
    ## 4️⃣ Seasonal Patterns: Summer and Winter Challenges
    
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
    
    # Identify actual worst months for cancellations
    worst_cancel_months = monthly_stats.nlargest(3, 'cancel_rate')['MonthName'].tolist()
    worst_cancel_month = worst_cancel_months[0] if worst_cancel_months else "February"
    
    # Seasonal breakdown - based on actual data
    st.markdown("### Seasonal Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        #### ❄️ Winter Weather Impact
        - **{worst_cancel_month}** typically shows highest cancellation rates
        - Weather disruptions (snow, ice, freezing conditions)
        - Airport de-icing operations cause delays
        - Holiday travel volume increases congestion
        - **Operational Note**: Airlines build in buffer time, but severe weather still causes cancellations
        """)
    
    with col2:
        st.markdown("""
        #### ☀️ Summer Operational Challenges
        - **June-July**: Peak delays due to multiple factors
        - **Convective weather** (thunderstorms) is common in summer
        - Peak travel season → airport capacity constraints
        - Higher flight volume → cascading delays
        - **Operational Note**: Summer delays are more about system capacity than weather severity
        """)
    
    # Best and worst months
    st.markdown("### Best and Worst Months to Fly")
    
    col1, col2 = st.columns(2)
    
    with col1:
        worst_months = monthly_stats.nlargest(5, 'avg_delay')[['MonthName', 'avg_delay', 'delay_rate', 'cancel_rate']]
        worst_months.columns = ['Month', 'Avg Delay (min)', 'Delay Rate (%)', 'Cancel Rate (%)']
        st.markdown("#### ❌ Worst Months (by Average Delay)")
        st.dataframe(worst_months, use_container_width=True, hide_index=True)
    
    with col2:
        best_months = monthly_stats.nsmallest(5, 'avg_delay')[['MonthName', 'avg_delay', 'delay_rate', 'cancel_rate']]
        best_months.columns = ['Month', 'Avg Delay (min)', 'Delay Rate (%)', 'Cancel Rate (%)']
        st.markdown("#### ✅ Best Months (by Average Delay)")
        st.dataframe(best_months, use_container_width=True, hide_index=True)
    
    # Cancellation analysis
    st.markdown("### Cancellation Patterns")
    worst_cancel = monthly_stats.nlargest(3, 'cancel_rate')[['MonthName', 'cancel_rate', 'avg_delay']]
    worst_cancel.columns = ['Month', 'Cancel Rate (%)', 'Avg Delay (min)']
    st.markdown("**Highest Cancellation Rates:**")
    st.dataframe(worst_cancel, use_container_width=True, hide_index=True)
    
    st.success("""
    **💡 Key Insight**: 
    - **September-November** (fall) offer the best on-time performance - mild weather and moderate travel volume
    - **June-July** have highest delays due to summer thunderstorms + peak travel season capacity constraints
    - **Winter months** (especially {}) show highest cancellation rates due to severe weather
    - **Practical Takeaway**: While summer and winter are peak travel times, understanding the operational 
    factors (weather patterns, capacity constraints) helps set realistic expectations in terms of delays and cancellations.
    """.format(worst_cancel_month))

# INSIGHT 5: The Distance Myth
elif insight_section == "5️⃣ The Distance Myth":
    st.markdown("""
    ## 5️⃣ The Distance Myth: Longer ≠ More Delayed
    
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

# INSIGHT 6: Year-over-Year Trends
elif insight_section == "6️⃣ Year-over-Year Trends":
    st.markdown("""
    ## 6️⃣ Year-over-Year Trends: The Impact of Time and Events
    
    **How have flight delays changed over the years?** Our analysis reveals significant 
    year-to-year variations, with the COVID-19 pandemic creating a dramatic shift in patterns.
    """)
    
    # Yearly aggregation
    yearly_stats = df.groupby('Year').agg(
        avg_delay=('ArrDelayMinutes', 'mean'),
        delay_rate=('IsArrDelayed', lambda x: x.mean() * 100),
        cancel_rate=('Cancelled', lambda x: x.mean() * 100),
        flights=('Cancelled', 'size')
    ).reset_index()
    
    st.markdown("### Yearly Delay Trends")
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Average Arrival Delay by Year", "Delay Rate and Cancellation Rate by Year"),
        vertical_spacing=0.12
    )
    
    # Average delay
    fig.add_trace(
        go.Scatter(
            x=yearly_stats['Year'],
            y=yearly_stats['avg_delay'],
            mode='lines+markers',
            name='Avg Delay',
            line=dict(color='red', width=3),
            marker=dict(size=12)
        ),
        row=1, col=1
    )
    
    # Delay rate and cancellation rate
    fig.add_trace(
        go.Scatter(
            x=yearly_stats['Year'],
            y=yearly_stats['delay_rate'],
            mode='lines+markers',
            name='Delay Rate',
            line=dict(color='orange', width=3),
            marker=dict(size=12)
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=yearly_stats['Year'],
            y=yearly_stats['cancel_rate'],
            mode='lines+markers',
            name='Cancel Rate',
            line=dict(color='purple', width=3, dash='dash'),
            marker=dict(size=12)
        ),
        row=2, col=1
    )
    
    # Highlight COVID period
    fig.add_vrect(x0=2020, x1=2021, fillcolor="yellow", opacity=0.2, layer="below", row=1, col=1)
    fig.add_vrect(x0=2020, x1=2021, fillcolor="yellow", opacity=0.2, layer="below", row=2, col=1)
    
    fig.update_xaxes(title_text="Year", row=2, col=1)
    fig.update_yaxes(title_text="Avg Delay (min)", row=1, col=1)
    fig.update_yaxes(title_text="Rate (%)", row=2, col=1)
    
    fig.update_layout(height=700, showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Yearly comparison table
    st.markdown("### Year-by-Year Comparison")
    yearly_display = yearly_stats.copy()
    yearly_display['Total Flights'] = yearly_display['flights'].apply(lambda x: f"{x:,}")
    yearly_display = yearly_display[['Year', 'avg_delay', 'delay_rate', 'cancel_rate', 'Total Flights']]
    yearly_display.columns = ['Year', 'Avg Delay (min)', 'Delay Rate (%)', 'Cancel Rate (%)', 'Total Flights']
    yearly_display = yearly_display.round(2)
    st.dataframe(yearly_display, use_container_width=True, hide_index=True)
    
    # Pre vs Post COVID analysis
    st.markdown("### Pre-Pandemic vs Pandemic Period")
    
    pre_covid = df[df['Year'].isin([2018, 2019])]
    covid = df[df['Year'].isin([2020, 2021])]
    post_covid = df[df['Year'] == 2022] if 2022 in df['Year'].values else None
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 📅 Pre-Pandemic (2018-2019)")
        pre_stats = {
            'Avg Delay': f"{pre_covid['ArrDelayMinutes'].mean():.1f} min",
            'Delay Rate': f"{pre_covid['IsArrDelayed'].mean() * 100:.1f}%",
            'Cancel Rate': f"{pre_covid['Cancelled'].mean() * 100:.2f}%",
            'Flights': f"{len(pre_covid):,}"
        }
        for key, val in pre_stats.items():
            st.metric(key, val)
    
    with col2:
        st.markdown("#### 🦠 Pandemic (2020-2021)")
        covid_stats = {
            'Avg Delay': f"{covid['ArrDelayMinutes'].mean():.1f} min",
            'Delay Rate': f"{covid['IsArrDelayed'].mean() * 100:.1f}%",
            'Cancel Rate': f"{covid['Cancelled'].mean() * 100:.2f}%",
            'Flights': f"{len(covid):,}"
        }
        for key, val in covid_stats.items():
            st.metric(key, val)
    
    with col3:
        if post_covid is not None and len(post_covid) > 0:
            st.markdown("#### ✈️ Recovery (2022)")
            post_stats = {
                'Avg Delay': f"{post_covid['ArrDelayMinutes'].mean():.1f} min",
                'Delay Rate': f"{post_covid['IsArrDelayed'].mean() * 100:.1f}%",
                'Cancel Rate': f"{post_covid['Cancelled'].mean() * 100:.2f}%",
                'Flights': f"{len(post_covid):,}"
            }
            for key, val in post_stats.items():
                st.metric(key, val)
        else:
            st.info("2022 data not available in this sample")
    
    st.success("""
    **💡 Key Insight**: 
    - **2018-2019**: Pre-pandemic baseline shows typical delay patterns
    - **2020-2021**: Dramatic reduction in flight volume, delays persisted but in less magnitude, probably due to less airport congestion and more efficiency in the little amount of flights there were. However, there was a noticeable increase in cancellations during that period.
    - **2022**: Recovery period showing return to pre-pandemic patterns
    - **Year is a strong predictor** in our model because it captures these systemic changes
    - Understanding year-over-year trends helps contextualize current delay predictions
    """)

st.markdown("---")
st.info("💡 **These insights guide our predictive modeling**: We prioritize temporal features, airline characteristics, and operational factors over distance-based features.")

