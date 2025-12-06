"""
Flight Delay Prediction & Visual Analytics
Main Streamlit Application

This app provides interactive visualizations and predictive analytics
for US domestic flight delays, cancellations, and diversions.
"""

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Flight Delay Analytics",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Main content
st.markdown('<div class="main-header">✈️ Flight Delay Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Interactive Visual Analytics & Predictive Modeling for US Domestic Flights</div>', unsafe_allow_html=True)

st.markdown("---")

# Introduction
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ## Welcome to Flight Delay Analytics
    
    This interactive application provides comprehensive insights into US domestic flight delays, 
    cancellations, and diversions using data from 2018-2022.
    
    ### 🎯 What You Can Do Here:
    
    - **📊 Explore Data**: Interactively filter and visualize flight delay patterns with customizable charts
    - **💡 Key Insights**: Discover curated findings and stories from our exploratory data analysis
    - **🤖 Predict Delays**: Use machine learning to predict flight delays
    - **🔍 Model Explainability**: Understand what drives delay predictions
    
    ### 📈 Dataset Overview:
    
    Our analysis covers **US domestic flights** from January 2018 to July 2022, including:
    - Flight schedules and actual times
    - Delays (departure and arrival)
    - Cancellations and diversions
    - Airlines, routes, and airports
    - Temporal patterns (time of day, day of week, seasonality)
    
    **📊 Data Note**: The dataset has been sampled using reservoir sampling to ensure 
    representative distribution while maintaining computational efficiency. The sample 
    accurately represents the full dataset's patterns and distributions, though it is 
    not the complete dataset.
    """)

with col2:
    st.markdown("""
    ### 🚀 Quick Start
    
    1. **Explore Data** to get familiar with the dataset through interactive visualizations
    
    2. **Key Insights** to see our main discoveries and understand delay patterns
    
    3. Use the **sidebar filters** on each page to customize your view
    
    ---
    
    ### 📊 Project Goals
    
    - Build powerful **visual analytics** to understand flight delay patterns
    
    - Develop **ML models** to predict delays and cancellations
    
    - Provide **explainable AI** insights for decision support
    """)

st.markdown("---")

# Navigation guide
st.markdown("## 📍 Navigate Using the Sidebar")

nav_col1, nav_col2, nav_col3, nav_col4 = st.columns(4)

with nav_col1:
    st.markdown("""
    ### 🏠 Home
    Project overview and quick statistics
    """)

with nav_col2:
    st.markdown("""
    ### 📊 Explore Data
    Interactive filters and customizable visualizations
    """)

with nav_col3:
    st.markdown("""
    ### 💡 Key Insights
    Curated storytelling with main discoveries
    """)

with nav_col4:
    st.markdown("""
    ### 🤖 Model & Explainability
    Predict delays and understand model decisions
    """)

st.markdown("---")

# Footer
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p>Built with Streamlit</p>
    <p>Visual Analytics Final Project | 2025</p>
</div>
""", unsafe_allow_html=True)

