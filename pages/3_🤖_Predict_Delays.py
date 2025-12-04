"""
Predict Delays Page - Interactive flight delay prediction with model performance
"""

import datetime as dt
import os

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import plotly.graph_objects as go
import plotly.express as px

from utils import load_flight_data, prepare_flight_features

st.title("🤖 Flight Delay Prediction")
st.markdown(
    "Predict flight delays and explore model performance metrics."
)

st.markdown("---")


# -----------------------------
# Helpers: load model & metadata
# -----------------------------
@st.cache_resource
def load_trained_model():
    """Load the trained model and preprocessors from disk."""
    model_path = "models/arrival_delay_model.pkl"
    preprocessor_path = "models/preprocessors.pkl"
    metrics_path = "models/metrics.pkl"
    
    if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
        return None, None, None
    
    model = joblib.load(model_path)
    preprocessors = joblib.load(preprocessor_path)
    metrics = joblib.load(metrics_path) if os.path.exists(metrics_path) else None
    return model, preprocessors, metrics


@st.cache_data
def load_metadata(sample_rows_per_file: int = 50_000, random_state: int = 42):
    """Load a sample of data to populate dropdowns (airlines, airports)"""
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


def get_dep_time_blk(hour: int) -> str:
    """Get departure time block"""
    return f"{hour:02d}00-{hour:02d}59"


def get_distance_group(distance: float) -> int:
    """Get distance group"""
    if distance < 250:
        return 1
    elif distance < 500:
        return 2
    elif distance < 750:
        return 3
    elif distance < 1000:
        return 4
    elif distance < 1250:
        return 5
    elif distance < 1500:
        return 6
    elif distance < 1750:
        return 7
    elif distance < 2000:
        return 8
    elif distance < 2250:
        return 9
    elif distance < 2500:
        return 10
    else:
        return 11


# -----------------------------
# Load resources
# -----------------------------
with st.spinner("Loading model and metadata..."):
    model, preprocessors, metrics = load_trained_model()
    
    if model is None:
        st.error("❌ Model not found! Please train the model first by running `python train_model.py`")
        st.stop()
    
    df_meta, airlines, origins, dests, route_distance = load_metadata()


# -----------------------------
# Tabs for different sections
# -----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🚀 Make a Prediction",
    "📊 Model Performance", 
    "🎯 Confusion Matrix",
    "📈 ROC Curve",
    "⚖️ Class Balance"
])


# =============================
# TAB 1: Make a Prediction
# =============================
with tab1:
    st.header("Make a Flight Delay Prediction")
    
    left_col, right_col = st.columns([1.3, 1])

    with left_col:
        st.subheader("✏️ Flight Details")

        # 1) Date
        flight_date = st.date_input(
            "Flight date",
            value=dt.date(2022, 4, 4),
            min_value=dt.date(2018, 1, 1),
            max_value=dt.date(2022, 12, 31),
        )
        day_of_week = flight_date.weekday()
        day_of_week_name = flight_date.strftime("%A")
        month_name = flight_date.strftime("%B")
        month = flight_date.month
        year = flight_date.year
        quarter = (month - 1) // 3 + 1
        day_of_month = flight_date.day

        # 2) Airline
        airline = st.selectbox("Airline", options=airlines)

        # 3) Origin & destination
        origin = st.selectbox("Origin airport (IATA)", options=origins, 
                            index=origins.index("DEN") if "DEN" in origins else 0)
        dest = st.selectbox("Destination airport (IATA)", options=dests, 
                          index=dests.index("LAX") if "LAX" in dests else 0)

        # 4) Departure time
        dep_time = st.time_input("Scheduled departure time", value=dt.time(12, 0))
        dep_hour = dep_time.hour
        dep_minute = dep_time.minute
        crs_dep_time = dep_hour * 100 + dep_minute
        dep_time_of_day = get_dep_time_of_day(dep_hour)
        dep_time_blk = get_dep_time_blk(dep_hour)

        # 5) Distance
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
            default_distance = 1000.0

        distance = st.number_input(
            "Route distance (miles)",
            min_value=50.0,
            max_value=5000.0,
            value=round(default_distance, 1),
            step=10.0,
        )
        distance_group = get_distance_group(distance)

        st.markdown("")


    with right_col:
        st.subheader("📌 Prediction")

        st.markdown(
            "The model predicts whether the flight will arrive **>15 minutes late** "
            "using only pre-departure information."
        )

        predict_button = st.button("🚀 Predict Delay Risk", type="primary")

        result_placeholder = st.empty()
        gauge_placeholder = st.empty()


    # Make prediction
    if predict_button:
        # Build input dataframe
        feature_cols = preprocessors['feature_cols']
        
        input_data = {
            'Airline': airline,
            'Origin': origin,
            'Dest': dest,
            'DepTimeOfDay': dep_time_of_day,
            'DayOfWeekName': day_of_week_name,
            'MonthName': month_name,
            'DepTimeBlk': dep_time_blk,
            'DayOfWeek': day_of_week,
            'DayofMonth': day_of_month,
            'DepHour': dep_hour,
            'Month': month,
            'Quarter': quarter,
            'Year': year,
            'Distance': distance,
            'DistanceGroup': distance_group,
            'CRSDepTime': crs_dep_time,
        }
        
        X_input = pd.DataFrame([{col: input_data.get(col, 0) for col in feature_cols}])

        with st.spinner("Running model prediction..."):
            # Preprocess
            categorical_cols = preprocessors['categorical_cols']
            numerical_cols = preprocessors['numerical_cols']
            label_encoders = preprocessors['label_encoders']
            imputer_num = preprocessors['imputer_num']
            scaler = preprocessors['scaler']
            
            # Handle categoricals
            for col in categorical_cols:
                if col in X_input.columns:
                    if isinstance(X_input[col].dtype, pd.CategoricalDtype):
                        X_input[col] = X_input[col].astype(str)
                    
                    X_input[col] = X_input[col].fillna('Unknown').astype(str)
                    
                    le = label_encoders[col]
                    value = X_input[col].iloc[0]
                    if value in le.classes_:
                        X_input[col] = le.transform([value])
                    else:
                        X_input[col] = le.transform([le.classes_[0]])
            
            # Handle numericals
            if len(numerical_cols) > 0 and imputer_num is not None:
                X_input[numerical_cols] = imputer_num.transform(X_input[numerical_cols])
                X_input[numerical_cols] = scaler.transform(X_input[numerical_cols])
            
            # Convert to numeric
            for col in X_input.columns:
                X_input[col] = pd.to_numeric(X_input[col], errors='coerce')
            X_input = X_input.fillna(0)
            
            X_input_array = X_input.values.astype(float)
            
            # Predict
            proba_delay = float(model.predict_proba(X_input_array)[0, 1])
            pred_label = int(model.predict(X_input_array)[0])
            
            # Save to session state for Explainability page
            st.session_state.prediction_features = {
                'airline': airline,
                'origin': origin,
                'dest': dest,
                'dep_time_of_day': dep_time_of_day,
                'day_of_week_name': day_of_week_name,
                'month_name': month_name,
                'dep_time_blk': dep_time_blk,
                'day_of_week': day_of_week,
                'day_of_month': day_of_month,
                'dep_hour': dep_hour,
                'month': month,
                'quarter': quarter,
                'year': year,
                'distance': distance,
                'distance_group': distance_group,
                'crs_dep_time': crs_dep_time,
            }
            st.session_state.prediction_X_preprocessed = X_input_array
            st.session_state.prediction_result = {
                'proba_delay': proba_delay,
                'pred_label': pred_label
            }

        # Interpret
        risk_pct = proba_delay * 100

        if risk_pct < 20:
            risk_level = "Low"
            risk_color = "🟢"
            gauge_color = "green"
        elif risk_pct < 40:
            risk_level = "Medium"
            risk_color = "🟡"
            gauge_color = "yellow"
        else:
            risk_level = "High"
            risk_color = "🔴"
            gauge_color = "red"

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk_pct,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Delay Risk %", 'font': {'size': 24}},
            delta={'reference': 17.1, 'increasing': {'color': "red"}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': gauge_color},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 20], 'color': 'lightgreen'},
                    {'range': [20, 40], 'color': 'lightyellow'},
                    {'range': [40, 100], 'color': 'lightcoral'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
        
        gauge_placeholder.plotly_chart(fig, use_container_width=True)

        result_placeholder.markdown(
            f"""
            ### {risk_color} Predicted Delay Risk: **{risk_pct:.1f}%**

            - **Model Prediction:** {"⚠️ Delayed (>15 min)" if pred_label == 1 else "✅ On-Time"}
            - **Risk Level:** **{risk_level}**
            
            ---
            
            #### ℹ️ Interpretation
            
            - The baseline delay rate is **17.1%** - your flight is compared to this
            - The gauge shows your flight's delay risk
            - For detailed explanation of what drives this prediction, visit **🔍 Model Explainability**
            """
        )
        
        # Add button to explain this prediction
        st.markdown("---")
        if st.button("🔍 Explain This Prediction", type="secondary", use_container_width=True):
            st.info("💡 Navigate to the **🔍 Model Explainability** page to see detailed SHAP explanations for this prediction!")

    else:
        st.info("👈 Fill in the flight details and click **Predict Delay Risk**")


# =============================
# TAB 2: Model Performance
# =============================
with tab2:
    st.header("📊 Model Performance Overview")
    
    if metrics is None:
        st.warning("Metrics not available. Please re-train the model.")
    else:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.1%}")
            
        with col2:
            st.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
            
        with col3:
            precision_delayed = metrics['classification_report']['1']['precision']
            st.metric("Precision (Delayed)", f"{precision_delayed:.1%}")
            
        with col4:
            recall_delayed = metrics['classification_report']['1']['recall']
            st.metric("Recall (Delayed)", f"{recall_delayed:.1%}")
        
        st.markdown("---")
        
        # Classification report
        st.subheader("📋 Detailed Classification Report")
        
        report = metrics['classification_report']
        
        report_df = pd.DataFrame({
            'Class': ['On-Time (0)', 'Delayed (1)'],
            'Precision': [report['0']['precision'], report['1']['precision']],
            'Recall': [report['0']['recall'], report['1']['recall']],
            'F1-Score': [report['0']['f1-score'], report['1']['f1-score']],
            'Support': [int(report['0']['support']), int(report['1']['support'])]
        })
        
        st.dataframe(report_df.style.format({
            'Precision': '{:.1%}',
            'Recall': '{:.1%}',
            'F1-Score': '{:.3f}',
            'Support': '{:,}'
        }), use_container_width=True)
        
        st.markdown("---")
        
        # Training info
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 📖 Metric Definitions
            
            - **Accuracy**: Overall correct predictions
            - **Precision**: Of predicted delays, how many were actual delays
            - **Recall**: Of actual delays, how many did we catch
            - **F1-Score**: Harmonic mean of precision and recall
            - **ROC-AUC**: Model's ability to distinguish classes
            """)
        
        with col2:
            st.markdown(f"""
            ### 📈 Dataset Info
            
            - **Training Samples**: {metrics['n_train']:,}
            - **Test Samples**: {metrics['n_test']:,}
            - **Delay Rate (Train)**: {metrics['delay_rate_train']:.1%}
            - **Delay Rate (Test)**: {metrics['delay_rate_test']:.1%}
            - **Model Type**: Random Forest (500 trees)
            """)


# =============================
# TAB 3: Confusion Matrix
# =============================
with tab3:
    st.header("🎯 Confusion Matrix Analysis")
    
    if metrics is None:
        st.warning("Metrics not available.")
    else:
        col_cm, col_exp = st.columns([1.2, 1])

        with col_cm:
            cm = metrics['confusion_matrix']
            
            # Annotated heatmap
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Predicted On-Time', 'Predicted Delayed'],
                y=['Actually On-Time', 'Actually Delayed'],
                text=[[f'{cm[0,0]:,}<br>{cm[0,0]/cm.sum()*100:.1f}%',
                       f'{cm[0,1]:,}<br>{cm[0,1]/cm.sum()*100:.1f}%'],
                      [f'{cm[1,0]:,}<br>{cm[1,0]/cm.sum()*100:.1f}%',
                       f'{cm[1,1]:,}<br>{cm[1,1]/cm.sum()*100:.1f}%']],
                texttemplate='%{text}',
                textfont={"size": 16},
                colorscale='Blues',
                showscale=False,
            ))
            
            fig.update_layout(
                title="Confusion Matrix (Test Set)",
                xaxis_title="Predicted Label",
                yaxis_title="Actual Label",
                height=450,
                font=dict(size=14)
            )
            
            st.plotly_chart(fig, use_container_width=True)

        with col_exp:
            st.markdown("### 📖 Understanding the Matrix")
            
            st.markdown(f"""
            **Total Test Samples:** {cm.sum():,}
            
            **Breakdown:**
            - ✅ **True Negatives ({cm[0,0]:,}):** Correctly predicted on-time
            - ✅ **True Positives ({cm[1,1]:,}):** Correctly predicted delayed
            - ❌ **False Positives ({cm[0,1]:,}):** Predicted delay, but on-time
            - ❌ **False Negatives ({cm[1,0]:,}):** Predicted on-time, but delayed
            
            **Key Insight:**
            The model is conservative about predicting delays. It catches {cm[1,1]/(cm[1,0]+cm[1,1])*100:.1f}% 
            of actual delays (recall) but has more false negatives than false positives.
            """)
            
            # Error types
            st.markdown("### ⚠️ Error Analysis")
            
            false_positive_rate = cm[0,1] / (cm[0,0] + cm[0,1])
            false_negative_rate = cm[1,0] / (cm[1,0] + cm[1,1])
            
            st.markdown(f"""
            - **False Positive Rate**: {false_positive_rate:.1%}
            - **False Negative Rate**: {false_negative_rate:.1%}
            
            The model makes more "false negative" errors (missing delays) than 
            "false positive" errors (false alarms).
            """)


# =============================
# TAB 4: ROC Curve
# =============================
with tab4:
    st.header("📈 ROC Curve Analysis")
    
    if metrics is None:
        st.warning("Metrics not available.")
    else:
        col_roc, col_roc_exp = st.columns([1.4, 1])

        with col_roc:
            fpr = metrics['fpr']
            tpr = metrics['tpr']
            roc_auc = metrics['roc_auc']
            
            fig = go.Figure()
            
            # ROC curve
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'ROC Curve (AUC = {roc_auc:.3f})',
                line=dict(color='blue', width=3),
                fill='tozeroy',
                fillcolor='rgba(0, 100, 255, 0.2)',
                hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>'
            ))
            
            # Random baseline
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Random Baseline (AUC = 0.5)',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            fig.update_layout(
                title="Receiver Operating Characteristic (ROC) Curve",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate (Recall)",
                height=500,
                showlegend=True,
                hovermode='closest'
            )
            
            fig.update_xaxes(range=[0, 1])
            fig.update_yaxes(range=[0, 1])
            
            st.plotly_chart(fig, use_container_width=True)

        with col_roc_exp:
            st.markdown("### 📖 Understanding ROC-AUC")
            
            st.markdown(f"""
            **ROC-AUC Score:** {roc_auc:.3f}
            
            The ROC curve shows the trade-off between:
            - **True Positive Rate (Recall):** % of actual delays we catch
            - **False Positive Rate:** % of on-time flights wrongly flagged
            
            **AUC Interpretation:**
            - 0.5: Random guessing
            - 0.7: Acceptable
            - 0.8: Good
            - 0.9: Excellent
            - 1.0: Perfect
            
            **Our Score ({roc_auc:.3f}):** 
            The model is moderately good at distinguishing between delayed and on-time flights.
            
            **Why Not Higher?**
            - Using only pre-departure info
            - Flight delays have inherent unpredictability
            - Weather, air traffic, mechanical issues are not in our data
            """)
            
            st.markdown("---")
            
            st.markdown("""
            ### 🎚️ Threshold Tuning
            
            The ROC curve shows performance at all possible classification thresholds. 
            Currently we use 0.5 (default), but we could:
            
            - **Lower threshold** → More delays caught (higher recall), but more false alarms
            - **Raise threshold** → Fewer false alarms, but miss more actual delays
            
            The optimal threshold depends on the cost of false positives vs false negatives.
            """)


# =============================
# TAB 5: Class Balance
# =============================
with tab5:
    st.header("⚖️ Class Imbalance Analysis")
    
    if metrics is None:
        st.warning("Metrics not available.")
    else:
        cm = metrics['confusion_matrix']
        delay_rate = metrics['delay_rate_test']
        
        st.markdown(f"""
        The dataset is **highly imbalanced**, with only **{delay_rate:.1%}** of flights delayed 
        (more than 15 minutes). This creates challenges for the model.
        """)
        
        col_dist, col_exp_dist = st.columns([1, 1])

        with col_dist:
            # Class distribution
            class_counts = [cm[0].sum(), cm[1].sum()]
            class_labels = ['On-Time Flights', 'Delayed Flights']
            
            fig = go.Figure(data=[
                go.Bar(
                    x=class_labels,
                    y=class_counts,
                    text=[f'{count:,}<br>({count/sum(class_counts)*100:.1f}%)' 
                          for count in class_counts],
                    textposition='auto',
                    marker_color=['lightgreen', 'lightcoral'],
                    textfont=dict(size=14)
                )
            ])
            
            fig.update_layout(
                title="Class Distribution in Test Set",
                yaxis_title="Number of Flights",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)

        with col_exp_dist:
            st.markdown("### 🎯 Impact of Imbalance")
            
            st.markdown(f"""
            **Class Ratio:** ~{(1-delay_rate)/delay_rate:.1f}:1 
            (On-Time : Delayed)
            
            **Effects on Model:**
            - Model sees {(1-delay_rate)/delay_rate:.1f}x more on-time flights
            - Learns to be "conservative" about predicting delays
            - Results in **lower recall** for minority class (delays)
            - Higher precision for majority class (on-time)
            
            **Trade-offs:**
            - ✅ Good at identifying on-time flights (86% precision)
            - ⚠️ Misses some delays (30% recall for delayed)
            - ⚖️ Balances false alarms vs missed delays
            """)
        
        st.markdown("---")
        
        # Detailed metrics by class
        st.subheader("📊 Per-Class Performance")
        
        report = metrics['classification_report']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### ✅ On-Time Flights (Majority Class)")
            st.metric("Precision", f"{report['0']['precision']:.1%}")
            st.metric("Recall", f"{report['0']['recall']:.1%}")
            st.metric("F1-Score", f"{report['0']['f1-score']:.3f}")
            st.metric("Support", f"{int(report['0']['support']):,}")
            
            st.markdown("""
            The model performs well on the majority class with high precision and recall.
            """)
        
        with col2:
            st.markdown("### ⚠️ Delayed Flights (Minority Class)")
            st.metric("Precision", f"{report['1']['precision']:.1%}")
            st.metric("Recall", f"{report['1']['recall']:.1%}")
            st.metric("F1-Score", f"{report['1']['f1-score']:.3f}")
            st.metric("Support", f"{int(report['1']['support']):,}")
            
            st.markdown("""
            Lower recall indicates the model misses many delays due to class imbalance.
            """)
        
        st.markdown("---")
        
        # Solutions
        with st.expander("🔧 Handling Class Imbalance"):
            st.markdown("""
            **Current Approach:**
            - Using `class_weight='balanced_subsample'` in Random Forest
            - This gives more weight to minority class samples during training
            
            **Alternative Approaches:**
            
            1. **Resampling:**
               - SMOTE: Generate synthetic minority samples
               - Undersampling: Reduce majority class
               - Hybrid: Combine both approaches
            
            2. **Algorithm-Level:**
               - Adjust classification threshold (currently 0.5)
               - Use cost-sensitive learning
               - Focal loss for neural networks
            
            3. **Evaluation:**
               - Focus on metrics beyond accuracy
               - Use F1-score, ROC-AUC, precision-recall curves
               - Consider business costs of false positives vs false negatives
            
            4. **Ensemble Methods:**
               - Balanced Random Forest
               - Easy Ensemble / Balanced Bagging
               - Combination of multiple models
            """)

st.markdown("---")

# Model info footer
with st.expander("ℹ️ About the Model"):
    st.markdown("""
    **Model Type:** Random Forest Classifier (500 trees, max_depth=20)
    
    **Features Used (Pre-Departure Only):**
    - **Categorical**: Airline, Origin, Destination, Time blocks, Day names
    - **Numerical**: Day of week/month, Hour, Distance, Scheduled departure time
    
    **Performance Summary:**
    - Overall Accuracy: ~79%
    - ROC-AUC: ~0.68
    - Balanced for class imbalance
    
    **Important Notes:**
    - Uses ONLY pre-departure information
    - Does NOT use actual departure delay, taxi time, or flight time
    - Realistic for real-world prediction scenarios
    - Cannot predict unexpected events (weather, mechanicals)
    """)
