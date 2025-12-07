"""
Model Explainability Page - SHAP-based feature importance and what-if analysis
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import joblib
import plotly.graph_objects as go
import plotly.express as px
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from utils import load_flight_data, prepare_flight_features

st.title("🔍 Model Explainability")
st.markdown(
    "Understand **what features drive predictions** and explore **how changes affect outcomes**."
)

st.markdown("---")


# -----------------------------
# Load model artifacts
# -----------------------------
@st.cache_resource
def load_model_artifacts():
    """Load model, preprocessors, metrics, surrogate tree, and SHAP data."""
    model_path = "models/arrival_delay_model.pkl"
    preprocessor_path = "models/preprocessors.pkl"
    metrics_path = "models/metrics.pkl"
    surrogate_path = "models/surrogate_tree.pkl"
    shap_path = "models/shap_data.pkl"
    
    if not os.path.exists(model_path):
        return None, None, None, None, None
    
    model = joblib.load(model_path)
    preprocessors = joblib.load(preprocessor_path) if os.path.exists(preprocessor_path) else None
    metrics = joblib.load(metrics_path) if os.path.exists(metrics_path) else None
    surrogate_tree = joblib.load(surrogate_path) if os.path.exists(surrogate_path) else None
    shap_data = joblib.load(shap_path) if os.path.exists(shap_path) else None
    
    return model, preprocessors, metrics, surrogate_tree, shap_data


with st.spinner("Loading model artifacts..."):
    model, preprocessors, metrics, surrogate_tree, shap_data = load_model_artifacts()
    
    if model is None:
        st.error("❌ Model not found! Please train the model first by running `python train_model.py`")
        st.stop()


# Check if SHAP is available
try:
    import shap
    shap_available = True
except ImportError:
    shap_available = False


# -----------------------------
# Tabs for different explainability views
# -----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌍 Global Importance",
    "🎯 Local Explanations", 
    "🔄 What-If Analysis",
    "🌳 Decision Tree",
    "📊 Feature Interactions"
])


# =============================
# TAB 1: Global Feature Importance
# =============================
with tab1:
    st.header("🌍 Global Feature Importance")
    st.markdown("""
    Which features matter most across **all predictions**? These charts show the overall 
    importance and impact of each feature on the model's decisions.
    """)
    
    if metrics['feature_importances'] is not None:
        feature_names = preprocessors['feature_cols']
        importances = metrics['feature_importances']
        
        # Create DataFrame
        feat_imp_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Bar chart of top features
            top_n = st.slider("Number of top features to show", 5, 25, 15, key="global_top_n")
            
            fig = px.bar(
                feat_imp_df.head(top_n),
                x='Importance',
                y='Feature',
                orientation='h',
                title=f"Top {top_n} Most Important Features (Random Forest)",
                color='Importance',
                color_continuous_scale='Viridis'
            )
            
            fig.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                height=600,
                xaxis_title="Importance Score",
                yaxis_title=""
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 📖 Top 10 Features")
            
            for idx, row in feat_imp_df.head(10).iterrows():
                pct = row['Importance'] * 100 / feat_imp_df['Importance'].sum()
                st.markdown(f"**{idx+1}. {row['Feature']}**")
                # Convert to Python float to avoid float32 type error
                progress_value = float(row['Importance'] / feat_imp_df['Importance'].max())
                st.progress(progress_value)
                st.caption(f"Score: {row['Importance']:.4f} ({pct:.1f}%)")
            
            st.markdown("---")
            st.markdown("""
            **Interpretation:**
            - Higher scores = more influence
            - These are averaged across all predictions
            - Model relies most on temporal and route features
            """)
        
        # SHAP Summary Plot (if available)
        if shap_available and shap_data is not None:
            st.markdown("---")
            st.subheader("📊 SHAP Feature Importance")
            st.markdown("""
            SHAP values provide a more nuanced view of feature importance by showing 
            both the **magnitude** and **direction** of each feature's impact.
            """)
            
            try:
                # Ensure data is in correct format
                shap_vals = shap_data['shap_values']
                X_sample = shap_data['X_test_sample']
                feature_names = shap_data['feature_names']
                
                # Convert to numpy if needed
                if isinstance(shap_vals, pd.DataFrame):
                    shap_vals = shap_vals.values
                if isinstance(X_sample, pd.DataFrame):
                    X_sample = X_sample.values
                
                # SHAP summary plot (bar)
                fig, ax = plt.subplots(figsize=(10, 8))
                shap.summary_plot(
                    shap_vals,
                    X_sample,
                    feature_names=feature_names,
                    show=False,
                    plot_type="bar"
                )
                st.pyplot(fig)
                plt.close()
                
                st.markdown("""
                **How to read this chart:**
                - Each bar represents average absolute SHAP value (impact magnitude)
                - Features at the top have the strongest influence on predictions
                """)
                
                # Detailed SHAP beeswarm plot
                with st.expander("🐝 See Detailed SHAP Summary (Beeswarm Plot)"):
                    fig, ax = plt.subplots(figsize=(10, 10))
                    shap.summary_plot(
                        shap_vals,
                        X_sample,
                        feature_names=feature_names,
                        show=False
                    )
                    st.pyplot(fig)
                    plt.close()
                    
                    st.markdown("""
                    **How to read this:**
                    - Each dot is a flight prediction
                    - Color: Red = high feature value, Blue = low feature value
                    - X-axis: Impact on prediction (left = pushes toward on-time, right = pushes toward delay)
                    - Features are ordered by importance (top = most important)
                    """)
            except Exception as e:
                st.error(f"Error displaying SHAP plots: {e}")
                st.info("SHAP data may need to be recomputed. Please re-run `python train_model.py`")
        else:
            st.info("💡 SHAP values not available. Re-train the model with SHAP support for advanced insights.")
    
    else:
        st.info("Feature importances not available for this model type.")


# =============================
# TAB 2: Local Explanations
# =============================
with tab2:
    st.header("🎯 Local Explanations - Individual Predictions")
    st.markdown("""
    Understand why the model made a specific prediction. Explain a prediction from the 
    **Predict Delays** page or select a test sample.
    """)
    
    if shap_available and shap_data is not None:
        # Check if we have a prediction from Predict Delays page
        use_custom_prediction = False
        custom_X = None
        custom_features = None
        
        if 'prediction_features' in st.session_state and st.session_state.prediction_features is not None:
            st.info("💡 Explaining your prediction from the Predict Delays page!")
            use_custom_prediction = True
            custom_features = st.session_state.prediction_features
            # We'll compute SHAP for this custom prediction below
        
        # Select a sample to explain
        n_samples = len(shap_data['X_test_sample'])
        
        if not use_custom_prediction:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                sample_idx = st.number_input(
                    f"Select test sample to explain (0-{n_samples-1})",
                    min_value=0,
                    max_value=n_samples-1,
                    value=0,
                    step=1
                )
            
            with col2:
                if st.button("🎲 Random Sample", key="random_sample"):
                    sample_idx = np.random.randint(0, n_samples)
                    st.rerun()
            
            # Get the sample
            X_sample = shap_data['X_test_sample'][sample_idx:sample_idx+1]
            
            # Extract SHAP values - ensure it's a 1D array
            shap_vals = shap_data['shap_values']
            if isinstance(shap_vals, np.ndarray):
                if shap_vals.ndim == 2:
                    shap_values_sample = shap_vals[sample_idx]
                else:
                    shap_values_sample = shap_vals[sample_idx] if shap_vals.ndim > 0 else shap_vals
            elif isinstance(shap_vals, (list, pd.Series)):
                shap_values_sample = np.array(shap_vals[sample_idx])
            else:
                shap_values_sample = np.array(shap_vals)[sample_idx]
            
            # Ensure it's 1D
            shap_values_sample = np.array(shap_values_sample).flatten()
            
            y_true = shap_data['y_test_sample'].iloc[sample_idx] if hasattr(shap_data['y_test_sample'], 'iloc') else shap_data['y_test_sample'][sample_idx]
            
            # Make prediction
            y_pred = model.predict(X_sample)[0]
            y_proba = model.predict_proba(X_sample)[0, 1]
        else:
            # Use custom prediction from Predict Delays page
            # We need to compute SHAP for this custom input
            explainer = shap_data['explainer']
            
            # Get preprocessed features from session state
            # The features should already be preprocessed
            if 'prediction_X_preprocessed' in st.session_state:
                X_sample = st.session_state.prediction_X_preprocessed
                # Ensure X_sample is 2D
                if isinstance(X_sample, np.ndarray):
                    if X_sample.ndim == 1:
                        X_sample = X_sample.reshape(1, -1)
                else:
                    X_sample = np.array(X_sample).reshape(1, -1)
                
                # Compute SHAP for this sample
                shap_values_custom = explainer.shap_values(X_sample, check_additivity=False)
                
                # Extract SHAP values for class 1 (delayed)
                if isinstance(shap_values_custom, list):
                    shap_values_sample = np.array(shap_values_custom[1][0]).flatten()  # Class 1, first sample
                else:
                    shap_values_sample = np.array(shap_values_custom[0]).flatten()
                
                y_pred = model.predict(X_sample)[0]
                y_proba = model.predict_proba(X_sample)[0, 1]
                y_true = None  # No ground truth for custom predictions
            else:
                st.warning("Preprocessed features not found. Please make a prediction on the Predict Delays page first.")
                st.stop()
        
        # Display prediction info
        st.markdown("---")
        
        if use_custom_prediction:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Predicted Label", "Delayed" if y_pred == 1 else "On-Time")
            with col2:
                st.metric("Delay Probability", f"{y_proba:.1%}")
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Actual Label", "Delayed" if y_true == 1 else "On-Time")
            with col2:
                st.metric("Predicted Label", "Delayed" if y_pred == 1 else "On-Time")
            with col3:
                st.metric("Delay Probability", f"{y_proba:.1%}")
        
        # SHAP Waterfall plot
        st.markdown("---")
        st.subheader("💧 SHAP Waterfall Plot")
        st.markdown("""
        This shows how each feature contributes to pushing the prediction from the 
        **base value** (average prediction) toward the **final prediction**.
        """)
        
        # Ensure shapes are correct
        shap_values_sample = np.array(shap_values_sample).flatten()
        feature_names = shap_data['feature_names']
        
        # Validate lengths match
        if len(shap_values_sample) != len(feature_names):
            st.error(f"Length mismatch: SHAP values ({len(shap_values_sample)}) vs features ({len(feature_names)})")
            st.info("This might indicate an issue with SHAP computation. Please check the model training.")
            st.stop()
        
        try:
            # Get X_sample values as 1D array
            if isinstance(X_sample, np.ndarray):
                X_sample_1d = X_sample[0] if X_sample.ndim > 1 else X_sample
            elif hasattr(X_sample, 'iloc'):
                X_sample_1d = X_sample.iloc[0].values
            else:
                X_sample_1d = np.array(X_sample).flatten()
            
            # Ensure X_sample_1d matches feature count
            X_sample_1d = np.array(X_sample_1d).flatten()[:len(feature_names)]
            
            # Create explanation object for waterfall plot
            explanation = shap.Explanation(
                values=shap_values_sample.reshape(1, -1),  # Shape: (1, n_features)
                base_values=np.array([shap_data['expected_value']]),  # Shape: (1,)
                data=X_sample_1d.reshape(1, -1),  # Shape: (1, n_features)
                feature_names=feature_names
            )
            
            fig, ax = plt.subplots(figsize=(10, 8))
            shap.plots.waterfall(explanation[0], show=False)  # Use explanation[0] for single sample
            st.pyplot(fig)
            plt.close()
        except Exception as e:
            st.warning(f"Could not create waterfall plot: {e}")
            st.info("Showing alternative bar chart visualization...")
            
            # Alternative: Bar plot of SHAP values
            # Ensure arrays are same length
            min_len = min(len(feature_names), len(shap_values_sample))
            shap_df = pd.DataFrame({
                'Feature': feature_names[:min_len],
                'SHAP Value': shap_values_sample[:min_len]
            }).sort_values('SHAP Value', key=abs, ascending=False).head(15)
            
            fig = px.bar(
                shap_df,
                x='SHAP Value',
                y='Feature',
                orientation='h',
                color='SHAP Value',
                color_continuous_scale='RdBu_r',
                title="Top 15 Feature Contributions (SHAP Values)"
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **How to read this:**
        - Starting point: E[f(X)] = base prediction (average across all data)
        - Each bar adds or subtracts from the prediction
        - Red bars push toward "Delayed", Blue bars push toward "On-Time"
        - Final value: f(x) = the actual prediction for this flight
        """)
        
        # Feature values table
        st.markdown("---")
        st.subheader("📋 Feature Values for This Sample")
        
        # Get feature values correctly
        if isinstance(X_sample, np.ndarray):
            X_values = X_sample[0] if X_sample.ndim > 1 else X_sample
        elif hasattr(X_sample, 'iloc'):
            X_values = X_sample.iloc[0].values
        else:
            X_values = np.array(X_sample).flatten()
        
        # Ensure all arrays are same length and 1D
        X_values = np.array(X_values).flatten()
        shap_values_sample = np.array(shap_values_sample).flatten()
        feature_names = shap_data['feature_names']
        
        # Match lengths
        min_len = min(len(feature_names), len(X_values), len(shap_values_sample))
        
        feature_df = pd.DataFrame({
            'Feature': feature_names[:min_len],
            'Value': X_values[:min_len],
            'SHAP Value': shap_values_sample[:min_len],
            'Impact': ['Increases Delay Risk' if sv > 0 else 'Decreases Delay Risk' 
                      for sv in shap_values_sample[:min_len]]
        }).sort_values('SHAP Value', key=abs, ascending=False)
        
        st.dataframe(feature_df.head(15), use_container_width=True)
        
        # Force plot (custom Plotly visualization)
        with st.expander("🔍 See Force Plot Visualization"):
            st.markdown("""
            This visualization shows how each feature pushes the prediction toward or away from delay.
            Features pushing right (red) increase delay risk, features pushing left (blue) decrease it.
            The final prediction is the sum of the base value and all feature contributions.
            """)
            
            try:
                # Ensure all arrays match
                force_shap = shap_values_sample[:min_len]
                force_X = X_values[:min_len]
                force_features = feature_names[:min_len]
                
                # Calculate cumulative sum for force plot effect
                base_value = shap_data['expected_value']
                sorted_indices = np.argsort(np.abs(force_shap))[::-1]  # Sort by absolute impact
                
                # Build cumulative contributions
                cumulative = base_value
                contributions = []
                feature_list = []
                colors = []
                
                for idx in sorted_indices[:15]:  # Top 15 features
                    shap_val = force_shap[idx]
                    cumulative += shap_val
                    contributions.append(cumulative)
                    feature_list.append(f"{force_features[idx]} = {force_X[idx]:.2f}")
                    # Red for positive (increases delay), blue for negative (decreases delay)
                    colors.append('red' if shap_val > 0 else 'blue')
                
                # Create custom force plot with Plotly
                fig = go.Figure()
                
                # Add base value line
                fig.add_hline(
                    y=base_value,
                    line_dash="dash",
                    line_color="gray",
                    annotation_text=f"Base Value: {base_value:.3f}",
                    annotation_position="right"
                )
                
                # Add feature contributions as bars
                for i, (contrib, feat_name, color) in enumerate(zip(contributions, feature_list, colors)):
                    prev_contrib = base_value if i == 0 else contributions[i-1]
                    fig.add_trace(go.Bar(
                        x=[i],
                        y=[contrib - prev_contrib],
                        base=prev_contrib,
                        name=feat_name,
                        marker_color=color,
                        text=[f"{feat_name}<br>Impact: {contrib - prev_contrib:+.3f}"],
                        textposition='auto',
                        hovertemplate=f"{feat_name}<br>Contribution: %{{y:+.3f}}<br>Cumulative: {contrib:.3f}<extra></extra>"
                    ))
                
                # Add final prediction line
                final_pred = contributions[-1] if contributions else base_value
                fig.add_hline(
                    y=final_pred,
                    line_dash="dot",
                    line_color="green",
                    line_width=3,
                    annotation_text=f"Final Prediction: {final_pred:.3f}",
                    annotation_position="left"
                )
                
                fig.update_layout(
                    title="Force Plot: Feature Contributions to Prediction",
                    xaxis_title="Features (sorted by impact)",
                    yaxis_title="Cumulative Prediction Value",
                    height=500,
                    showlegend=False,
                    xaxis=dict(
                        tickmode='array',
                        tickvals=list(range(len(feature_list))),
                        ticktext=[f.split('=')[0].strip() for f in feature_list],
                        tickangle=-45
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Alternative: horizontal bar chart showing contributions
                st.markdown("**Feature Contributions (Horizontal View):**")
                contrib_df = pd.DataFrame({
                    'Feature': [f.split('=')[0].strip() for f in feature_list],
                    'Contribution': [contributions[i] - (base_value if i == 0 else contributions[i-1]) 
                                    for i in range(len(contributions))],
                    'Cumulative': contributions
                })
                
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(
                    x=contrib_df['Contribution'],
                    y=contrib_df['Feature'],
                    orientation='h',
                    marker_color=['red' if c > 0 else 'blue' for c in contrib_df['Contribution']],
                    text=[f"{c:+.3f}" for c in contrib_df['Contribution']],
                    textposition='auto',
                    hovertemplate='%{y}<br>Contribution: %{x:+.3f}<extra></extra>'
                ))
                
                fig2.add_vline(
                    x=0,
                    line_dash="dash",
                    line_color="gray"
                )
                
                fig2.update_layout(
                    title="Feature Contributions to Prediction",
                    xaxis_title="SHAP Value (Contribution)",
                    yaxis_title="Feature",
                    height=400,
                    showlegend=False
                )
                
                st.plotly_chart(fig2, use_container_width=True)
                
            except Exception as e:
                st.warning(f"Could not create force plot visualization: {e}")
                st.info("Try using the waterfall plot above for a similar visualization.")
    
    else:
        st.warning("""
        ⚠️ **SHAP data not available**
        
        To enable local explanations:
        1. Install SHAP: `pip install shap`
        2. Re-train the model: `python train_model.py`
        
        This will compute SHAP values for test samples.
        """)


# =============================
# TAB 3: What-If Analysis
# =============================
with tab3:
    st.header("🔄 What-If Analysis")
    st.markdown("""
    Explore how changing features affects the prediction. Adjust values below and see 
    the impact on delay risk in real-time.
    """)
    
    st.info("""
    **📝 Note on CRSDepTime Format**: 
    CRSDepTime uses HHMM format (24-hour time):
    - Example: 1430 = 2:30 PM (14:30)
    - Example: 2359 = 11:59 PM (23:59)
    - Range: 1 to 2359
    """)
    
    if shap_available and shap_data is not None:
        # Load original data to get feature ranges
        @st.cache_data
        def get_original_feature_ranges():
            """Get original (unscaled) feature ranges from training data"""
            df_raw = load_flight_data()
            df = prepare_flight_features(df_raw)
            df = df[(df["Cancelled"] == False) & (df["Diverted"] == False)]
            return df
        
        # Get original data for feature ranges
        df_original = get_original_feature_ranges()
        
        # Start with a base sample
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🎛️ Adjust Features")
            
            # Get a random sample as baseline
            if 'whatif_base_idx' not in st.session_state:
                st.session_state.whatif_base_idx = 0
            
            if st.button("🎲 Load Random Flight", key="whatif_random"):
                st.session_state.whatif_base_idx = np.random.randint(0, len(shap_data['X_test_sample']))
                st.rerun()
            
            base_idx = st.session_state.whatif_base_idx
            X_base = shap_data['X_test_sample'][base_idx:base_idx+1].copy()
            
            # Get feature names and current values
            feature_names = shap_data['feature_names']
            
            # Create adjustable features (focus on most important ones)
            feat_imp_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': metrics['feature_importances']
            }).sort_values('Importance', ascending=False)
            
            # Filter out Month (keep only MonthName for readability)
            top_features = feat_imp_df.head(10)['Feature'].tolist()
            if 'Month' in top_features and 'MonthName' in top_features:
                top_features = [f for f in top_features if f != 'Month']  # Remove Month, keep MonthName
            elif 'Month' in top_features:
                # If Month is in top but MonthName is not, try to replace it
                if 'MonthName' in feature_names:
                    idx = top_features.index('Month')
                    top_features[idx] = 'MonthName'
            
            # Helper function to convert original value to encoded
            def convert_to_encoded(feat_name, original_val):
                """Convert original feature value to encoded/scaled value"""
                if feat_name in preprocessors['categorical_cols']:
                    # Categorical: use label encoder
                    le = preprocessors['label_encoders'][feat_name]
                    # Find closest category
                    if isinstance(original_val, str):
                        if original_val in le.classes_:
                            return le.transform([original_val])[0]
                        else:
                            # Return most common
                            return 0
                    else:
                        return original_val  # Already encoded
                else:
                    # Numerical: impute then scale
                    imputer = preprocessors['imputer_num']
                    scaler = preprocessors['scaler']
                    num_cols = preprocessors['numerical_cols']
                    
                    if feat_name not in num_cols:
                        return original_val
                    
                    feat_idx_in_num = num_cols.index(feat_name)
                    
                    # Get median for imputation
                    median_val = imputer.statistics_[feat_idx_in_num]
                    imputed_val = original_val if not np.isnan(original_val) else median_val
                    
                    # Scale
                    mean_val = scaler.mean_[feat_idx_in_num]
                    scale_val = scaler.scale_[feat_idx_in_num]
                    scaled_val = (imputed_val - mean_val) / scale_val
                    
                    return scaled_val
            
            # Helper function to convert encoded value to original
            def convert_to_original(feat_name, encoded_val):
                """Convert encoded/scaled value back to original scale"""
                if feat_name in preprocessors['categorical_cols']:
                    # Categorical: use label encoder
                    le = preprocessors['label_encoders'][feat_name]
                    if int(encoded_val) < len(le.classes_):
                        return le.inverse_transform([int(encoded_val)])[0]
                    else:
                        return le.classes_[0]
                else:
                    # Numerical: unscale then un-impute
                    scaler = preprocessors['scaler']
                    num_cols = preprocessors['numerical_cols']
                    
                    if feat_name not in num_cols:
                        return encoded_val
                    
                    feat_idx_in_num = num_cols.index(feat_name)
                    mean_val = scaler.mean_[feat_idx_in_num]
                    scale_val = scaler.scale_[feat_idx_in_num]
                    
                    # Unscale
                    original_val = encoded_val * scale_val + mean_val
                    return original_val
            
            # Create sliders/inputs for top features (using original values)
            adjusted_values_original = {}
            adjusted_values_encoded = {}
            
            st.markdown("**Adjust Top Features (Original Scale):**")
            
            for feat in top_features[:6]:  # Show top 6 adjustable features
                feat_idx = feature_names.index(feat)
                encoded_value = X_base[0, feat_idx]
                original_value = convert_to_original(feat, encoded_value)
                
                # Determine if categorical or numerical
                if feat in preprocessors['categorical_cols']:
                    # Categorical: show dropdown with original categories
                    le = preprocessors['label_encoders'][feat]
                    categories = le.classes_.tolist()
                    
                    current_cat = original_value if isinstance(original_value, str) else categories[0]
                    if current_cat not in categories:
                        current_cat = categories[0]
                    
                    new_cat = st.selectbox(
                        feat,
                        options=categories,
                        index=categories.index(current_cat),
                        help=f"Current: {current_cat}"
                    )
                    adjusted_values_original[feat_idx] = new_cat
                    adjusted_values_encoded[feat_idx] = convert_to_encoded(feat, new_cat)
                else:
                    # Numerical feature: get original range
                    if feat in df_original.columns:
                        feat_min_orig = float(df_original[feat].min())
                        feat_max_orig = float(df_original[feat].max())
                        feat_median_orig = float(df_original[feat].median())
                    else:
                        # Fallback: use encoded range converted back
                        feat_min_enc = float(shap_data['X_test_sample'][:, feat_idx].min())
                        feat_max_enc = float(shap_data['X_test_sample'][:, feat_idx].max())
                        feat_min_orig = convert_to_original(feat, feat_min_enc)
                        feat_max_orig = convert_to_original(feat, feat_max_enc)
                        feat_median_orig = (feat_min_orig + feat_max_orig) / 2
                    
                    # Special handling for CRSDepTime (must be int)
                    if feat == 'CRSDepTime':
                        # Convert to readable format for display
                        crs_hour = int(original_value) // 100
                        crs_min = int(original_value) % 100
                        readable_time = f"{crs_hour:02d}:{crs_min:02d}"
                        
                        new_value_orig = st.slider(
                            f"{feat} (Format: HHMM, e.g., 1430 = 14:30, 2359 = 23:59)",
                            min_value=int(feat_min_orig),
                            max_value=int(feat_max_orig),
                            value=int(original_value),
                            step=1,  # Must be int for int sliders
                            help=f"Current: {int(original_value)} ({readable_time})"
                        )
                    else:
                        # Determine step size based on feature type
                        if feat in ['Year']:
                            step = 1.0
                        elif feat in ['Month', 'DayofMonth', 'DayOfWeek', 'DepHour']:
                            step = 1.0
                        elif feat in ['Distance']:
                            step = 50.0
                        else:
                            step = (feat_max_orig - feat_min_orig) / 100
                        
                        new_value_orig = st.slider(
                            feat,
                            min_value=feat_min_orig,
                            max_value=feat_max_orig,
                            value=float(original_value),
                            step=step,
                            help=f"Original value: {original_value:.2f}"
                        )
                    adjusted_values_original[feat_idx] = new_value_orig
                    adjusted_values_encoded[feat_idx] = convert_to_encoded(feat, new_value_orig)
            
            # Apply adjustments (use encoded values for model)
            X_adjusted = X_base.copy()
            for feat_idx, new_val_encoded in adjusted_values_encoded.items():
                X_adjusted[0, feat_idx] = new_val_encoded
        
        with col2:
            st.subheader("📊 Prediction Results")
            
            # Original prediction
            y_proba_original = model.predict_proba(X_base)[0, 1]
            y_pred_original = model.predict(X_base)[0]
            
            # Adjusted prediction
            y_proba_adjusted = model.predict_proba(X_adjusted)[0, 1]
            y_pred_adjusted = model.predict(X_adjusted)[0]
            
            # Show comparison
            st.markdown("**Original Flight:**")
            st.metric("Delay Probability", f"{y_proba_original:.1%}")
            st.metric("Prediction", "Delayed" if y_pred_original == 1 else "On-Time")
            
            st.markdown("---")
            
            st.markdown("**After Adjustments:**")
            delta = (y_proba_adjusted - y_proba_original) * 100
            st.metric(
                "Delay Probability",
                f"{y_proba_adjusted:.1%}",
                delta=f"{delta:+.1f}%",
                delta_color="inverse"
            )
            st.metric("Prediction", "Delayed" if y_pred_adjusted == 1 else "On-Time")
            
            # Visual comparison
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=['Original', 'Adjusted'],
                y=[y_proba_original * 100, y_proba_adjusted * 100],
                marker_color=['lightblue', 'lightcoral' if y_proba_adjusted > y_proba_original else 'lightgreen'],
                text=[f'{y_proba_original:.1%}', f'{y_proba_adjusted:.1%}'],
                textposition='auto',
            ))
            
            fig.update_layout(
                title="Delay Probability Comparison",
                yaxis_title="Delay Risk (%)",
                height=300,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Show what changed
        st.markdown("---")
        st.subheader("📝 Changes Made")
        
        changes = []
        for feat_idx, new_val_orig in adjusted_values_original.items():
            feat_name = feature_names[feat_idx]
            old_val_encoded = X_base[0, feat_idx]
            old_val_orig = convert_to_original(feat_name, old_val_encoded)
            
            if new_val_orig != old_val_orig:
                if isinstance(new_val_orig, str):
                    changes.append({
                        'Feature': feat_name,
                        'Original Value': str(old_val_orig),
                        'New Value': str(new_val_orig),
                        'Change': f"{'Changed' if new_val_orig != old_val_orig else 'Unchanged'}"
                    })
                else:
                    changes.append({
                        'Feature': feat_name,
                        'Original Value': f"{old_val_orig:.2f}",
                        'New Value': f"{new_val_orig:.2f}",
                        'Change': f"{new_val_orig - old_val_orig:+.2f}"
                    })
        
        if changes:
            st.dataframe(pd.DataFrame(changes), use_container_width=True)
        else:
            st.info("No changes made yet. Adjust the sliders above to see impact.")
        
        # Counterfactual suggestions
        with st.expander("💡 Suggestions to Reduce Delay Risk"):
            st.markdown("""
            **Based on feature importance, you could:**
            
            1. **Choose different departure time**: Morning flights tend to be more reliable
            2. **Select different route**: Some airports have better on-time performance
            3. **Pick different airline**: Airlines vary in operational efficiency
            4. **Fly on different days**: Weekday patterns differ from weekends
            5. **Consider shorter routes**: Longer flights have more delay opportunities
            
            Use the sliders above to test these scenarios!
            """)
    
    else:
        st.warning("⚠️ SHAP data required for What-If Analysis. Please re-train the model with SHAP support.")


# =============================
# TAB 4: Decision Tree Visualization
# =============================
with tab4:
    st.header("🌳 Decision Tree Surrogate Model")
    st.markdown("""
    A shallow decision tree (max_depth=3) trained to approximate the main model's behavior. 
    While the actual model uses XGBoost (an ensemble of many trees), this surrogate tree 
    provides a simplified, interpretable view of the key decision rules.
    
    **Note:** This is a simplified approximation - the full XGBoost model is more complex 
    and accurate, but harder to visualize. Use SHAP values for precise feature importance.
    """)
    
    if surrogate_tree is not None:
        # Get feature names
        feature_names = preprocessors['feature_cols']
        class_names = metrics['class_names']
        
        # Create figure
        fig, ax = plt.subplots(figsize=(25, 12))
        plot_tree(
            surrogate_tree,
            feature_names=feature_names,
            class_names=class_names,
            filled=True,
            rounded=True,
            fontsize=11,
            ax=ax,
            impurity=True,
            proportion=True
        )
        plt.tight_layout()
        
        st.pyplot(fig)
        plt.close()
        
        st.markdown("""
        **How to Read This Tree:**
        
        - **Root (top):** First decision
        - **Branches:** Decision paths based on thresholds
        - **Leaves (bottom):** Final predictions
        - **Color:** Blue = On-Time, Orange = Delayed (intensity = confidence)
        - **Samples:** Proportion of training data reaching node
        - **Value:** [On-Time %, Delayed %]
        - **Gini:** Impurity (0 = pure, 0.5 = mixed)
        
        **Note:** This is a simplified model (depth=3). The actual Random Forest uses 500 deeper trees.
        """)
        
        # Tree statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Tree Depth", surrogate_tree.get_depth())
        with col2:
            st.metric("Number of Leaves", surrogate_tree.get_n_leaves())
        with col3:
            st.metric("Number of Nodes", surrogate_tree.tree_.node_count)
        
        # Feature splits
        st.markdown("---")
        st.subheader("🔀 Features Used in Tree Splits")
        
        surr_importances = surrogate_tree.feature_importances_
        surr_feat_imp = pd.DataFrame({
            'Feature': feature_names,
            'Importance': surr_importances
        }).sort_values('Importance', ascending=False)
        
        surr_feat_imp = surr_feat_imp[surr_feat_imp['Importance'] > 0]
        
        if not surr_feat_imp.empty:
            fig = px.bar(
                surr_feat_imp,
                x='Importance',
                y='Feature',
                orientation='h',
                title="Feature Importance in Surrogate Tree"
            )
            fig.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("Surrogate decision tree not available.")


# =============================
# TAB 5: Feature Interactions
# =============================
with tab5:
    st.header("📊 Feature Interactions")
    st.markdown("""
    How do pairs of features interact to influence predictions? Dependence plots show 
    the relationship between feature values and their impact on predictions.
    """)
    
    if shap_available and shap_data is not None:
        # SHAP dependence plots
        st.subheader("🔗 SHAP Dependence Plots")
        
        feature_names = shap_data['feature_names']
        
        # Select features to explore
        col1, col2 = st.columns(2)
        
        with col1:
            main_feature = st.selectbox(
                "Select main feature",
                options=feature_names,
                index=0
            )
        
        with col2:
            interaction_feature = st.selectbox(
                "Select interaction feature (or 'auto')",
                options=['auto'] + feature_names,
                index=0
            )
        
        # Create dependence plot
        try:
            shap_vals = shap_data['shap_values']
            X_sample = shap_data['X_test_sample']
            
            # Convert to numpy if needed
            if isinstance(shap_vals, pd.DataFrame):
                shap_vals = shap_vals.values
            elif not isinstance(shap_vals, np.ndarray):
                shap_vals = np.array(shap_vals)
            
            if isinstance(X_sample, pd.DataFrame):
                X_sample = X_sample.values
            elif not isinstance(X_sample, np.ndarray):
                X_sample = np.array(X_sample)
            
            # Ensure correct dimensions
            # shap_vals should be (n_samples, n_features)
            # X_sample should be (n_samples, n_features)
            if shap_vals.ndim == 1:
                # If 1D, assume it's for one sample, reshape to (1, n_features)
                shap_vals = shap_vals.reshape(1, -1)
            elif shap_vals.ndim > 2:
                # If more than 2D, flatten extra dimensions
                shap_vals = shap_vals.reshape(shap_vals.shape[0], -1)
            
            if X_sample.ndim == 1:
                X_sample = X_sample.reshape(1, -1)
            elif X_sample.ndim > 2:
                X_sample = X_sample.reshape(X_sample.shape[0], -1)
            
            # Ensure same number of samples
            n_samples = min(shap_vals.shape[0], X_sample.shape[0])
            shap_vals = shap_vals[:n_samples]
            X_sample = X_sample[:n_samples]
            
            main_idx = feature_names.index(main_feature)
            
            # Ensure main_idx is within bounds
            if main_idx >= shap_vals.shape[1] or main_idx >= X_sample.shape[1]:
                raise ValueError(f"Feature index {main_idx} out of bounds. SHAP shape: {shap_vals.shape}, X shape: {X_sample.shape}")
            
            # Create Plotly scatter plot (more reliable than SHAP plots in Streamlit)
            fig = go.Figure()
            
            if interaction_feature != 'auto':
                interaction_idx = feature_names.index(interaction_feature)
                
                if interaction_idx >= X_sample.shape[1]:
                    raise ValueError(f"Interaction feature index {interaction_idx} out of bounds")
                
                # Create scatter plot with color coding
                fig.add_trace(go.Scatter(
                    x=X_sample[:, main_idx],
                    y=shap_vals[:, main_idx],
                    mode='markers',
                    marker=dict(
                        color=X_sample[:, interaction_idx],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title=interaction_feature),
                        size=6,
                        opacity=0.7
                    ),
                    hovertemplate=f'{main_feature}: %{{x:.2f}}<br>SHAP Value: %{{y:.3f}}<br>{interaction_feature}: %{{marker.color:.2f}}<extra></extra>',
                    name='Samples'
                ))
                
                # Add trend line
                z = np.polyfit(X_sample[:, main_idx], shap_vals[:, main_idx], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(X_sample[:, main_idx].min(), X_sample[:, main_idx].max(), 100)
                fig.add_trace(go.Scatter(
                    x=x_trend,
                    y=p(x_trend),
                    mode='lines',
                    name='Trend',
                    line=dict(color='red', dash='dash', width=2)
                ))
            else:
                # No interaction feature
                fig.add_trace(go.Scatter(
                    x=X_sample[:, main_idx],
                    y=shap_vals[:, main_idx],
                    mode='markers',
                    marker=dict(size=6, opacity=0.7, color='blue'),
                    hovertemplate=f'{main_feature}: %{{x:.2f}}<br>SHAP Value: %{{y:.3f}}<extra></extra>',
                    name='Samples'
                ))
                
                # Add trend line
                z = np.polyfit(X_sample[:, main_idx], shap_vals[:, main_idx], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(X_sample[:, main_idx].min(), X_sample[:, main_idx].max(), 100)
                fig.add_trace(go.Scatter(
                    x=x_trend,
                    y=p(x_trend),
                    mode='lines',
                    name='Trend',
                    line=dict(color='red', dash='dash', width=2)
                ))
            
            fig.update_layout(
                title=f"SHAP Dependence Plot: {main_feature}" + (f" (colored by {interaction_feature})" if interaction_feature != 'auto' else ""),
                xaxis_title=main_feature,
                yaxis_title="SHAP Value (Impact on Prediction)",
                height=500,
                hovermode='closest'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error creating dependence plot: {e}")
            import traceback
            with st.expander("Error Details"):
                st.code(traceback.format_exc())
            st.info("Please ensure SHAP data is properly computed and has correct dimensions.")
        
        st.markdown("""
        **How to read this:**
        - **X-axis:** Feature value
        - **Y-axis:** SHAP value (impact on prediction)
        - **Color:** Interaction feature value (if not auto)
        - **Pattern:** Shows how the feature's impact varies with its value
        
        **Insights:**
        - Horizontal pattern = constant impact regardless of value
        - Upward trend = higher values increase delay risk
        - Downward trend = higher values decrease delay risk
        - Color variation = interaction with another feature
        """)
        
        # Feature correlation heatmap
        st.markdown("---")
        st.subheader("🔥 Feature Correlation Heatmap")
        
        # Calculate correlations
        X_sample = shap_data['X_test_sample']
        if isinstance(X_sample, pd.DataFrame):
            X_df = X_sample
        else:
            X_df = pd.DataFrame(X_sample, columns=feature_names)
        
        # Select top features
        top_n = 10
        top_features = pd.DataFrame({
            'Feature': feature_names,
            'Importance': metrics['feature_importances']
        }).sort_values('Importance', ascending=False).head(top_n)['Feature'].tolist()
        
        corr_matrix = X_df[top_features].corr()
        
        fig = px.imshow(
            corr_matrix,
            labels=dict(color="Correlation"),
            x=top_features,
            y=top_features,
            color_continuous_scale='RdBu_r',
            zmin=-1,
            zmax=1,
            title=f"Correlation Matrix (Top {top_n} Features)"
        )
        
        fig.update_layout(height=600)
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **Insights:**
        - Red = positive correlation (features move together)
        - Blue = negative correlation (features move opposite)
        - White = no correlation
        - Strong correlations suggest features provide similar information
        """)
    
    else:
        st.warning("⚠️ SHAP data required for interaction analysis.")


# Footer
st.markdown("---")
st.info("""
🔗 **See model performance metrics** on the **🤖 Predict Delays** page.

💡 **Tip:** Use the What-If Analysis tab to explore how to reduce delay risk!
""")
