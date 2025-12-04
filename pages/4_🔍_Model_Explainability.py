"""
Model Explainability Page - SHAP-based feature importance and what-if analysis
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import plotly.graph_objects as go
import plotly.express as px
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

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
                st.progress(row['Importance'] / feat_imp_df['Importance'].max())
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
        
        # Force plot (alternative visualization)
        with st.expander("🔍 See Force Plot"):
            st.markdown("""
            Force plot shows the same information in a different format - features pushing 
            right (red) increase delay risk, features pushing left (blue) decrease it.
            """)
            
            try:
                # Ensure all arrays match
                force_shap = shap_values_sample[:min_len]
                force_X = X_values[:min_len]
                force_features = feature_names[:min_len]
                
                fig, ax = plt.subplots(figsize=(14, 3))
                shap.force_plot(
                    shap_data['expected_value'],
                    force_shap,
                    force_X,
                    feature_names=force_features,
                    show=False,
                    matplotlib=True
                )
                st.pyplot(fig)
                plt.close()
            except Exception as e:
                st.warning(f"Could not display force plot: {e}")
    
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
    
    if shap_available and shap_data is not None:
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
            
            top_features = feat_imp_df.head(10)['Feature'].tolist()
            
            # Create sliders/inputs for top features
            adjusted_values = {}
            
            st.markdown("**Adjust Top Features:**")
            
            for feat in top_features[:6]:  # Show top 6 adjustable features
                feat_idx = feature_names.index(feat)
                original_value = X_base[0, feat_idx]
                
                # Determine if categorical or numerical
                if feat in preprocessors['categorical_cols']:
                    # For encoded categoricals, show as is
                    new_value = st.number_input(
                        f"{feat} (encoded)",
                        value=float(original_value),
                        step=1.0,
                        help="This is an encoded categorical feature"
                    )
                else:
                    # Numerical feature
                    # Get min/max from data
                    feat_min = float(shap_data['X_test_sample'][:, feat_idx].min())
                    feat_max = float(shap_data['X_test_sample'][:, feat_idx].max())
                    
                    new_value = st.slider(
                        feat,
                        min_value=feat_min,
                        max_value=feat_max,
                        value=float(original_value),
                        help=f"Original: {original_value:.2f}"
                    )
                
                adjusted_values[feat_idx] = new_value
            
            # Apply adjustments
            X_adjusted = X_base.copy()
            for feat_idx, new_val in adjusted_values.items():
                X_adjusted[0, feat_idx] = new_val
        
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
        for feat_idx, new_val in adjusted_values.items():
            feat_name = feature_names[feat_idx]
            old_val = X_base[0, feat_idx]
            if new_val != old_val:
                changes.append({
                    'Feature': feat_name,
                    'Original Value': f"{old_val:.2f}",
                    'New Value': f"{new_val:.2f}",
                    'Change': f"{new_val - old_val:+.2f}"
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
    A shallow decision tree trained to approximate the Random Forest's behavior. 
    This shows the key decision rules the model learned.
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
            if isinstance(X_sample, pd.DataFrame):
                X_sample = X_sample.values
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            main_idx = feature_names.index(main_feature)
            
            if interaction_feature == 'auto':
                shap.dependence_plot(
                    main_idx,
                    shap_vals,
                    X_sample,
                    feature_names=feature_names,
                    show=False
                )
            else:
                interaction_idx = feature_names.index(interaction_feature)
                shap.dependence_plot(
                    main_idx,
                    shap_vals,
                    X_sample,
                    feature_names=feature_names,
                    interaction_index=interaction_idx,
                    show=False
                )
            
            st.pyplot(fig)
            plt.close()
        except Exception as e:
            st.error(f"Error creating dependence plot: {e}")
            st.info("Please ensure SHAP data is properly computed.")
        
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
