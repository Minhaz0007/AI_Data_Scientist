"""
Predictive Modeling Component
Supports Regression and Classification with multiple algorithms.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.ml_engine import MLEngine


def render():
    """Render the Predictive Modeling page."""
    if st.session_state['data'] is None:
        st.warning("Please load data first in the Data Ingestion page.")
        return

    df = st.session_state['data']
    ml = MLEngine()

    st.markdown("""
    Build and evaluate machine learning models for prediction tasks.
    Supports both **Regression** (continuous targets) and **Classification** (categorical targets).
    """)

    # Task Type Selection
    task_type = st.radio("Select Task Type", ["Regression", "Classification"], horizontal=True)

    st.markdown("---")

    # Column Selection
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Feature Selection")
        all_cols = df.columns.tolist()

        # Smart defaults - exclude likely target columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        feature_cols = st.multiselect(
            "Select Feature Columns (X)",
            options=all_cols,
            default=numeric_cols[:min(5, len(numeric_cols))],
            help="Select the columns to use as input features"
        )

    with col2:
        st.subheader("Target Selection")
        remaining_cols = [c for c in all_cols if c not in feature_cols]

        target_col = st.selectbox(
            "Select Target Column (y)",
            options=remaining_cols if remaining_cols else all_cols,
            help="Select the column to predict"
        )

    if not feature_cols or not target_col:
        st.info("Please select at least one feature and a target column.")
        return

    # Model Selection
    st.markdown("---")
    st.subheader("Model Configuration")

    col1, col2, col3 = st.columns(3)

    with col1:
        if task_type == "Regression":
            model_options = list(ml.get_regression_models().keys())
        else:
            model_options = list(ml.get_classification_models().keys())

        selected_model = st.selectbox("Select Model", model_options)

    with col2:
        test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)

    with col3:
        run_automl = st.checkbox("Run AutoML (Compare All Models)", value=False)

    # Training
    if st.button("Train Model", type="primary"):
        with st.spinner("Training model..."):
            X = df[feature_cols]
            y = df[target_col]

            # Check for issues
            if y.isnull().sum() > len(y) * 0.5:
                st.error("Target column has too many missing values (>50%).")
                return

            # Drop rows with missing target
            valid_mask = ~y.isnull()
            X = X[valid_mask]
            y = y[valid_mask]

            if run_automl:
                st.subheader("AutoML Results")
                results = ml.auto_ml(X, y, task=task_type.lower(), test_size=test_size)

                if results['best_model']:
                    # Model Comparison Table
                    comparison_data = []
                    for r in results['all_results']:
                        row = {'Model': r['model_name']}
                        if task_type == "Regression":
                            row['Test R²'] = r['test_r2']
                            row['Test RMSE'] = r['full_result']['metrics']['test_rmse']
                            row['CV R² (mean)'] = r['cv_r2']
                        else:
                            row['Test Accuracy'] = r['test_accuracy']
                            row['F1 Score'] = r['f1']
                            row['CV Accuracy (mean)'] = r['cv_accuracy']
                        comparison_data.append(row)

                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, use_container_width=True)

                    # Visualization
                    if task_type == "Regression":
                        fig = px.bar(comparison_df, x='Model', y='Test R²',
                                    title='Model Comparison (Test R²)',
                                    color='Test R²', color_continuous_scale='viridis')
                    else:
                        fig = px.bar(comparison_df, x='Model', y='Test Accuracy',
                                    title='Model Comparison (Test Accuracy)',
                                    color='Test Accuracy', color_continuous_scale='viridis')
                    st.plotly_chart(fig, use_container_width=True)

                    st.success(f"Best Model: **{results['best_model']['model_name']}**")

                    # Store best model result for detailed view
                    result = results['best_model']['full_result']
                else:
                    st.error("No models could be trained successfully.")
                    return

            else:
                # Train single model
                if task_type == "Regression":
                    result = ml.train_regression(X, y, model_name=selected_model, test_size=test_size)
                else:
                    result = ml.train_classification(X, y, model_name=selected_model, test_size=test_size)

                if 'error' in result:
                    st.error(result['error'])
                    return

            # Store in session
            st.session_state['trained_model'] = result

            # Display Results
            st.markdown("---")
            st.subheader(f"Model Results: {result['model_name']}")

            # Metrics
            metrics = result['metrics']
            if task_type == "Regression":
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Test R²", metrics['test_r2'])
                col2.metric("Test RMSE", metrics['test_rmse'])
                col3.metric("Test MAE", metrics['test_mae'])
                col4.metric("CV R² (±std)", f"{metrics['cv_r2_mean']} (±{metrics['cv_r2_std']})")

                # Actual vs Predicted Plot
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=result['predictions']['y_test'],
                    y=result['predictions']['y_pred_test'],
                    mode='markers',
                    name='Predictions',
                    marker=dict(color='blue', opacity=0.5)
                ))

                # Perfect prediction line
                min_val = min(result['predictions']['y_test'].min(), result['predictions']['y_pred_test'].min())
                max_val = max(result['predictions']['y_test'].max(), result['predictions']['y_pred_test'].max())
                fig.add_trace(go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(color='red', dash='dash')
                ))

                fig.update_layout(
                    title='Actual vs Predicted Values',
                    xaxis_title='Actual',
                    yaxis_title='Predicted'
                )
                st.plotly_chart(fig, use_container_width=True)

                # Residuals Plot
                residuals = result['predictions']['y_test'] - result['predictions']['y_pred_test']
                fig_resid = px.histogram(residuals, nbins=30, title='Residuals Distribution')
                fig_resid.update_layout(xaxis_title='Residual', yaxis_title='Count')
                st.plotly_chart(fig_resid, use_container_width=True)

            else:  # Classification
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Test Accuracy", metrics['test_accuracy'])
                col2.metric("Precision", metrics['precision'])
                col3.metric("Recall", metrics['recall'])
                col4.metric("F1 Score", metrics['f1'])

                if 'auc' in metrics:
                    st.metric("AUC-ROC", metrics['auc'])

                # Confusion Matrix
                st.subheader("Confusion Matrix")
                cm = result['confusion_matrix']
                class_names = result['class_names']

                fig_cm = px.imshow(
                    cm,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=[str(c) for c in class_names],
                    y=[str(c) for c in class_names],
                    color_continuous_scale='Blues',
                    text_auto=True
                )
                fig_cm.update_layout(title='Confusion Matrix')
                st.plotly_chart(fig_cm, use_container_width=True)

                # ROC Curve (for binary classification)
                if result['roc_data']:
                    roc = result['roc_data']
                    fig_roc = go.Figure()
                    fig_roc.add_trace(go.Scatter(
                        x=roc['fpr'], y=roc['tpr'],
                        mode='lines',
                        name=f"ROC (AUC = {roc['auc']})"
                    ))
                    fig_roc.add_trace(go.Scatter(
                        x=[0, 1], y=[0, 1],
                        mode='lines',
                        name='Random',
                        line=dict(dash='dash', color='gray')
                    ))
                    fig_roc.update_layout(
                        title='ROC Curve',
                        xaxis_title='False Positive Rate',
                        yaxis_title='True Positive Rate'
                    )
                    st.plotly_chart(fig_roc, use_container_width=True)

            # Feature Importance
            if result['feature_importance']:
                st.subheader("Feature Importance")

                fi_df = pd.DataFrame([
                    {'Feature': k, 'Importance': v}
                    for k, v in result['feature_importance'].items()
                ]).sort_values('Importance', ascending=True)

                fig_fi = px.bar(
                    fi_df.tail(15),
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title='Top 15 Feature Importances',
                    color='Importance',
                    color_continuous_scale='viridis'
                )
                st.plotly_chart(fig_fi, use_container_width=True)

            # Classification Report
            if task_type == "Classification":
                st.subheader("Classification Report")
                report = result['classification_report']
                report_df = pd.DataFrame(report).T
                st.dataframe(report_df.round(4), use_container_width=True)

    # Prediction on New Data
    st.markdown("---")
    st.subheader("Make Predictions")

    if 'trained_model' in st.session_state:
        st.info("Enter values for prediction:")

        model_result = st.session_state['trained_model']
        input_values = {}

        cols = st.columns(min(3, len(feature_cols)))
        for i, col_name in enumerate(feature_cols):
            with cols[i % 3]:
                if pd.api.types.is_numeric_dtype(df[col_name]):
                    input_values[col_name] = st.number_input(
                        col_name,
                        value=float(df[col_name].median()),
                        key=f"pred_{col_name}"
                    )
                else:
                    unique_vals = df[col_name].dropna().unique()
                    input_values[col_name] = st.selectbox(
                        col_name,
                        options=unique_vals,
                        key=f"pred_{col_name}"
                    )

        if st.button("Predict"):
            input_df = pd.DataFrame([input_values])
            input_encoded = pd.get_dummies(input_df, drop_first=True)

            # Align columns with training data
            for col in model_result['feature_names']:
                if col not in input_encoded.columns:
                    input_encoded[col] = 0
            input_encoded = input_encoded[model_result['feature_names']]

            # Scale
            input_scaled = model_result['scaler'].transform(input_encoded)

            # Predict
            prediction = model_result['model'].predict(input_scaled)[0]

            if task_type == "Classification" and model_result.get('label_encoder'):
                if hasattr(model_result['label_encoder'], 'inverse_transform'):
                    try:
                        prediction = model_result['label_encoder'].inverse_transform([int(prediction)])[0]
                    except:
                        pass

            st.success(f"Prediction: **{prediction}**")

            # Show probability for classification
            if task_type == "Classification" and hasattr(model_result['model'], 'predict_proba'):
                probs = model_result['model'].predict_proba(input_scaled)[0]
                st.write("Class Probabilities:")
                for i, prob in enumerate(probs):
                    st.write(f"  Class {i}: {prob:.4f}")
    else:
        st.info("Train a model first to make predictions.")
