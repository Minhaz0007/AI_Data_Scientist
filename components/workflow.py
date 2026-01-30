import streamlit as st
import pandas as pd
from utils.data_processor import auto_clean
from utils.ml_engine import ml_engine as ml

def render():
    st.header("Workflow Automation")
    st.markdown("Build and execute automated data pipelines.")

    if st.session_state['data'] is None:
        st.warning("Please upload a dataset first.")
        return

    df = st.session_state['data']

    st.subheader("Define Pipeline")

    steps = st.multiselect("Select Steps", ["Auto-Clean", "Feature Engineering (Date/Poly)", "Train Model (AutoML)"])

    if st.button("Run Pipeline"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        current_df = df.copy()
        total_steps = len(steps)
        completed_steps = 0

        # Step 1: Cleaning
        if "Auto-Clean" in steps:
            status_text.text("Running Auto-Clean...")
            current_df, log = auto_clean(current_df)
            st.success(f"Cleaned data: {len(log)} actions performed.")
            completed_steps += 1
            progress_bar.progress(int(completed_steps / total_steps * 100))

        # Step 2: Feature Engineering
        if "Feature Engineering (Date/Poly)" in steps:
            status_text.text("Running Feature Engineering...")
            # Simple logic: detect date cols, numeric cols
            numeric_cols = current_df.select_dtypes(include=['number']).columns.tolist()
            # Poly features for first 3 numerics
            if len(numeric_cols) > 0:
                # Drop rows with NaNs created by cleaning or original
                current_df = current_df.dropna()
                current_df = ml.create_polynomial_features(current_df, numeric_cols[:min(3, len(numeric_cols))])
            st.success(f"Features added. New shape: {current_df.shape}")
            completed_steps += 1
            progress_bar.progress(int(completed_steps / total_steps * 100))

        # Step 3: AutoML
        if "Train Model (AutoML)" in steps:
            status_text.text("Running AutoML...")
            # Assume last column is target for simplicity in auto-mode
            if not current_df.empty:
                target_col = current_df.columns[-1]
                st.info(f"Targeting column: {target_col}")

                # Drop rows with missing target
                current_df = current_df.dropna(subset=[target_col])

                X = current_df.drop(columns=[target_col])
                y = current_df[target_col]

                # Sample if too big
                if len(X) > 1000:
                    X = X.sample(1000, random_state=42)
                    y = y[X.index]

                results = ml.auto_ml(X, y, test_size=0.2, top_n=1)

                if results['best_model']:
                    st.success(f"Best Model: {results['best_model']['model_name']}")
                    score = results['best_model'].get('test_r2') or results['best_model'].get('test_accuracy')
                    st.metric("Score", round(score, 4) if score else 0)
                else:
                    st.error("AutoML failed.")
            else:
                st.error("Dataframe is empty.")

            completed_steps += 1
            progress_bar.progress(int(completed_steps / total_steps * 100))

        status_text.text("Pipeline Completed!")
        st.balloons()
