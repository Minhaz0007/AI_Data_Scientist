"""
Feature Engineering Component
Advanced feature creation, selection, and transformation tools.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils.ml_engine import MLEngine


def render():
    """Render the Feature Engineering page."""
    if st.session_state['data'] is None:
        st.warning("Please load data first in the Data Ingestion page.")
        return

    df = st.session_state['data']
    ml = MLEngine()

    st.markdown("""
    Create new features, select important features, and transform your data for better model performance.
    """)

    # Tabs for different operations
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Polynomial Features",
        "Interaction Features",
        "DateTime Features",
        "Binning",
        "Feature Selection"
    ])

    with tab1:
        st.subheader("Polynomial Features")
        st.markdown("Create polynomial and interaction terms from numeric columns.")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_cols:
            st.warning("No numeric columns available.")
        else:
            cols_for_poly = st.multiselect(
                "Select Columns for Polynomial Features",
                options=numeric_cols,
                max_selections=5,
                help="Select up to 5 columns (more may cause memory issues)"
            )

            degree = st.slider("Polynomial Degree", 2, 4, 2)

            if cols_for_poly and st.button("Create Polynomial Features", key="poly_btn"):
                with st.spinner("Creating polynomial features..."):
                    try:
                        df_new = ml.create_polynomial_features(df, cols_for_poly, degree=degree)

                        new_cols = [c for c in df_new.columns if c not in df.columns]
                        st.success(f"Created {len(new_cols)} new polynomial features!")

                        st.write("**New Features Preview:**")
                        st.dataframe(df_new[new_cols].head(10), use_container_width=True)

                        if st.button("Apply Changes", key="apply_poly"):
                            st.session_state['data'] = df_new
                            st.success("Features added to dataset!")
                            st.rerun()

                    except Exception as e:
                        st.error(f"Error: {str(e)}")

    with tab2:
        st.subheader("Interaction Features")
        st.markdown("Create multiplication and division interactions between numeric columns.")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_cols:
            st.warning("No numeric columns available.")
        else:
            cols_for_interact = st.multiselect(
                "Select Columns for Interactions",
                options=numeric_cols,
                max_selections=5,
                help="Select columns to create interaction features"
            )

            if cols_for_interact and len(cols_for_interact) >= 2:
                if st.button("Create Interaction Features", key="interact_btn"):
                    with st.spinner("Creating interaction features..."):
                        try:
                            df_new = ml.create_interaction_features(df, cols_for_interact)

                            new_cols = [c for c in df_new.columns if c not in df.columns]
                            st.success(f"Created {len(new_cols)} new interaction features!")

                            st.write("**New Features Preview:**")
                            st.dataframe(df_new[new_cols].head(10), use_container_width=True)

                            if st.button("Apply Changes", key="apply_interact"):
                                st.session_state['data'] = df_new
                                st.success("Features added to dataset!")
                                st.rerun()

                        except Exception as e:
                            st.error(f"Error: {str(e)}")
            elif cols_for_interact:
                st.info("Select at least 2 columns for interactions.")

    with tab3:
        st.subheader("DateTime Feature Extraction")
        st.markdown("Extract useful features from datetime columns (year, month, day, weekday, etc.).")

        # Find potential datetime columns
        datetime_cols = []
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                datetime_cols.append(col)
            else:
                try:
                    pd.to_datetime(df[col].head(10), errors='raise')
                    datetime_cols.append(col)
                except:
                    pass

        if not datetime_cols:
            st.warning("No datetime columns detected. Try parsing a column first.")

            # Option to convert a column
            st.subheader("Convert Column to DateTime")
            all_cols = df.columns.tolist()
            col_to_convert = st.selectbox("Select Column to Convert", all_cols)

            if st.button("Convert to DateTime"):
                try:
                    df_new = df.copy()
                    df_new[col_to_convert] = pd.to_datetime(df_new[col_to_convert], errors='coerce')
                    st.session_state['data'] = df_new
                    st.success(f"Converted {col_to_convert} to datetime!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            datetime_col = st.selectbox(
                "Select DateTime Column",
                options=datetime_cols
            )

            if st.button("Extract DateTime Features", key="datetime_btn"):
                with st.spinner("Extracting datetime features..."):
                    try:
                        df_new = ml.create_datetime_features(df, datetime_col)

                        new_cols = [c for c in df_new.columns if c not in df.columns]
                        st.success(f"Created {len(new_cols)} new datetime features!")

                        st.write("**New Features:**")
                        for col in new_cols:
                            st.write(f"- {col}")

                        st.write("**Preview:**")
                        st.dataframe(df_new[new_cols].head(10), use_container_width=True)

                        if st.button("Apply Changes", key="apply_datetime"):
                            st.session_state['data'] = df_new
                            st.success("Features added to dataset!")
                            st.rerun()

                    except Exception as e:
                        st.error(f"Error: {str(e)}")

    with tab4:
        st.subheader("Feature Binning")
        st.markdown("Convert continuous variables into categorical bins.")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_cols:
            st.warning("No numeric columns available.")
        else:
            col_to_bin = st.selectbox(
                "Select Column to Bin",
                options=numeric_cols
            )

            col1, col2 = st.columns(2)

            with col1:
                n_bins = st.slider("Number of Bins", 2, 20, 5)

            with col2:
                strategy = st.selectbox(
                    "Binning Strategy",
                    options=['quantile', 'uniform'],
                    format_func=lambda x: {
                        'quantile': 'Quantile (Equal Frequency)',
                        'uniform': 'Uniform (Equal Width)'
                    }.get(x, x)
                )

            # Show current distribution
            fig = px.histogram(df, x=col_to_bin, nbins=30, title=f'Distribution of {col_to_bin}')
            st.plotly_chart(fig, use_container_width=True)

            if st.button("Create Binned Feature", key="bin_btn"):
                with st.spinner("Creating binned feature..."):
                    try:
                        df_new = ml.create_binned_features(df, col_to_bin, n_bins=n_bins, strategy=strategy)

                        new_col = f'{col_to_bin}_binned'
                        st.success(f"Created binned feature: {new_col}")

                        # Show bin distribution
                        bin_counts = df_new[new_col].value_counts().sort_index()
                        fig = px.bar(x=bin_counts.index, y=bin_counts.values,
                                    title=f'Bin Distribution for {new_col}',
                                    labels={'x': 'Bin', 'y': 'Count'})
                        st.plotly_chart(fig, use_container_width=True)

                        if st.button("Apply Changes", key="apply_bin"):
                            st.session_state['data'] = df_new
                            st.success("Feature added to dataset!")
                            st.rerun()

                    except Exception as e:
                        st.error(f"Error: {str(e)}")

    with tab5:
        st.subheader("Feature Selection")
        st.markdown("Identify the most important features for your prediction task.")

        all_cols = df.columns.tolist()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        col1, col2 = st.columns(2)

        with col1:
            feature_cols = st.multiselect(
                "Select Feature Columns (X)",
                options=all_cols,
                default=numeric_cols[:min(10, len(numeric_cols))]
            )

        with col2:
            remaining = [c for c in all_cols if c not in feature_cols]
            target_col = st.selectbox(
                "Select Target Column (y)",
                options=remaining if remaining else all_cols
            )

        col1, col2, col3 = st.columns(3)

        with col1:
            selection_method = st.selectbox(
                "Selection Method",
                options=['mutual_info', 'f_score', 'rfe'],
                format_func=lambda x: {
                    'mutual_info': 'Mutual Information',
                    'f_score': 'F-Score (ANOVA)',
                    'rfe': 'Recursive Feature Elimination'
                }.get(x, x)
            )

        with col2:
            k_features = st.slider(
                "Number of Features to Select",
                1, min(len(feature_cols), 20) if feature_cols else 10,
                min(5, len(feature_cols)) if feature_cols else 5
            )

        if feature_cols and target_col and st.button("Run Feature Selection", type="primary"):
            with st.spinner("Analyzing feature importance..."):
                try:
                    X = df[feature_cols]
                    y = df[target_col].dropna()

                    # Align indices
                    common_idx = X.index.intersection(y.index)
                    X = X.loc[common_idx]
                    y = y.loc[common_idx]

                    results = ml.select_features(X, y, method=selection_method, k=k_features)

                    st.success(f"Selected {len(results['selected_features'])} features using {selection_method}!")

                    # Feature Scores
                    st.subheader("Feature Scores")

                    scores_df = pd.DataFrame([
                        {'Feature': k, 'Score': v}
                        for k, v in results['feature_scores'].items()
                    ])

                    if selection_method == 'rfe':
                        # For RFE, lower ranking is better
                        scores_df = scores_df.sort_values('Score', ascending=True)
                        scores_df['Selected'] = scores_df['Feature'].isin(results['selected_features'])
                    else:
                        scores_df = scores_df.sort_values('Score', ascending=False)
                        scores_df['Selected'] = scores_df['Feature'].isin(results['selected_features'])

                    # Visualization
                    fig = px.bar(
                        scores_df,
                        x='Score',
                        y='Feature',
                        orientation='h',
                        color='Selected',
                        title='Feature Importance Scores',
                        color_discrete_map={True: 'green', False: 'gray'}
                    )
                    fig.update_layout(height=max(400, len(feature_cols) * 25))
                    st.plotly_chart(fig, use_container_width=True)

                    # Selected Features
                    st.subheader("Selected Features")
                    for i, f in enumerate(results['selected_features'], 1):
                        st.write(f"{i}. {f}")

                    # Option to create subset
                    if st.button("Create Dataset with Selected Features Only"):
                        selected_cols = results['selected_features'] + [target_col]
                        df_new = df[selected_cols].copy()
                        st.session_state['data'] = df_new
                        st.success("Dataset updated with selected features only!")
                        st.rerun()

                except Exception as e:
                    st.error(f"Error: {str(e)}")

    # Current Dataset Info
    st.markdown("---")
    st.subheader("Current Dataset Summary")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", len(df))
    col2.metric("Columns", len(df.columns))
    col3.metric("Numeric", len(df.select_dtypes(include=[np.number]).columns))
    col4.metric("Categorical", len(df.select_dtypes(include=['object', 'category']).columns))

    with st.expander("View All Columns"):
        for col in df.columns:
            dtype = str(df[col].dtype)
            missing = df[col].isnull().sum()
            st.write(f"- **{col}** ({dtype}) - {missing} missing values")
