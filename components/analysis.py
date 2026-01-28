import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.data_processor import perform_clustering, detect_outliers, treat_outliers, perform_hypothesis_test

def render():
    st.header("Analysis Engine")

    if st.session_state['data'] is None:
        st.warning("Please upload a dataset first.")
        return

    df = st.session_state['data']

    tab1, tab2, tab3 = st.tabs(["Clustering", "Outlier Detection", "Statistical Tests"])

    with tab1:
        render_clustering(df)

    with tab2:
        render_outlier_detection(df)

    with tab3:
        render_hypothesis_testing(df)

def render_clustering(df):
    st.subheader("K-Means Clustering")

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns for clustering.")
        return

    selected_cols = st.multiselect("Select Features for Clustering", numeric_cols, default=numeric_cols[:2])
    n_clusters = st.slider("Number of Clusters", 2, 10, 3)

    if st.button("Run Clustering"):
        if len(selected_cols) < 2:
            st.error("Please select at least 2 columns.")
        else:
            try:
                with st.spinner("Running clustering..."):
                    result_df = perform_clustering(df, selected_cols, n_clusters)
                    st.success("Clustering complete!")

                    # Visualize
                    st.subheader("Cluster Visualization")
                    if len(selected_cols) >= 2:
                        fig = px.scatter(
                            result_df,
                            x=selected_cols[0],
                            y=selected_cols[1],
                            color='cluster',
                            hover_data=selected_cols,
                            title=f"K-Means Clustering (k={n_clusters})"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    # Cluster statistics
                    st.subheader("Cluster Summary")
                    cluster_stats = result_df.groupby('cluster')[selected_cols].mean()
                    st.dataframe(cluster_stats)

                    # Cluster sizes
                    cluster_sizes = result_df['cluster'].value_counts().sort_index()
                    fig_sizes = px.bar(x=cluster_sizes.index, y=cluster_sizes.values,
                                        labels={'x': 'Cluster', 'y': 'Count'},
                                        title="Cluster Sizes")
                    st.plotly_chart(fig_sizes, use_container_width=True)

                    st.write("Clustered Data Preview:")
                    st.dataframe(result_df.head())

                    # Option to save
                    if st.checkbox("Append cluster labels to current dataset"):
                        st.session_state['data'] = result_df
                        st.success("Cluster labels added to dataset!")
            except Exception as e:
                st.error(f"Error during clustering: {e}")

def render_outlier_detection(df):
    st.subheader("Outlier Detection")

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

    if not numeric_cols:
        st.warning("No numeric columns available for outlier detection.")
        return

    col1, col2 = st.columns(2)
    with col1:
        selected_col = st.selectbox("Select Column", numeric_cols, key="outlier_col")
        method = st.selectbox("Detection Method", ["IQR (Interquartile Range)", "Z-Score"])
    with col2:
        if "IQR" in method:
            threshold = st.slider("IQR Multiplier", 1.0, 3.0, 1.5, 0.1)
            method_code = 'iqr'
        else:
            threshold = st.slider("Z-Score Threshold", 2.0, 4.0, 3.0, 0.1)
            method_code = 'zscore'

    if st.button("Detect Outliers"):
        try:
            result_df, stats = detect_outliers(df, selected_col, method_code, threshold)

            if "error" in stats:
                st.error(stats["error"])
            else:
                col1, col2, col3 = st.columns(3)
                col1.metric("Outliers Found", stats['outlier_count'])
                col2.metric("Percentage", f"{stats['outlier_percentage']}%")
                col3.metric("Method", stats['method'].upper())

                st.write(f"**Bounds:** Lower = {stats['lower_bound']}, Upper = {stats['upper_bound']}")

                # Visualization
                fig = go.Figure()
                fig.add_trace(go.Box(y=df[selected_col], name=selected_col, boxpoints='outliers'))
                fig.add_hline(y=stats['lower_bound'], line_dash="dash", line_color="red",
                              annotation_text="Lower Bound")
                fig.add_hline(y=stats['upper_bound'], line_dash="dash", line_color="red",
                              annotation_text="Upper Bound")
                fig.update_layout(title=f"Outlier Detection: {selected_col}")
                st.plotly_chart(fig, use_container_width=True)

                # Show outliers
                if stats['outlier_values']:
                    st.write("**Sample Outlier Values:**")
                    st.write(stats['outlier_values'][:10])

                # Treatment options
                st.markdown("---")
                st.subheader("Treat Outliers")
                treatment = st.selectbox("Treatment Method", [
                    "Clip (Cap at bounds)",
                    "Remove outlier rows",
                    "Replace with median"
                ])

                if st.button("Apply Treatment"):
                    treatment_map = {
                        "Clip (Cap at bounds)": "clip",
                        "Remove outlier rows": "remove",
                        "Replace with median": "median"
                    }
                    treated_df = treat_outliers(
                        df, selected_col,
                        treatment_map[treatment],
                        stats['lower_bound'],
                        stats['upper_bound']
                    )
                    st.session_state['data'] = treated_df
                    st.success(f"Outliers treated using '{treatment}' method!")
                    st.rerun()

        except Exception as e:
            st.error(f"Error detecting outliers: {e}")

def render_hypothesis_testing(df):
    st.subheader("Statistical Hypothesis Testing")

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    all_cols = df.columns.tolist()

    test_type = st.selectbox("Select Test", [
        "Normality Test (Shapiro-Wilk)",
        "T-Test (Compare Two Columns)",
        "Correlation Test (Pearson)",
        "Chi-Square Test (Categorical Independence)"
    ])

    if "Normality" in test_type:
        if not numeric_cols:
            st.warning("Need numeric columns for normality test.")
            return
        col1 = st.selectbox("Select Column", numeric_cols, key="norm_col")
        col2 = None
        test_code = 'normality'

    elif "T-Test" in test_type:
        if len(numeric_cols) < 2:
            st.warning("Need at least 2 numeric columns for t-test.")
            return
        col1 = st.selectbox("Column 1", numeric_cols, key="ttest_col1")
        col2 = st.selectbox("Column 2", [c for c in numeric_cols if c != col1], key="ttest_col2")
        test_code = 'ttest'

    elif "Correlation" in test_type:
        if len(numeric_cols) < 2:
            st.warning("Need at least 2 numeric columns for correlation test.")
            return
        col1 = st.selectbox("Column 1", numeric_cols, key="corr_col1")
        col2 = st.selectbox("Column 2", [c for c in numeric_cols if c != col1], key="corr_col2")
        test_code = 'correlation'

    elif "Chi-Square" in test_type:
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        # Also include numeric columns with few unique values
        categorical_cols += [c for c in numeric_cols if df[c].nunique() < 20]
        if len(categorical_cols) < 2:
            st.warning("Need at least 2 categorical columns for chi-square test.")
            return
        col1 = st.selectbox("Column 1", categorical_cols, key="chi_col1")
        col2 = st.selectbox("Column 2", [c for c in categorical_cols if c != col1], key="chi_col2")
        test_code = 'chi2'

    if st.button("Run Test"):
        try:
            results = perform_hypothesis_test(df, col1, col2, test_code)

            if "error" in results:
                st.error(results["error"])
            else:
                st.success(f"**{results['test']}**")

                # Display results
                cols = st.columns(3)
                if 'statistic' in results:
                    cols[0].metric("Test Statistic", results['statistic'])
                if 'correlation' in results:
                    cols[0].metric("Correlation", results['correlation'])
                cols[1].metric("P-Value", results['p_value'])
                cols[2].metric("Significant (p<0.05)", "Yes" if results['significant_at_05'] else "No")

                st.info(f"**Interpretation:** {results['interpretation']}")

                # Additional visualization for correlation
                if test_code == 'correlation' and col2:
                    fig = px.scatter(df, x=col1, y=col2, trendline="ols",
                                      title=f"Scatter Plot: {col1} vs {col2}")
                    st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error performing test: {e}")
